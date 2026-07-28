"""
Recursive tree builder that constructs a state-action-reaction trajectory tree
by querying an LLM at each expansion step.

The tree has the following node types in each cycle:

    state ──(robot action)──> state_robotaction
    state_robotaction ──(humans reaction)──> state_humansreaction
    state_humansreaction ──(consequence/observation)──> next_state  [with probability]

A *depth* of ``n_steps`` means the root state is expanded through ``n_steps``
full cycles of (robot-action, humans-reaction, consequence).
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from empo.simple_hierarchical_llm_modeler.llm_connector import LLMConnector
from empo.simple_hierarchical_llm_modeler.prompts import (
    batch_empowerment_prompt,
    consequences_prompt,
    empowerment_prompt,
    humans_reactions_prompt,
    robot_actions_prompt,
)

LOG = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\033\[[^m]*m")


def _truncate_ansi(line: str, width: int) -> str:
    """Truncate *line* (which may contain ANSI codes) to *width* visible columns."""
    parts = _ANSI_RE.split(line)
    codes = _ANSI_RE.findall(line)
    result: List[str] = []
    visible = 0
    for i, part in enumerate(parts):
        room = width - 1 - visible  # -1 for "\u2026"
        if len(part) > room:
            result.append(part[:max(room, 0)])
            result.append("\u2026\033[0m")
            return "".join(result)
        result.append(part)
        visible += len(part)
        if i < len(codes):
            result.append(codes[i])
    return "".join(result)


_FALLBACK_ACTION = [{"activity": "do nothing", "rationale": "fallback"}]
_FALLBACK_REACTION = [{"reaction": "do nothing", "rationale": "fallback"}]
_FALLBACK_CONSEQUENCE = [
    {
        "consequence": "nothing notable happens",
        "probability": 1.0,
        "rationale": "fallback",
    }
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class TreeNode:
    """A single node in the trajectory tree.

    Attributes:
        history: Ordered list of event descriptions leading to this node.
        node_type: One of ``"state"``, ``"robotaction"``, ``"humansreaction"``.
        depth: Current depth measured in full (robot-action, humans-reaction,
            consequence) cycles completed so far.
        children: List of ``(label, probability, child_node)`` triples.
            ``probability`` is 1.0 for robot-action and humans-reaction edges
            and equals the LLM-estimated probability for consequence edges.
        empowerment_estimate: For terminal-depth state nodes, the LLM-estimated
            number of meaningfully different futures.
    """

    history: List[str]
    node_type: str  # "state", "robotaction", "humansreaction"
    depth: int = 0
    children: List[Tuple[str, float, "TreeNode"]] = field(default_factory=list)
    empowerment_estimate: Optional[float] = None
    rationale: Optional[str] = None
    empowerment_rationale: Optional[str] = None

    # ANSI colour codes for terminal rendering
    _COLORS = {
        "state":          "\033[96m",   # light cyan/blue
        "robotaction":    "\033[37m",   # light grey
        "humansreaction": "\033[33;1m",   # bright/bold yellow
    }
    _RESET = "\033[0m"
    _BLINK_MARKER = "\033[5;91m██\033[0m"  # blinking red double-wide square

    def render(
        self,
        color: bool = True,
        root_label: str | None = None,
        active_node: "TreeNode | None" = None,
        blink_inline: bool = False,
        collapsed: "set[TreeNode] | None" = None,
    ) -> str:
        """Return a compact, colour-coded text representation of the tree.

        Args:
            color: If *False*, omit ANSI escape codes (plain text).
            root_label: Optional label to display for the root node instead
                of ``(root)``.
            active_node: If set, append a blinking red marker to this node.
            blink_inline: If *True*, the blinker appears at the end of the
                active node's label.  If *False* (default), it appears on a
                separate line below at the child-branch position.
            collapsed: Set of nodes whose children should be hidden.

        Returns:
            A multi-line string with ASCII-art connectors.
        """
        lines: List[str] = []
        self._render(lines, prefix="", connector="", is_last=True, color=color,
                     root_label=root_label, active_node=active_node,
                     blink_inline=blink_inline, collapsed=collapsed)
        return "\n".join(lines)

    def _render(
        self,
        lines: List[str],
        prefix: str,
        connector: str,
        is_last: bool,
        color: bool,
        root_label: str | None = None,
        active_node: "TreeNode | None" = None,
        blink_inline: bool = False,
        collapsed: "set[TreeNode] | None" = None,
    ) -> None:
        c = self._COLORS.get(self.node_type, "") if color else ""
        r = self._RESET if color and c else ""

        # Build the label for this node
        if self.node_type == "state":
            if root_label is not None and not self.history:
                label = root_label
            else:
                label = self.history[-1] if self.history else "(root)"
            if self.empowerment_estimate is not None:
                label += f" [emp≈{self.empowerment_estimate:.0f}]"
        else:
            # robotaction / humansreaction – label comes from the parent edge,
            # so just show the last history entry
            label = self.history[-1] if self.history else self.node_type

        lines.append(f"{prefix}{connector}{c}{label}{r}")

        # Prepare prefix for children
        if connector:
            child_prefix = prefix + ("  " if is_last else "│ ")
        else:
            child_prefix = prefix

        # If this node is collapsed, show indicator and skip children
        if collapsed and self in collapsed and self.children:
            lines[-1] += f" \033[90m⋯\033[0m"
            return

        # If this node is being expanded, show the blinker.
        # blink_inline=True  → blinker at end of label (empowerment estimation).
        # blink_inline=False → blinker on a separate line below (children incoming).
        if active_node is self:
            if blink_inline:
                lines[-1] += f" {self._BLINK_MARKER}"
            else:
                lines.append(f"{child_prefix}{self._BLINK_MARKER}")

        n = len(self.children)
        for i, (edge_label, prob, child) in enumerate(self.children):
            last_child = i == n - 1
            branch = "└─" if last_child else "├─"

            # For consequence edges, show probability on the connector line
            if self.node_type == "humansreaction":
                edge_info = f"(p={prob:.2f}) "
            else:
                edge_info = ""

            child._render(
                lines,
                prefix=child_prefix,
                connector=f"{branch}{edge_info}",
                is_last=last_child,
                color=color,
                root_label=None,
                active_node=active_node,
                blink_inline=blink_inline,
                collapsed=collapsed,
            )

    # ---- serialisation / export ------------------------------------------

    def to_dict(self, root_label: str | None = None) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of the full tree.

        The dictionary includes rationales and empowerment details so that
        the exported file is self-contained and human-readable.
        """
        # Label
        if self.node_type == "state":
            if root_label is not None and not self.history:
                label = root_label
            else:
                label = self.history[-1] if self.history else "(root)"
        else:
            label = self.history[-1] if self.history else self.node_type

        d: Dict[str, Any] = {
            "label": label,
            "node_type": self.node_type,
            "depth": self.depth,
        }
        if self.rationale:
            d["rationale"] = self.rationale
        if self.empowerment_estimate is not None:
            d["empowerment_estimate"] = self.empowerment_estimate
        if self.empowerment_rationale:
            d["empowerment_rationale"] = self.empowerment_rationale
        if self.children:
            d["children"] = []
            for edge_label, prob, child in self.children:
                entry: Dict[str, Any] = {"edge_label": edge_label}
                if self.node_type == "humansreaction":
                    entry["probability"] = prob
                entry["node"] = child.to_dict()
                d["children"].append(entry)
        return d

    def export(
        self,
        path: str,
        fmt: str = "json",
        root_label: str | None = None,
    ) -> None:
        """Export the tree to a file.

        Args:
            path: Output file path.
            fmt: ``"json"`` (default), ``"yaml"``, or ``"html"``.
            root_label: Optional label for the root node.
        """
        data = self.to_dict(root_label=root_label)

        if fmt == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif fmt == "yaml":
            try:
                import yaml
            except ImportError:
                raise ImportError("PyYAML is required for YAML export: pip install pyyaml")
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        elif fmt == "html":
            html = _tree_to_html(data)
            with open(path, "w") as f:
                f.write(html)
        else:
            raise ValueError(f"Unknown export format: {fmt!r}. Use 'json', 'yaml', or 'html'.")


def _node_html(node: Dict[str, Any], prob: float | None = None) -> str:
    """Render a single tree node (and its children) as nested HTML."""
    import html as html_mod

    label = html_mod.escape(node["label"])
    ntype = node["node_type"]
    color = {"state": "#5ec4c4", "robotaction": "#b0b0b0", "humansreaction": "#e6c84c"}.get(ntype, "#ccc")

    # Probability prefix for consequence (state) edges, matching terminal output
    prob_prefix = f"(p={prob:.2f}) " if prob is not None else ""

    parts = [f'<span style="color:{color};font-weight:bold">{prob_prefix}{label}</span>']
    if "empowerment_estimate" in node:
        parts.append(f' <span style="color:#8f8">[emp\u2248{node["empowerment_estimate"]:.0f}]</span>')
    if "rationale" in node:
        parts.append(f'<div style="margin-left:1em;color:#999;font-size:0.85em">\u2192 {html_mod.escape(node["rationale"])}</div>')
    if "empowerment_rationale" in node:
        parts.append(f'<div style="margin-left:1em;color:#9d9;font-size:0.85em">\u2192 {html_mod.escape(node["empowerment_rationale"])}</div>')

    children = node.get("children", [])
    if children:
        child_items = []
        for ch in children:
            ch_prob = ch.get("probability")
            child_items.append(_node_html(ch["node"], ch_prob))
        inner = "\n".join(f"<li>{c}</li>" for c in child_items)
        return (
            f"<details open><summary>{''.join(parts)}</summary>"
            f"<ul>{inner}</ul></details>"
        )
    else:
        return f"{''.join(parts)}"


_HTML_HEADER = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<title>Tree Export</title>"
    "<style>"
    "body{font-family:monospace;background:#1e1e1e;color:#ddd;padding:1em}"
    "h2{color:#fff;border-bottom:1px solid #555;padding-bottom:0.3em}"
    "details{margin-left:1.5em}"
    "summary{cursor:pointer;list-style:disclosure-closed}"
    "details[open]>summary{list-style:disclosure-open}"
    "ul{list-style:none;padding-left:1em}"
    "li{margin:0.2em 0}"
    "</style></head><body>"
)
_HTML_FOOTER = "</body></html>"


def _tree_to_html(data: Dict[str, Any]) -> str:
    """Generate a self-contained expandable HTML page from a tree dict."""
    body = _node_html(data)
    return f"{_HTML_HEADER}{body}{_HTML_FOOTER}"


def _trees_to_html(
    sections: List[Tuple[str, Dict[str, Any]]],
) -> str:
    """Generate an HTML page with multiple labelled tree sections.

    Args:
        sections: List of ``(heading, tree_dict)`` pairs.
    """
    parts = [_HTML_HEADER]
    for heading, data in sections:
        import html as html_mod
        parts.append(f"<h2>{html_mod.escape(heading)}</h2>")
        parts.append(_node_html(data))
    parts.append(_HTML_FOOTER)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Live in-place renderer
# ---------------------------------------------------------------------------


class LiveTreeRenderer:
    """Renders a growing tree in-place in the terminal.

    The tree is drawn starting from the top of the screen.  A dashed
    separator and a rolling status section (at most *status_lines* high)
    are kept at the bottom so the user can follow progress without
    duplicating information already visible in the tree.

    Usage::

        renderer = LiveTreeRenderer(root_label="My scenario")
        build_tree(llm, ..., on_update=renderer.update)
        renderer.finish()
    """

    _STATUS_LINES = 5

    # Concise query-type labels (no tree content repeated)
    _QUERY_LABELS = {
        "state":          "Querying robot actions",
        "robotaction":    "Querying humans' reactions",
        "humansreaction": "Querying consequences",
    }

    def __init__(
        self,
        root: TreeNode | None = None,
        root_label: str | None = None,
        context_lines: List[str] | None = None,
        stream: Any = None,
    ) -> None:
        self.root = root
        self.root_label = root_label
        self.context_lines = context_lines or []
        self._stream = stream or sys.stderr
        self._status: Deque[str] = collections.deque(maxlen=self._STATUS_LINES)
        self._started = False
        self._blink_inline = False
        self._node_count = 0
        self._t0 = time.monotonic()

    # ---- internal helpers ------------------------------------------------

    def _term_size(self) -> tuple[int, int]:
        """Return ``(columns, rows)``."""
        try:
            sz = os.get_terminal_size(self._stream.fileno())
            return sz.columns, sz.lines
        except (AttributeError, ValueError, OSError):
            return 120, 40

    def _build_tree_lines(
        self,
        active_node: TreeNode | None,
        cols: int,
        collapsed: "set[TreeNode] | None" = None,
    ) -> List[str]:
        """Render tree to a list of truncated, ANSI-decorated lines."""
        if self.root is None:
            return []

        text = self.root.render(
            color=True, root_label=self.root_label, active_node=active_node,
            blink_inline=self._blink_inline, collapsed=collapsed,
        )
        blink = TreeNode._BLINK_MARKER
        emp_re = re.compile(r" \[emp≈\d+\]")
        tree_lines = []
        for ln in text.split("\n"):
            plain = _ANSI_RE.sub("", ln)
            emp_match = emp_re.search(plain)
            has_blink = blink in ln

            # Compute suffix to always preserve at end of line
            suffix = ""
            suffix_vis = 0
            if emp_match:
                suffix += emp_match.group()
                suffix_vis += len(emp_match.group())
            if has_blink:
                suffix += f" {blink}"
                suffix_vis += 3  # " ██" = 3 visible chars

            if suffix:
                trunc = _truncate_ansi(ln, cols - suffix_vis)
                # Strip suffix parts that survived truncation
                trunc_plain = _ANSI_RE.sub("", trunc)
                if emp_match and emp_match.group() in trunc_plain:
                    # emp tag survived — remove it so we don't double it
                    trunc = trunc.replace(emp_match.group(), "", 1)
                if has_blink and blink in trunc:
                    trunc = trunc.replace(f" {blink}", "", 1)
                # Strip trailing reset before appending suffix
                if trunc.endswith("\033[0m"):
                    trunc = trunc[:-4]
                trunc += f"\033[0m{suffix}"
                tree_lines.append(trunc)
            else:
                tree_lines.append(_truncate_ansi(ln, cols))
        return tree_lines

    @staticmethod
    def _ancestors_of(root: TreeNode, target: TreeNode | None) -> "set[TreeNode]":
        """Return the set of all ancestors of *target* (inclusive)."""
        if target is None:
            return set()
        result: set[TreeNode] = set()

        def _find(node: TreeNode, path: List[TreeNode]) -> bool:
            path.append(node)
            if node is target:
                result.update(path)
                return True
            for _, _, child in node.children:
                if _find(child, path):
                    return True
            path.pop()
            return False

        _find(root, [])
        return result

    @staticmethod
    def _subtree_size(node: TreeNode) -> int:
        """Count all nodes in the subtree rooted at *node*."""
        total = 1
        for _, _, child in node.children:
            total += LiveTreeRenderer._subtree_size(child)
        return total

    @staticmethod
    def _collapsible_by_size(
        root: TreeNode, ancestors: "set[TreeNode]",
    ) -> "List[TreeNode]":
        """Return all non-ancestor internal nodes, sorted smallest-subtree-first.

        This ensures we always collapse the branch that removes the fewest
        lines, preventing large top-level branches from being collapsed
        when smaller sub-branches would suffice.
        """
        result: List[TreeNode] = []

        def _walk(node: TreeNode) -> None:
            for _, _, child in node.children:
                if child in ancestors:
                    _walk(child)
                elif child.children:
                    # Collect this node AND recurse into it for finer targets
                    result.append(child)
                    _walk(child)

        _walk(root)
        result.sort(key=LiveTreeRenderer._subtree_size)
        return result

    def _paint(self, active_node: TreeNode | None) -> None:
        """Redraw the full screen: tree at top, status at bottom."""
        cols, rows = self._term_size()

        # Context lines (higher-level path) shown in magenta/pink above the tree
        ctx_lines = []
        for cl in self.context_lines:
            ctx_lines.append(_truncate_ansi(f"\033[35m{cl}\033[0m", cols))

        # Reserve space: context lines + 1 dashed separator + _STATUS_LINES
        status_height = 1 + self._STATUS_LINES
        tree_area = rows - status_height - len(ctx_lines)

        # First pass: render tree without collapsing
        tree_lines = self._build_tree_lines(active_node, cols)

        # Auto-collapse completed subtrees top-to-bottom if tree overflows
        if len(tree_lines) > tree_area and self.root is not None:
            ancestors = self._ancestors_of(self.root, active_node)
            collapsible = self._collapsible_by_size(self.root, ancestors)
            collapsed: set[TreeNode] = set()
            for node in collapsible:
                collapsed.add(node)
                tree_lines = self._build_tree_lines(active_node, cols, collapsed)
                if len(tree_lines) <= tree_area:
                    break

        # Truncate / pad tree lines to fit
        shown_tree = tree_lines[:tree_area]

        w = self._stream.write

        # Reset scroll region to full screen so we can paint everywhere
        w(f"\033[1;{rows}r")
        # Home cursor (top-left)
        w("\033[H")

        # Context lines (higher-level path in pink)
        for cl in ctx_lines:
            w(f"\033[2K{cl}\n")

        for line in shown_tree:
            w(f"\033[2K{line}\n")
        # Clear remaining tree area
        for _ in range(tree_area - len(shown_tree)):
            w("\033[2K\n")

        # Dashed separator
        w(f"\033[2K\033[90m{'─' * cols}\033[0m\n")

        # Status lines (pad to exactly _STATUS_LINES).
        # The last line must NOT end with \n – otherwise the cursor
        # would land on row (rows+1), scrolling the whole screen by
        # one line and pushing the root node off the top.
        status_snapshot = list(self._status)
        for i in range(self._STATUS_LINES):
            nl = "\n" if i < self._STATUS_LINES - 1 else ""
            if i < len(status_snapshot):
                w(f"\033[2K{_truncate_ansi(status_snapshot[i], cols)}{nl}")
            else:
                w(f"\033[2K{nl}")

        # Set scroll region to status area only so any stray output
        # (e.g. from l2p logging) stays contained and doesn't push
        # the tree off screen.
        status_top = rows - self._STATUS_LINES + 1  # 1-indexed
        w(f"\033[{status_top};{rows}r")
        w(f"\033[{rows};{cols}H")  # park cursor at bottom-right

        self._stream.flush()

    # ---- public API ------------------------------------------------------

    def log(self, message: str) -> None:
        """Append a line to the rolling status section."""
        self._status.append(message)

    def update(self, active_node: TreeNode | None = None, message: str = "") -> None:
        """Re-render the tree, marking *active_node* with a blinking marker."""
        if self.root is None:
            if active_node is not None:
                self.root = active_node
            else:
                return

        # Empowerment estimation → blinker inline at end of label.
        # Everything else (querying actions/reactions/consequences) → below.
        self._blink_inline = (message == "Estimating empowerment")

        if not self._started:
            # Switch to alternate screen buffer & hide cursor
            self._stream.write("\033[?1049h\033[?25l\033[2J")
            self._started = True

        # Generate a concise status message
        if active_node is not None:
            self._node_count += 1
            elapsed = time.monotonic() - self._t0
            label = message or self._QUERY_LABELS.get(active_node.node_type, "Expanding")
            self.log(
                f"[{elapsed:5.1f}s] #{self._node_count:>3d}  {label} (depth {active_node.depth})"
            )

        self._paint(active_node)

    def finish(self) -> None:
        """Do a final render without any active marker."""
        elapsed = time.monotonic() - self._t0
        nodes = count_nodes(self.root) if self.root else 0
        leaves = len(collect_leaves(self.root)) if self.root else 0
        self.log(f"[{elapsed:5.1f}s] Done — {nodes} nodes, {leaves} leaves.")
        self._paint(active_node=None)

    def close(self) -> None:
        """Reset scroll region, show cursor, leave alternate screen."""
        cols, rows = self._term_size()
        w = self._stream.write
        w(f"\033[1;{rows}r")     # reset scroll region
        w("\033[?25h")           # show cursor
        w("\033[?1049l")         # leave alternate screen buffer
        self._stream.flush()
        self._started = False


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_list(text: str) -> List[Dict[str, Any]]:
    """Robustly extract the first JSON list from *text*."""
    # Try to find JSON array in the text
    start = text.find("[")
    if start == -1:
        raise ValueError(f"No JSON list found in LLM response: {text!r}")
    # Find matching closing bracket
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    raise ValueError(f"Could not parse JSON list from LLM response: {text!r}")


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Robustly extract the first JSON object from *text*."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    raise ValueError(f"Could not parse JSON object from LLM response: {text!r}")


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


def build_tree(
    llm: LLMConnector,
    initial_state_description: str,
    n_steps: int = 2,
    n_robotactions: int = 3,
    n_humansreactions: int = 3,
    n_consequences: int = 2,
    higher_level_context: Optional[str] = None,
    time_horizon: Optional[str] = None,
    on_update: Optional[Callable[[Optional[TreeNode], str], None]] = None,
) -> TreeNode:
    """Build a trajectory tree by recursively querying the LLM.

    Args:
        llm: LLM connector for making queries.
        initial_state_description: Natural-language description of the
            starting situation.
        n_steps: Maximum number of full (action, reaction, consequence)
            cycles to expand.
        n_robotactions: Number of distinct robot actions per state.
        n_humansreactions: Number of distinct human reactions per robot action.
        n_consequences: Number of distinct consequences per human reaction.
        higher_level_context: Optional higher-level context text for
            hierarchical mode.
        time_horizon: Optional time-horizon label (e.g. ``"the next hour"``).
        on_update: Optional callback invoked with the node currently being
            expanded (or ``None`` when expansion finishes). Used by
            :class:`LiveTreeRenderer` for in-place display.

    Returns:
        The root :class:`TreeNode` of the constructed tree.
    """
    root = TreeNode(history=[], node_type="state", depth=0)
    _expand_state(
        llm=llm,
        node=root,
        initial_state=initial_state_description,
        n_steps=n_steps,
        n_robotactions=n_robotactions,
        n_humansreactions=n_humansreactions,
        n_consequences=n_consequences,
        higher_level_context=higher_level_context,
        time_horizon=time_horizon,
        on_update=on_update,
    )
    # Batch-estimate empowerment for all terminal nodes
    terminals = collect_leaves(root)
    _batch_estimate_empowerment(
        llm=llm,
        terminals=terminals,
        initial_state=initial_state_description,
        higher_level_context=higher_level_context,
        on_update=on_update,
    )
    if on_update is not None:
        on_update(None, "")
    return root


def _expand_state(
    llm: LLMConnector,
    node: TreeNode,
    initial_state: str,
    n_steps: int,
    n_robotactions: int,
    n_humansreactions: int,
    n_consequences: int,
    higher_level_context: Optional[str],
    time_horizon: Optional[str] = None,
    on_update: Optional[Callable[[Optional[TreeNode], str], None]] = None,
) -> None:
    """Expand a *state* node by generating robot actions (or terminal estimates)."""
    assert node.node_type == "state"

    if node.depth >= n_steps:
        # Terminal depth – empowerment will be estimated in batch later
        return

    # Ask for robot actions
    if on_update is not None:
        on_update(node, "Querying robot actions")
    prompt = robot_actions_prompt(
        initial_state, node.history, n_robotactions, higher_level_context,
        time_horizon,
    )
    raw = llm.query(prompt)
    try:
        actions = _parse_json_list(raw)
    except ValueError:
        LOG.warning("Failed to parse robot actions; creating single fallback action")
        actions = list(_FALLBACK_ACTION)

    if not actions:
        LOG.warning(
            "Parsed robot actions but received empty list; creating single fallback action"
        )
        actions = list(_FALLBACK_ACTION)

    # Create ALL child nodes first so the tree shows full breadth
    for act in actions:
        action_desc = act.get("activity", act.get("action", "unknown action"))
        child = TreeNode(
            history=node.history + [f"Robot: {action_desc}"],
            node_type="robotaction",
            depth=node.depth,
            rationale=act.get("rationale"),
        )
        node.children.append((action_desc, 1.0, child))

    # Then expand each child
    for _, _, child in node.children:
        _expand_robotaction(
            llm=llm,
            node=child,
            initial_state=initial_state,
            n_steps=n_steps,
            n_robotactions=n_robotactions,
            n_humansreactions=n_humansreactions,
            n_consequences=n_consequences,
            higher_level_context=higher_level_context,
            time_horizon=time_horizon,
            on_update=on_update,
        )


def _expand_robotaction(
    llm: LLMConnector,
    node: TreeNode,
    initial_state: str,
    n_steps: int,
    n_robotactions: int,
    n_humansreactions: int,
    n_consequences: int,
    higher_level_context: Optional[str],
    time_horizon: Optional[str] = None,
    on_update: Optional[Callable[[Optional[TreeNode], str], None]] = None,
) -> None:
    """Expand a *robotaction* node by generating humans' reactions."""
    assert node.node_type == "robotaction"

    if on_update is not None:
        on_update(node, "Querying humans' reactions")

    prompt = humans_reactions_prompt(
        initial_state, node.history, n_humansreactions, higher_level_context,
        time_horizon,
    )
    raw = llm.query(prompt)
    try:
        reactions = _parse_json_list(raw)
    except ValueError:
        LOG.warning("Failed to parse human reactions; creating single fallback")
        reactions = list(_FALLBACK_REACTION)

    if not reactions:
        LOG.warning(
            "Parsed human reactions but received empty list; creating single fallback"
        )
        reactions = list(_FALLBACK_REACTION)

    # Create ALL child nodes first so the tree shows full breadth
    for react in reactions:
        reaction_desc = react.get("reaction", "unknown reaction")
        child = TreeNode(
            history=node.history + [f"Humans: {reaction_desc}"],
            node_type="humansreaction",
            depth=node.depth,
            rationale=react.get("rationale"),
        )
        node.children.append((reaction_desc, 1.0, child))

    # Then expand each child
    for _, _, child in node.children:
        _expand_humansreaction(
            llm=llm,
            node=child,
            initial_state=initial_state,
            n_steps=n_steps,
            n_consequences=n_consequences,
            higher_level_context=higher_level_context,
            n_robotactions=n_robotactions,
            n_humansreactions=n_humansreactions,
            time_horizon=time_horizon,
            on_update=on_update,
        )


def _expand_humansreaction(
    llm: LLMConnector,
    node: TreeNode,
    initial_state: str,
    n_steps: int,
    n_consequences: int,
    higher_level_context: Optional[str],
    n_robotactions: int,
    n_humansreactions: int,
    time_horizon: Optional[str] = None,
    on_update: Optional[Callable[[Optional[TreeNode], str], None]] = None,
) -> None:
    """Expand a *humansreaction* node by generating probabilistic consequences."""
    assert node.node_type == "humansreaction"

    if on_update is not None:
        on_update(node, "Querying consequences")

    prompt = consequences_prompt(
        initial_state, node.history, n_consequences, higher_level_context,
        time_horizon,
    )
    raw = llm.query(prompt)
    try:
        consequences = _parse_json_list(raw)
    except ValueError:
        LOG.warning("Failed to parse consequences; creating single fallback")
        consequences = list(_FALLBACK_CONSEQUENCE)

    if not consequences:
        LOG.warning(
            "Parsed consequences but received empty list; creating single fallback"
        )
        consequences = list(_FALLBACK_CONSEQUENCE)

    # Normalize probabilities so they sum to 1
    probs = [float(c.get("probability", 1.0)) for c in consequences]
    total = sum(probs)
    if total <= 0:
        probs = [1.0 / len(consequences)] * len(consequences)
    else:
        probs = [p / total for p in probs]

    # Create ALL child nodes first so the tree shows full breadth
    for cons, prob in zip(consequences, probs):
        cons_desc = cons.get("consequence", "unknown consequence")
        child = TreeNode(
            history=node.history + [f"Obs: {cons_desc}"],
            node_type="state",
            depth=node.depth + 1,
            rationale=cons.get("rationale"),
        )
        node.children.append((cons_desc, prob, child))

    # Then expand each child
    for _, _, child in node.children:
        _expand_state(
            llm=llm,
            node=child,
            initial_state=initial_state,
            n_steps=n_steps,
            n_robotactions=n_robotactions,
            n_humansreactions=n_humansreactions,
            n_consequences=n_consequences,
            higher_level_context=higher_level_context,
            time_horizon=time_horizon,
            on_update=on_update,
        )


# ---------------------------------------------------------------------------
# Batched empowerment estimation
# ---------------------------------------------------------------------------

# Rough token estimate: ~4 chars per token
_CHARS_PER_TOKEN = 4


def _batch_estimate_empowerment(
    llm: LLMConnector,
    terminals: List[TreeNode],
    initial_state: str,
    higher_level_context: Optional[str],
    on_update: Optional[Callable[[Optional[TreeNode], str], None]] = None,
) -> None:
    """Estimate empowerment for all terminal nodes using batched prompts.

    Groups terminals into batches sized to fit the LLM's context window,
    sends one prompt per batch, and parses the list of results back onto
    the nodes.  This ensures cross-consistent scoring within each batch.
    """
    if not terminals:
        return

    ctx_len = getattr(llm, "context_length", 4096)
    # Reserve tokens for the response (~200 tokens per terminal for JSON)
    response_reserve = 200 * len(terminals)
    # Available tokens for prompt content
    available = max(ctx_len - response_reserve, ctx_len // 2)
    available_chars = available * _CHARS_PER_TOKEN

    # Estimate per-terminal cost: history lines
    def _terminal_chars(node: TreeNode) -> int:
        return sum(len(line) for line in node.history) + 40 * len(node.history)

    # Build the shared preamble size (situation + instructions)
    preamble_probe = batch_empowerment_prompt(
        initial_state, [[]], higher_level_context
    )
    preamble_chars = len(preamble_probe)

    # Partition terminals into batches that fit the context window
    batches: List[List[TreeNode]] = []
    current_batch: List[TreeNode] = []
    current_chars = preamble_chars
    for node in terminals:
        node_chars = _terminal_chars(node)
        if current_batch and current_chars + node_chars > available_chars:
            batches.append(current_batch)
            current_batch = []
            current_chars = preamble_chars
        current_batch.append(node)
        current_chars += node_chars
    if current_batch:
        batches.append(current_batch)

    # Process each batch, collecting reference examples from the first
    reference_examples: List[dict] = []
    for batch_idx, batch in enumerate(batches):
        if on_update is not None:
            on_update(batch[0], f"Estimating empowerment (batch of {len(batch)})")

        histories = [node.history for node in batch]
        prompt = batch_empowerment_prompt(
            initial_state, histories, higher_level_context,
            reference_examples=reference_examples if batch_idx > 0 else None,
        )
        raw = llm.query(prompt)

        try:
            results = _parse_json_list(raw)
        except ValueError:
            LOG.warning(
                "Failed to parse batch empowerment response; "
                "falling back to individual estimates"
            )
            results = []

        # Assign results back to nodes
        for i, node in enumerate(batch):
            if i < len(results):
                try:
                    node.empowerment_estimate = float(
                        results[i].get("estimate", 1.0)
                    )
                    node.empowerment_rationale = results[i].get("rationale")
                except (ValueError, TypeError):
                    LOG.warning(
                        "Bad empowerment entry at index %d; defaulting to 1.0", i
                    )
                    node.empowerment_estimate = 1.0
            else:
                LOG.warning(
                    "Batch response missing entry %d; defaulting to 1.0", i
                )
                node.empowerment_estimate = 1.0

        # After the first batch, collect ~20% of results as reference examples
        if batch_idx == 0 and len(batches) > 1 and results:
            n_ref = max(1, len(results) // 5)
            # Pick evenly spaced examples for diverse calibration
            step = max(1, len(results) // n_ref)
            for k in range(0, len(results), step):
                if len(reference_examples) >= n_ref:
                    break
                reference_examples.append({
                    "history": batch[k].history,
                    "result": results[k],
                })

        # Update display for each node that now has an estimate
        if on_update is not None:
            for node in batch:
                on_update(node, "Estimating empowerment")


# ---------------------------------------------------------------------------
# Utility: count nodes
# ---------------------------------------------------------------------------


def count_nodes(root: TreeNode) -> int:
    """Return the total number of nodes in the tree rooted at *root*."""
    total = 1
    for _, _, child in root.children:
        total += count_nodes(child)
    return total


def collect_leaves(root: TreeNode) -> List[TreeNode]:
    """Return all leaf nodes (no children) in the tree."""
    if not root.children:
        return [root]
    leaves: List[TreeNode] = []
    for _, _, child in root.children:
        leaves.extend(collect_leaves(child))
    return leaves
