"""
LLM connector interface for the simple hierarchical LLM modeler.

Defines a lightweight protocol for LLM interaction that can wrap the L2P
BaseLLM or any other LLM backend. Each call uses a fresh context (no
conversation history), as required by the tree-building algorithm.

:class:`CachedLLMConnector` wraps any connector with a disk-backed cache
so that identical prompts are not re-queried across restarts.

:class:`StatsTrackingLLM` wraps any connector to collect per-query-type
token usage statistics, distinguishing reasoning from response tokens.
"""

import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

LOG = logging.getLogger(__name__)


@runtime_checkable
class LLMConnector(Protocol):
    """Protocol for querying an LLM with a single prompt in a fresh context.

    Implementations may wrap the L2P ``BaseLLM``, call an API directly via
    ``ollama`` / ``openai`` / ``anthropic``, or provide canned responses for
    testing.
    """

    @property
    def context_length(self) -> int:
        """Return the model's context window size in tokens."""
        ...

    def query(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the raw text response.

        Each call MUST use a fresh context (no conversation history from
        previous calls).
        """
        ...


class L2PConnector:
    """Wraps a vendored L2P ``BaseLLM`` instance to satisfy :class:`LLMConnector`."""

    def __init__(self, base_llm) -> None:
        self._llm = base_llm

    @property
    def context_length(self) -> int:
        return getattr(self._llm, "context_length", 4096)

    def query(self, prompt: str) -> str:  # noqa: D401
        return self._llm.query(prompt)


# Model-to-context-length mapping for Anthropic models (conservative defaults).
_ANTHROPIC_CONTEXT_LENGTHS: Dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-3-7-sonnet-20250219": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
}


class AnthropicConnector:
    """Native Anthropic API connector satisfying :class:`LLMConnector`.

    Requires the ``anthropic`` package (``pip install anthropic``).

    Usage::

        llm = AnthropicConnector(
            model="claude-sonnet-4-20250514",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        tree = build_tree(llm, ...)

    The connector also exposes a ``query_log`` attribute compatible with
    :class:`StatsTrackingLLM` so that real token counts are recorded.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        context_length: Optional[int] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicConnector: "
                "pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._context_length = (
            context_length
            or _ANTHROPIC_CONTEXT_LENGTHS.get(model, 200_000)
        )
        self.query_log: List[Dict[str, Any]] = []

    @property
    def context_length(self) -> int:
        return self._context_length

    def query(self, prompt: str) -> str:  # noqa: D401
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from the response
        text = "".join(
            block.text for block in message.content if hasattr(block, "text")
        )
        # Log token usage for StatsTrackingLLM
        self.query_log.append({
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "reasoning_tokens": 0,
        })
        return text


class CachedLLMConnector:
    """Disk-backed caching wrapper around any :class:`LLMConnector`.

    Responses are stored as individual JSON files in *cache_dir*, keyed by a
    SHA-256 hash of the prompt text.  On a cache hit the LLM is not called,
    saving tokens and latency.  New responses are written to disk immediately
    so that even a partial run (e.g. interrupted by Ctrl-C) preserves its
    progress.

    Usage::

        raw_llm = OPENAI(model="gemini-2.5-flash", ...)
        llm = CachedLLMConnector(raw_llm, cache_dir="outputs/llm_cache")
        tree = build_tree(llm, ...)
    """

    def __init__(
        self,
        inner: LLMConnector,
        cache_dir: str | os.PathLike = "outputs/llm_cache",
    ) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    # ── public helpers ────────────────────────────────────────────────

    @property
    def context_length(self) -> int:
        return getattr(self._inner, "context_length", 4096)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def clear(self) -> int:
        """Delete all cached entries.  Returns the number of files removed."""
        count = 0
        for p in self._cache_dir.glob("*.json"):
            p.unlink()
            count += 1
        self._hits = self._misses = 0
        LOG.info("Cleared %d cached LLM responses from %s", count, self._cache_dir)
        return count

    # ── LLMConnector interface ────────────────────────────────────────

    def query(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_file = self._cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text("utf-8"))
                self._hits += 1
                LOG.debug("Cache hit  [%s] (%d hits so far)", key[:12], self._hits)
                return data["response"]
            except (json.JSONDecodeError, KeyError):
                LOG.warning("Corrupt cache entry %s — re-querying LLM", key[:12])

        response = self._inner.query(prompt)
        self._misses += 1
        LOG.debug("Cache miss [%s] (%d misses so far)", key[:12], self._misses)

        cache_file.write_text(
            json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False),
            "utf-8",
        )
        return response


# ---------------------------------------------------------------------------
# Stats-tracking wrapper
# ---------------------------------------------------------------------------

# Prompt-type detection rules.  Each entry is (query_type, check_function).
# The first match wins, so order matters.
_QUERY_TYPE_RULES: List[tuple] = [
    ("robot_actions", lambda p: "robot can perform" in p or "robot can engage" in p),
    ("human_reactions", lambda p: "things" in p and "affected humans" in p),
    ("consequences", lambda p: "ways the situation could look" in p),
    ("batch_empowerment", lambda p: "Score all scenarios on the SAME SCALE" in p),
    ("empowerment", lambda p: "meaningfully different futures" in p),
    ("hierarchical_status", lambda p: "success of the higher-level" in p),
    ("consequence_matching", lambda p: "correspond to one of these" in p),
]


def _detect_query_type(prompt: str) -> str:
    """Return a query-type label based on prompt content."""
    for qtype, check in _QUERY_TYPE_RULES:
        if check(prompt):
            return qtype
    return "other"


def _empty_bucket() -> Dict[str, int]:
    return {
        "queries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }


class StatsTrackingLLM:
    """Wrapper that collects per-query-type token usage statistics.

    Wraps any :class:`LLMConnector` and records token counts for each
    query, broken down by prompt type (robot actions, human reactions,
    consequences, empowerment, etc.) and by reasoning vs. response tokens.

    If *raw_llm* is provided and has a ``query_log`` attribute (as the
    L2P ``OPENAI`` class does), real token counts — including reasoning
    tokens — are extracted after each call.  Otherwise, tokens are
    estimated from text length (~4 characters per token).

    Usage::

        stats_llm = StatsTrackingLLM(llm, raw_llm=openai_instance)
        tree = build_tree(stats_llm, ...)
        stats_llm.print_report()
    """

    def __init__(
        self,
        inner: Any,
        raw_llm: Any = None,
    ) -> None:
        self._inner = inner
        self._raw_llm = raw_llm
        self._stats: Dict[str, Dict[str, int]] = defaultdict(_empty_bucket)
        self._total_queries = 0

    @property
    def context_length(self) -> int:
        return getattr(self._inner, "context_length", 4096)

    def query(self, prompt: str) -> str:
        qtype = _detect_query_type(prompt)

        # Note query_log length before the call so we can detect new entries
        log_before = (
            len(self._raw_llm.query_log)
            if self._raw_llm and hasattr(self._raw_llm, "query_log")
            else 0
        )

        response = self._inner.query(prompt)

        # Try to extract real token counts from the raw LLM's query log
        has_real = False
        if self._raw_llm and hasattr(self._raw_llm, "query_log"):
            log_after = len(self._raw_llm.query_log)
            if log_after > log_before:
                entry = self._raw_llm.query_log[-1]
                in_tok = entry.get("input_tokens", 0)
                out_tok = entry.get("output_tokens", 0)
                reason_tok = entry.get("reasoning_tokens", 0)
                has_real = True

        if not has_real:
            # Estimate from text length (~4 chars per token)
            in_tok = len(prompt) // 4
            out_tok = len(response) // 4
            reason_tok = 0

        bucket = self._stats[qtype]
        bucket["queries"] += 1
        bucket["input_tokens"] += in_tok
        bucket["output_tokens"] += out_tok
        bucket["reasoning_tokens"] += reason_tok
        self._total_queries += 1

        return response

    @property
    def stats(self) -> Dict[str, Dict[str, int]]:
        """Per-query-type token statistics."""
        return dict(self._stats)

    def print_report(self, file: Any = None) -> None:
        """Print a formatted token usage report.

        Columns: query type, number of queries, input tokens, output
        tokens (total), of which reasoning tokens, and response tokens
        (= output − reasoning).
        """
        import sys

        out = file or sys.stdout
        if not self._stats:
            print("No LLM queries recorded.", file=out)
            return

        # Column headers
        header = (
            f"{'Query Type':<24s} {'Queries':>7s} {'Input':>9s} "
            f"{'Output':>9s} {'Reasoning':>9s} {'Response':>9s}"
        )
        sep = "─" * len(header)

        print(f"\n{sep}", file=out)
        print("Token Usage Statistics", file=out)
        print(sep, file=out)
        print(header, file=out)
        print(sep, file=out)

        # Deterministic order: known types first, then any extras
        known_order = [
            "robot_actions",
            "human_reactions",
            "consequences",
            "batch_empowerment",
            "empowerment",
            "hierarchical_status",
            "consequence_matching",
        ]
        ordered = [k for k in known_order if k in self._stats]
        ordered += sorted(k for k in self._stats if k not in known_order)

        totals = _empty_bucket()
        for qtype in ordered:
            b = self._stats[qtype]
            resp = b["output_tokens"] - b["reasoning_tokens"]
            print(
                f"{qtype:<24s} {b['queries']:>7d} {b['input_tokens']:>9d} "
                f"{b['output_tokens']:>9d} {b['reasoning_tokens']:>9d} "
                f"{resp:>9d}",
                file=out,
            )
            for k in totals:
                totals[k] += b[k]

        resp_total = totals["output_tokens"] - totals["reasoning_tokens"]
        print(sep, file=out)
        print(
            f"{'TOTAL':<24s} {totals['queries']:>7d} "
            f"{totals['input_tokens']:>9d} {totals['output_tokens']:>9d} "
            f"{totals['reasoning_tokens']:>9d} {resp_total:>9d}",
            file=out,
        )
        print(sep, file=out)
