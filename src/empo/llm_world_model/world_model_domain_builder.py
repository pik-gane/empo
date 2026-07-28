"""
Multi-agent domain extraction from natural language scene descriptions.

Extends L2P's DomainBuilder to support multi-agent domain extraction with
per-agent action schemas, concurrent effect identification, and optional
probabilistic effects.
"""

import json
import logging
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from l2p.domain_builder import DomainBuilder
from l2p.llm.base import BaseLLM
from l2p.utils.pddl_types import Action, Function, Predicate
from l2p.utils.pddl_validator import SyntaxValidator

from empo.llm_world_model.types import AgentSpec, ConcurrentEffect, MADomainSpec

LOG = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    """Load a prompt template from the bundled prompts directory."""
    path = os.path.join(_PROMPTS_DIR, name)
    with open(path, "r") as f:
        return f.read()


def _format_agents_summary(agents: List[AgentSpec]) -> str:
    """Format agents into a human-readable summary for prompts."""
    lines = []
    for a in agents:
        lines.append(f"- {a.name} (type: {a.agent_type})")
        if a.capabilities:
            lines.append(f"  capabilities: {', '.join(a.capabilities)}")
    return "\n".join(lines)


def _format_predicates_summary(predicates: List[Predicate]) -> str:
    """Format predicates into a human-readable summary for prompts."""
    lines = []
    for p in predicates:
        desc = p.get("desc", "")
        raw = p.get("raw", p.get("clean", p.get("name", "")))
        if desc:
            lines.append(f"- {raw}: '{desc}'")
        else:
            lines.append(f"- {raw}")
    return "\n".join(lines)


def _format_types_summary(types: Dict[str, str]) -> str:
    """Format types into a human-readable summary."""
    lines = []
    for name, desc in types.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _format_actions_summary(agent_actions: Dict[str, List[Action]]) -> str:
    """Format per-agent actions into a summary for concurrent effects prompt."""
    lines = []
    for agent_name, actions in agent_actions.items():
        lines.append(f"Agent '{agent_name}':")
        for a in actions:
            desc = a.get("desc", "")
            lines.append(f"  - {a['name']}: {desc}" if desc else f"  - {a['name']}")
    return "\n".join(lines)


def _parse_types_from_llm(raw: str) -> Dict[str, str]:
    """Parse types from LLM output in '- name: description' format."""
    types: Dict[str, str] = {}
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            line = line[2:]  # Remove '- '
            name, _, desc = line.partition(":")
            name = name.strip()
            desc = desc.strip()
            if name and name != "object":
                types[name] = desc
    return types


def _parse_predicates_from_llm(raw: str) -> List[Predicate]:
    """Parse predicates from LLM output.

    Expects format: - (pred_name ?p1 - type1 ?p2 - type2): 'description'
    """
    predicates: List[Predicate] = []
    text = raw
    if "### New Predicates" in text:
        text = text.split("### New Predicates", 1)[1]

    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("- ("):
            continue

        # Extract the predicate raw form and description
        line = line[2:]  # Remove '- '

        # Split on ): to separate raw form from description
        if "):" in line:
            raw_form, _, desc = line.partition("):")
            raw_form = raw_form + ")"
            desc = desc.strip().strip("'\"")
        else:
            raw_form = line
            desc = ""

        # Parse predicate name and parameters
        inner = raw_form.strip()
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1].strip()
        else:
            continue

        parts = inner.split()
        if not parts:
            continue

        pred_name = parts[0]

        # Parse parameters: ?param - type pairs
        params = OrderedDict()
        i = 1
        while i < len(parts):
            if parts[i].startswith("?"):
                param_name = parts[i].lstrip("?")
                param_type = ""
                if i + 1 < len(parts) and parts[i + 1] == "-":
                    if i + 2 < len(parts):
                        param_type = parts[i + 2]
                        i += 3
                    else:
                        i += 2
                else:
                    i += 1
                params[param_name] = param_type
            else:
                i += 1

        predicates.append(
            {
                "name": pred_name,
                "desc": desc,
                "raw": raw_form,
                "params": params,
                "clean": raw_form,
            }
        )

    return predicates


def _parse_action_from_llm(raw: str, action_name: str) -> Optional[Action]:
    """Parse a single action from LLM output with ### sections."""
    # Parse parameters
    params = OrderedDict()
    if "### Action Parameters" in raw:
        params_section = raw.split("### Action Parameters", 1)[1]
        if "###" in params_section:
            params_section = params_section.split("###", 1)[0]
        # Extract from code blocks
        blocks = re.findall(r"```(.*?)```", params_section, re.DOTALL)
        params_text = blocks[0] if blocks else params_section
        for line in params_text.split("\n"):
            line = line.strip()
            if line.startswith("- ") and "?" in line:
                line = line[2:]  # Remove '- '
                # Parse ?param - type: 'desc' format
                if ":" in line:
                    line = line[: line.index(":")].strip()
                parts = line.split()
                if len(parts) >= 3 and parts[0].startswith("?") and parts[1] == "-":
                    param_name = parts[0].lstrip("?")
                    param_type = parts[2]
                    params[param_name] = param_type

    # Parse preconditions
    preconditions = ""
    if "### Action Preconditions" in raw:
        precond_section = raw.split("### Action Preconditions", 1)[1]
        if "###" in precond_section:
            precond_section = precond_section.split("###", 1)[0]
        blocks = re.findall(r"```(.*?)```", precond_section, re.DOTALL)
        if blocks:
            preconditions = blocks[0].strip()

    # Parse effects
    effects = ""
    if "### Action Effects" in raw:
        effects_section = raw.split("### Action Effects", 1)[1]
        if "###" in effects_section:
            effects_section = effects_section.split("###", 1)[0]
        blocks = re.findall(r"```(.*?)```", effects_section, re.DOTALL)
        if blocks:
            effects = blocks[0].strip()

    if not params and not preconditions and not effects:
        return None

    return {
        "name": action_name,
        "desc": "",
        "raw": "",
        "params": params,
        "preconditions": preconditions,
        "effects": effects,
    }


def _parse_concurrent_effects_json(raw: str) -> List[ConcurrentEffect]:
    """Parse LLM output for concurrent effects (JSON array)."""
    text = raw
    if "### CONCURRENT EFFECTS" in text:
        text = text.split("### CONCURRENT EFFECTS", 1)[1]

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        # No concurrent effects found — this is valid (all commutative)
        return []

    json_str = text[start : end + 1]
    data = json.loads(json_str)

    if not isinstance(data, list):
        return []

    effects = []
    for item in data:
        if not isinstance(item, dict):
            continue
        effects.append(
            ConcurrentEffect(
                agent_a=str(item.get("agent_a", "")),
                action_a=str(item.get("action_a", "")),
                agent_b=str(item.get("agent_b", "")),
                action_b=str(item.get("action_b", "")),
                effect_type=str(item.get("effect_type", "commutative")),
                resolution=str(item.get("resolution", "")),
                pddl_condition=item.get("pddl_condition"),
                pddl_effect=item.get("pddl_effect"),
            )
        )
    return effects


class WorldModelDomainBuilder(DomainBuilder):
    """
    Extends L2P's DomainBuilder for multi-agent world model extraction.

    Differences from base DomainBuilder:
    1. Actions are agent-owned (each action belongs to exactly one agent)
    2. Supports concurrent action effect resolution
    3. Supports :probabilistic-effects (PPDDL)
    4. No goal-related methods
    """

    def extract_types(
        self,
        model: BaseLLM,
        scene_desc: str,
        prompt_template: str = None,
        max_retries: int = 3,
    ) -> Tuple[Dict[str, str], str]:
        """Extract types from scene description using our own parser.

        Returns:
            (types_dict, llm_raw_output)
        """
        if prompt_template is None:
            prompt_template = (
                "Extract the types of entities in this scene.\n\n"
                "Scene description:\n{scene_desc}\n\n"
                "List each type as:\n- name: description\n\n### Types\n"
            )

        prompt = prompt_template.replace("{scene_desc}", scene_desc)

        last_error = None
        for attempt in range(max_retries):
            raw_output = model.query(prompt)
            try:
                types = _parse_types_from_llm(raw_output)
                if types:
                    return types, raw_output
                raise ValueError("No types found in LLM output")
            except ValueError as e:
                last_error = e
                LOG.warning("Attempt %d/%d: %s", attempt + 1, max_retries, e)

        raise ValueError(
            f"Failed to extract types after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def formalize_shared_predicates(
        self,
        model: BaseLLM,
        scene_desc: str,
        agents: List[AgentSpec],
        prompt_template: str = None,
        types: Dict[str, str] = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> Tuple[List[Predicate], str, Tuple[bool, str]]:
        """
        Extract typed predicates describing the shared state space.

        Returns:
            (predicates, llm_raw_output, (validation_ok, validation_msg))
        """
        if prompt_template is None:
            prompt_template = _load_prompt("formalize_predicates.txt")

        agents_summary = _format_agents_summary(agents)
        types_str = _format_types_summary(types or {})

        prompt = (
            prompt_template.replace("{scene_desc}", scene_desc)
            .replace("{agents_summary}", agents_summary)
            .replace("{types}", types_str)
        )

        last_error = None
        for attempt in range(max_retries):
            raw_output = model.query(prompt)
            try:
                predicates = _parse_predicates_from_llm(raw_output)
                if predicates:
                    return predicates, raw_output, (True, "OK")
                raise ValueError("No predicates found in LLM output")
            except ValueError as e:
                last_error = e
                LOG.warning("Attempt %d/%d: %s", attempt + 1, max_retries, e)

        raise ValueError(
            f"Failed to extract predicates after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def formalize_agent_actions(
        self,
        model: BaseLLM,
        scene_desc: str,
        agent: AgentSpec,
        prompt_template: str = None,
        types: Dict[str, str] = None,
        predicates: List[Predicate] = None,
        functions: List[Function] = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> Tuple[List[Action], List[Predicate], str, Tuple[bool, str]]:
        """
        Extract action schemas owned by a specific agent.

        Returns:
            (actions, new_predicates, llm_output, validation_info)
        """
        if prompt_template is None:
            prompt_template = _load_prompt("formalize_agent_actions.txt")

        preds_str = _format_predicates_summary(predicates or [])
        types_str = _format_types_summary(types or {})

        prompt = (
            prompt_template.replace("{scene_desc}", scene_desc)
            .replace("{agent_name}", agent.name)
            .replace("{agent_type}", agent.agent_type)
            .replace("{agent_capabilities}", ", ".join(agent.capabilities))
            .replace("{types}", types_str)
            .replace("{predicates}", preds_str)
        )

        actions = []
        new_predicates: List[Predicate] = []
        all_raw: List[str] = []
        failed_hints: List[str] = []

        for hint in agent.action_hints:
            action_prompt = prompt + f"\n\nFormalize the action: {hint}"

            last_error = None
            for attempt in range(max_retries):
                raw_output = model.query(action_prompt)
                try:
                    action = _parse_action_from_llm(raw_output, hint)
                    if action:
                        action["desc"] = (
                            f"[agent:{agent.name}] {action.get('desc', '')}"
                        ).strip()
                        actions.append(action)

                    # Parse any new predicates
                    if "### New Predicates" in raw_output:
                        new_preds = _parse_predicates_from_llm(raw_output)
                        new_predicates.extend(new_preds)

                    all_raw.append(raw_output)
                    break
                except Exception as e:
                    last_error = e
                    LOG.warning(
                        "Attempt %d/%d for action '%s': %s",
                        attempt + 1,
                        max_retries,
                        hint,
                        e,
                    )
            else:
                LOG.warning(
                    "Failed to parse action '%s' for agent '%s' after %d attempts",
                    hint,
                    agent.name,
                    max_retries,
                )
                failed_hints.append(hint)
                all_raw.append(f"[Failed to parse action '{hint}': {last_error}]")

        combined_raw = "\n---\n".join(all_raw)
        if failed_hints:
            validation_msg = (
                f"Failed to parse {len(failed_hints)} action(s) for agent "
                f"'{agent.name}': {failed_hints}"
            )
            return actions, new_predicates, combined_raw, (False, validation_msg)
        return actions, new_predicates, combined_raw, (True, "OK")

    def identify_concurrent_effects(
        self,
        model: BaseLLM,
        scene_desc: str,
        agents: List[AgentSpec],
        agent_actions: Dict[str, List[Action]],
        predicates: List[Predicate],
        prompt_template: str = None,
        max_retries: int = 3,
    ) -> Tuple[List[ConcurrentEffect], str]:
        """
        Identify how concurrent actions from different agents interact.

        Returns:
            (concurrent_effects, llm_raw_output)
        """
        if prompt_template is None:
            prompt_template = _load_prompt("identify_concurrent_effects.txt")

        actions_summary = _format_actions_summary(agent_actions)

        prompt = prompt_template.replace("{scene_desc}", scene_desc).replace(
            "{agent_actions_summary}", actions_summary
        )

        last_error = None
        for attempt in range(max_retries):
            raw_output = model.query(prompt)
            try:
                effects = _parse_concurrent_effects_json(raw_output)
                LOG.info("Identified %d concurrent effects", len(effects))
                return effects, raw_output
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                LOG.warning(
                    "Attempt %d/%d: Failed to parse concurrent effects: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )

        raise ValueError(
            f"Failed to parse concurrent effects after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def build_ma_domain(
        self,
        scene_desc: str,
        agents: List[AgentSpec],
        model: BaseLLM,
        prompt_templates: Dict[str, str] = None,
        enable_probabilistic: bool = False,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> MADomainSpec:
        """
        Orchestrate full multi-agent domain extraction.

        Pipeline:
        1. extract_types()
        2. formalize_shared_predicates()
        3. For each agent: formalize_agent_actions()
        4. identify_concurrent_effects()
        5. Assemble MADomainSpec
        """
        templates = prompt_templates or {}

        # Step 1: Extract types
        types, _ = self.extract_types(
            model=model,
            scene_desc=scene_desc,
            prompt_template=templates.get("types"),
            max_retries=max_retries,
        )

        # Step 2: Extract shared predicates
        predicates, _, _ = self.formalize_shared_predicates(
            model=model,
            scene_desc=scene_desc,
            agents=agents,
            prompt_template=templates.get("predicates"),
            types=types,
            syntax_validator=syntax_validator,
            max_retries=max_retries,
        )

        # Step 3: For each agent, extract actions
        agent_actions: Dict[str, List[Action]] = {}
        all_new_predicates: List[Predicate] = []
        for agent in agents:
            actions, new_preds, _, _ = self.formalize_agent_actions(
                model=model,
                scene_desc=scene_desc,
                agent=agent,
                prompt_template=templates.get("agent_actions"),
                types=types,
                predicates=predicates + all_new_predicates,
                syntax_validator=syntax_validator,
                max_retries=max_retries,
            )
            agent_actions[agent.name] = actions
            all_new_predicates.extend(new_preds)

        all_predicates = predicates + all_new_predicates

        # Step 4: Identify concurrent effects
        concurrent_effects, _ = self.identify_concurrent_effects(
            model=model,
            scene_desc=scene_desc,
            agents=agents,
            agent_actions=agent_actions,
            predicates=all_predicates,
            prompt_template=templates.get("concurrent_effects"),
            max_retries=max_retries,
        )

        # Step 5: Assemble MADomainSpec
        requirements = [":typing", ":multi-agent"]
        if enable_probabilistic:
            requirements.append(":probabilistic-effects")

        return MADomainSpec(
            name="world",
            types=types,
            predicates=all_predicates,
            agents=agents,
            agent_actions=agent_actions,
            concurrent_effects=concurrent_effects,
            requirements=requirements,
        )
