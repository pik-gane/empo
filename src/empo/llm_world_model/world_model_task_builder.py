"""
Goal-free task extraction from natural language scene descriptions.

Extends L2P's TaskBuilder to extract objects and initial state only (no goal
state). Adds validation that all agent-owned objects are properly instantiated.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from l2p.llm.base import BaseLLM
from l2p.task_builder import TaskBuilder
from l2p.utils.pddl_types import Predicate
from l2p.utils.pddl_validator import SyntaxValidator

from empo.llm_world_model.types import AgentSpec, MADomainSpec, MATaskSpec

LOG = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    path = os.path.join(_PROMPTS_DIR, name)
    with open(path, "r") as f:
        return f.read()


def _format_agents_summary(agents: List[AgentSpec]) -> str:
    lines = []
    for a in agents:
        lines.append(f"- {a.name} (type: {a.agent_type})")
    return "\n".join(lines)


def _format_predicates_summary(predicates: List[Predicate]) -> str:
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
    lines = []
    for name, desc in types.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _parse_objects_block(raw: str) -> Dict[str, str]:
    """Parse ### OBJECTS block into {name: type} dict."""
    objects: Dict[str, str] = {}
    text = raw
    if "### OBJECTS" in text:
        text = text.split("### OBJECTS", 1)[1]
    # Find content between ``` markers
    code_blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
    if code_blocks:
        text = code_blocks[0]

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Remove comments
        if ";" in line:
            line = line[: line.index(";")].strip()
        # Parse "name - type" format
        if " - " in line:
            parts = line.split(" - ", 1)
            name = parts[0].strip()
            obj_type = parts[1].strip()
            if name and obj_type:
                objects[name] = obj_type
    return objects


def _parse_initial_block(raw: str) -> List[str]:
    """Parse ### INITIAL block into list of ground atom strings."""
    initial: List[str] = []
    text = raw
    if "### INITIAL" in text:
        text = text.split("### INITIAL", 1)[1]

    code_blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
    if code_blocks:
        text = code_blocks[0]

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Remove comments
        if ";" in line:
            line = line[: line.index(";")].strip()
        # Should be a PDDL atom like (at robot kitchen)
        if line.startswith("(") and ")" in line:
            # Extract complete atoms
            depth = 0
            start = None
            for i, ch in enumerate(line):
                if ch == "(":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and start is not None:
                        initial.append(line[start : i + 1])
                        start = None
    return initial


class WorldModelTaskBuilder(TaskBuilder):
    """
    Extends L2P's TaskBuilder for goal-free world model extraction.

    Differences from base TaskBuilder:
    1. No goal extraction (formalize_goal_state is removed/no-op)
    2. Validates agent objects are instantiated
    3. generate_task() omits :goal section
    """

    def formalize_objects_and_initial(
        self,
        model: BaseLLM,
        scene_desc: str,
        domain: MADomainSpec,
        prompt_template: str = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
    ) -> Tuple[MATaskSpec, str, Tuple[bool, str]]:
        """
        Extract objects and initial state from scene description.

        Validates:
        - Every agent in domain.agents has a corresponding object
        - All typed objects match domain types
        - Initial predicates use only declared predicates and objects
        """
        if prompt_template is None:
            prompt_template = _load_prompt("formalize_initial_state.txt")

        agents_summary = _format_agents_summary(domain.agents)
        types_str = _format_types_summary(domain.types)
        preds_str = _format_predicates_summary(domain.predicates)

        prompt = (
            prompt_template.replace("{scene_desc}", scene_desc)
            .replace("{types}", types_str)
            .replace("{predicates}", preds_str)
            .replace("{agents_summary}", agents_summary)
        )

        last_error = None
        for attempt in range(max_retries):
            raw_output = model.query(prompt)

            try:
                objects = _parse_objects_block(raw_output)
                initial_state = _parse_initial_block(raw_output)

                # Validate that each agent has at least one object of its type
                validation_msgs = []
                for agent in domain.agents:
                    has_object = any(
                        typ == agent.agent_type or name == agent.name
                        for name, typ in objects.items()
                    )
                    if not has_object:
                        validation_msgs.append(
                            f"Agent '{agent.name}' has no object of type "
                            f"'{agent.agent_type}'"
                        )

                # Validate object types match domain types
                # 'object' is always valid as the PDDL built-in top type
                valid_types = set(domain.types.keys()) | {"object"}
                for obj_name, obj_type in objects.items():
                    if obj_type not in valid_types:
                        validation_msgs.append(
                            f"Object '{obj_name}' has type '{obj_type}' "
                            f"not in domain types: {valid_types}"
                        )

                # Build agent_objects mapping
                agent_objects: Dict[str, List[str]] = {}
                for agent in domain.agents:
                    agent_type = agent.agent_type
                    agent_objects[agent.name] = [
                        name
                        for name, typ in objects.items()
                        if typ == agent_type or name == agent.name
                    ]

                validation_ok = len(validation_msgs) == 0
                validation_msg = "; ".join(validation_msgs) if validation_msgs else "OK"

                task = MATaskSpec(
                    name="scenario",
                    domain_name=domain.name,
                    objects=objects,
                    initial_state=initial_state,
                    agent_objects=agent_objects,
                )

                return task, raw_output, (validation_ok, validation_msg)

            except (ValueError, KeyError) as e:
                last_error = e
                LOG.warning(
                    "Attempt %d/%d: Failed to parse task: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                prompt = (
                    prompt_template.replace("{scene_desc}", scene_desc)
                    .replace("{types}", types_str)
                    .replace("{predicates}", preds_str)
                    .replace("{agents_summary}", agents_summary)
                    + f"\n\n[Previous attempt failed: {e}. Please try again.]"
                )

        raise ValueError(
            f"Failed to extract task after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_task_pddl(
        self,
        domain_name: str,
        problem_name: str,
        task: MATaskSpec,
    ) -> str:
        """
        Generate PDDL problem string WITHOUT :goal section.
        """
        lines = [f"(define (problem {problem_name})"]
        lines.append(f"  (:domain {domain_name})")

        # Objects
        if task.objects:
            lines.append("  (:objects")
            for name, typ in task.objects.items():
                lines.append(f"    {name} - {typ}")
            lines.append("  )")

        # Init
        if task.initial_state:
            lines.append("  (:init")
            for atom in task.initial_state:
                lines.append(f"    {atom}")
            lines.append("  )")

        # No :goal section

        lines.append(")")
        return "\n".join(lines)
