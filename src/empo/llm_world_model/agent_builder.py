"""
Agent identification from natural language scene descriptions.

This module provides the AgentBuilder class that prompts an LLM to identify
distinct agents in a natural language scene description and extract their
specifications (name, type, capabilities, action hints).
"""

import json
import logging
import os
from typing import List, Tuple

from l2p.llm.base import BaseLLM

from empo.llm_world_model.types import AgentSpec

LOG = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    """Load a prompt template from the bundled prompts directory."""
    path = os.path.join(_PROMPTS_DIR, name)
    with open(path, "r") as f:
        return f.read()


def _parse_agents_json(raw: str) -> List[AgentSpec]:
    """Parse the LLM output to extract a list of AgentSpec objects.

    Expects the LLM output to contain a JSON array under `### AGENTS`.
    Falls back to finding any JSON array in the output.
    """
    # Try to extract JSON block after ### AGENTS header
    text = raw
    if "### AGENTS" in text:
        text = text.split("### AGENTS", 1)[1]

    # Find JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON array in LLM output:\n{raw[:500]}")

    json_str = text[start : end + 1]
    data = json.loads(json_str)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    agents = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object in array, got {type(item).__name__}")
        agents.append(
            AgentSpec(
                name=str(item.get("name", "")),
                agent_type=str(item.get("agent_type", "")),
                capabilities=[str(c) for c in item.get("capabilities", [])],
                action_hints=[str(h) for h in item.get("action_hints", [])],
            )
        )

    return agents


class AgentBuilder:
    """Extract agent specifications from natural language scene descriptions."""

    def identify_agents(
        self,
        model: BaseLLM,
        scene_desc: str,
        prompt_template: str = None,
        max_retries: int = 3,
    ) -> Tuple[List[AgentSpec], str]:
        """
        Prompt the LLM to identify agents in the scene description.

        Args:
            model: LLM instance to query.
            scene_desc: Natural language description of the scene.
            prompt_template: Prompt template string. If None, loads bundled default.
            max_retries: Maximum number of retries on parse failure.

        Returns:
            Tuple of (list of AgentSpec, raw LLM output string).
        """
        if prompt_template is None:
            prompt_template = _load_prompt("identify_agents.txt")

        prompt = prompt_template.replace("{scene_desc}", scene_desc)

        last_error = None
        for attempt in range(max_retries):
            raw_output = model.query(prompt)

            try:
                agents = _parse_agents_json(raw_output)
                if not agents:
                    raise ValueError("LLM returned empty agent list")
                LOG.info(
                    "Identified %d agents: %s",
                    len(agents),
                    [a.name for a in agents],
                )
                return agents, raw_output
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                LOG.warning(
                    "Attempt %d/%d: Failed to parse agents: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                # Add error context to prompt for retry
                prompt = (
                    prompt_template.replace("{scene_desc}", scene_desc)
                    + f"\n\n[Previous attempt failed with error: {e}. "
                    "Please output valid JSON.]"
                )

        raise ValueError(
            f"Failed to identify agents after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
