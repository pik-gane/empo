"""
MA-PDDL (Multi-Agent PDDL) file writer.

Writes MA-PDDL domain and problem files from MADomainSpec and MATaskSpec.
Supports factored (per-agent) and unfactored (single file) output.
No :goal section is ever produced.
"""

import json
import os
from typing import Dict, List, Union

from empo.llm_world_model.types import MADomainSpec, MATaskSpec


class MAPddlWriter:
    """
    Writes MA-PDDL domain and problem files from MADomainSpec and MATaskSpec.

    Output format follows MA-PDDL conventions:
    - Per-agent domain files with agent-owned actions
    - Shared types, predicates, and constants
    - :requirements includes :multi-agent
    - Optionally includes :probabilistic-effects

    Can produce either:
    - Factored output: separate domain/problem per agent
    - Unfactored output: single domain/problem with all agents
    """

    def write_domain(
        self, domain: MADomainSpec, factored: bool = False
    ) -> Union[str, Dict[str, str]]:
        """Generate MA-PDDL domain file(s).

        Args:
            domain: The multi-agent domain specification.
            factored: If True, return a dict mapping agent name to domain string.
                     If False, return a single domain string with all agents.

        Returns:
            Single domain string (unfactored) or dict of agent→domain (factored).
        """
        if factored:
            result = {}
            for agent in domain.agents:
                result[agent.name] = self._write_agent_domain(domain, agent.name)
            return result
        else:
            return self._write_unfactored_domain(domain)

    def write_problem(
        self,
        task: MATaskSpec,
        domain: MADomainSpec,
        factored: bool = False,
    ) -> Union[str, Dict[str, str]]:
        """Generate MA-PDDL problem file(s) without :goal.

        Args:
            task: The multi-agent task specification.
            domain: The domain specification (for agent info).
            factored: If True, return per-agent problem strings.

        Returns:
            Single problem string or dict of agent→problem.
        """
        if factored:
            result = {}
            for agent in domain.agents:
                result[agent.name] = self._write_agent_problem(task, agent.name)
            return result
        else:
            return self._write_unfactored_problem(task)

    def write_files(
        self,
        domain: MADomainSpec,
        task: MATaskSpec,
        output_dir: str,
        factored: bool = False,
    ) -> List[str]:
        """Write all files to disk and return paths.

        Also writes concurrent_effects.json sidecar file if there are
        non-commutative concurrent effects.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        if factored:
            domain_strs = self.write_domain(domain, factored=True)
            problem_strs = self.write_problem(task, domain, factored=True)
            for agent_name in domain_strs:
                dpath = os.path.join(output_dir, f"domain_{agent_name}.pddl")
                with open(dpath, "w") as f:
                    f.write(domain_strs[agent_name])
                paths.append(dpath)

                ppath = os.path.join(output_dir, f"problem_{agent_name}.pddl")
                with open(ppath, "w") as f:
                    f.write(problem_strs[agent_name])
                paths.append(ppath)
        else:
            dpath = os.path.join(output_dir, "domain.pddl")
            with open(dpath, "w") as f:
                f.write(self.write_domain(domain, factored=False))
            paths.append(dpath)

            ppath = os.path.join(output_dir, "problem.pddl")
            with open(ppath, "w") as f:
                f.write(self.write_problem(task, domain, factored=False))
            paths.append(ppath)

        # Write concurrent effects sidecar
        if domain.concurrent_effects:
            effects_path = os.path.join(output_dir, "concurrent_effects.json")
            effects_data = []
            for ce in domain.concurrent_effects:
                effects_data.append(
                    {
                        "agent_a": ce.agent_a,
                        "action_a": ce.action_a,
                        "agent_b": ce.agent_b,
                        "action_b": ce.action_b,
                        "effect_type": ce.effect_type,
                        "resolution": ce.resolution,
                        "pddl_condition": ce.pddl_condition,
                        "pddl_effect": ce.pddl_effect,
                    }
                )
            with open(effects_path, "w") as f:
                json.dump(effects_data, f, indent=2)
            paths.append(effects_path)

        return paths

    # --- Internal helpers ---

    def _write_unfactored_domain(self, domain: MADomainSpec) -> str:
        """Write a single domain file with all agents' actions."""
        lines = [f"(define (domain {domain.name})"]

        # Requirements
        if domain.requirements:
            lines.append(f"  (:requirements {' '.join(domain.requirements)})")

        # Types
        if domain.types:
            lines.append("  (:types")
            for name, desc in domain.types.items():
                lines.append(f"    {name} ; {desc}" if desc else f"    {name}")
            lines.append("  )")

        # Constants
        if domain.constants:
            lines.append("  (:constants")
            for name, typ in domain.constants.items():
                lines.append(f"    {name} - {typ}")
            lines.append("  )")

        # Predicates
        if domain.predicates:
            lines.append("  (:predicates")
            for p in domain.predicates:
                raw = p.get("raw", p.get("clean", ""))
                if raw:
                    lines.append(f"    {raw}")
                else:
                    lines.append(f"    ({p['name']})")
            lines.append("  )")

        # Functions
        if domain.functions:
            lines.append("  (:functions")
            for fn in domain.functions:
                raw = fn.get("raw", fn.get("clean", ""))
                if raw:
                    lines.append(f"    {raw}")
                else:
                    lines.append(f"    ({fn['name']})")
            lines.append("  )")

        # Actions (all agents)
        for agent_name, actions in domain.agent_actions.items():
            for action in actions:
                lines.append("")
                lines.append(f"  ; Agent: {agent_name}")
                lines.append(self._format_action(action, indent=2))

        # Concurrent effects as comments
        if domain.concurrent_effects:
            lines.append("")
            lines.append(
                "  ; === Concurrent Effect Rules "
                "(see concurrent_effects.json) ==="
            )
            for ce in domain.concurrent_effects:
                lines.append(
                    f"  ; {ce.agent_a}.{ce.action_a} × "
                    f"{ce.agent_b}.{ce.action_b}: "
                    f"{ce.effect_type} — {ce.resolution}"
                )

        lines.append(")")
        return "\n".join(lines)

    def _write_agent_domain(self, domain: MADomainSpec, agent_name: str) -> str:
        """Write a domain file for a specific agent."""
        lines = [f"(define (domain {domain.name}_{agent_name})"]

        if domain.requirements:
            lines.append(f"  (:requirements {' '.join(domain.requirements)})")

        if domain.types:
            lines.append("  (:types")
            for name, desc in domain.types.items():
                lines.append(f"    {name}")
            lines.append("  )")

        if domain.predicates:
            lines.append("  (:predicates")
            for p in domain.predicates:
                raw = p.get("raw", p.get("clean", ""))
                if raw:
                    lines.append(f"    {raw}")
            lines.append("  )")

        # Only this agent's actions
        agent_actions = domain.agent_actions.get(agent_name, [])
        for action in agent_actions:
            lines.append("")
            lines.append(self._format_action(action, indent=2))

        lines.append(")")
        return "\n".join(lines)

    def _write_unfactored_problem(self, task: MATaskSpec) -> str:
        """Write a single problem file (no :goal)."""
        lines = [f"(define (problem {task.name})"]
        lines.append(f"  (:domain {task.domain_name})")

        if task.objects:
            lines.append("  (:objects")
            for name, typ in task.objects.items():
                lines.append(f"    {name} - {typ}")
            lines.append("  )")

        if task.initial_state:
            lines.append("  (:init")
            for atom in task.initial_state:
                lines.append(f"    {atom}")
            lines.append("  )")

        # No :goal section
        lines.append(")")
        return "\n".join(lines)

    def _write_agent_problem(self, task: MATaskSpec, agent_name: str) -> str:
        """Write a problem file for a specific agent."""
        lines = [f"(define (problem {task.name}_{agent_name})"]
        lines.append(f"  (:domain {task.domain_name}_{agent_name})")

        # Include all objects (shared state)
        if task.objects:
            lines.append("  (:objects")
            for name, typ in task.objects.items():
                lines.append(f"    {name} - {typ}")
            lines.append("  )")

        if task.initial_state:
            lines.append("  (:init")
            for atom in task.initial_state:
                lines.append(f"    {atom}")
            lines.append("  )")

        lines.append(")")
        return "\n".join(lines)

    def _format_action(self, action: dict, indent: int = 2) -> str:
        """Format a single PDDL action."""
        pad = " " * indent
        lines = [f"{pad}(:action {action['name']}"]

        # Parameters
        params = action.get("params", {})
        if params:
            param_strs = [f"?{k} - {v}" for k, v in params.items()]
            lines.append(f"{pad}  :parameters ({' '.join(param_strs)})")
        else:
            lines.append(f"{pad}  :parameters ()")

        # Preconditions
        precond = action.get("preconditions", "")
        if precond:
            lines.append(f"{pad}  :precondition {precond}")

        # Effects
        effects = action.get("effects", "")
        if effects:
            lines.append(f"{pad}  :effect {effects}")

        lines.append(f"{pad})")
        return "\n".join(lines)
