"""
Minimal example: build a WorldModel from a tic-tac-toe scene description
using the free Gemini 2.0 Flash API, then run backward induction (Phase 1
and Phase 2) and roll out 20 episodes.

Requirements:
    pip install retry openai tiktoken

Usage:
    export GOOGLE_API_KEY="your-free-gemini-api-key"
    PYTHONPATH=src:vendor/l2p python examples/llm/tictactoe_world_model_example.py

Get a free API key at https://aistudio.google.com/apikey
"""

import logging
import os
import sys
from typing import Iterator, List, Tuple

import numpy as np

from empo.possible_goal import PossibleGoal, PossibleGoalGenerator

# Path to the L2P openaiSDK.yaml config
L2P_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "vendor", "l2p", "l2p", "llm", "utils", "openaiSDK.yaml",
    )
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Goal classes: each terminal state (at max_steps) is a possible goal
# ---------------------------------------------------------------------------

class TerminalStateGoal(PossibleGoal):
    """A goal that is achieved when the world is in a specific atom-set."""

    def __init__(self, env, target_atoms, index=None):
        super().__init__(env, index=index)
        self.target_atoms = target_atoms  # frozenset of ground atoms
        self._hash = hash(self.target_atoms)
        super()._freeze()

    def is_achieved(self, state) -> int:
        _, current_atoms = state  # (step_count, frozenset)
        return 1 if current_atoms == self.target_atoms else 0

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, TerminalStateGoal)
            and self.target_atoms == other.target_atoms
        )

    def __repr__(self):
        return f"TerminalStateGoal({len(self.target_atoms)} atoms)"


class TerminalStateGoalGenerator(PossibleGoalGenerator):
    """Enumerate all reachable terminal states as goals (uniform weight)."""

    def __init__(self, env, terminal_states):
        super().__init__(env, indexed=True)
        self._goals = [
            TerminalStateGoal(env, atoms, index=i)
            for i, atoms in enumerate(terminal_states)
        ]
        self._weight = 1.0 / len(self._goals) if self._goals else 1.0

    def generate(
        self, state, human_agent_index: int
    ) -> Iterator[Tuple[PossibleGoal, float]]:
        for g in self._goals:
            yield g, self._weight


def collect_terminal_states(world_model, max_states: int = 1000) -> List:
    """BFS to find reachable terminal atom-sets, stopping at *max_states*."""
    from collections import deque

    terminal_atoms = set()
    visited = set()
    queue = deque([world_model.get_state()])

    # Per-agent action counts
    agent_names = world_model._agent_names
    agent_counts = [world_model._agent_action_counts[n] for n in agent_names]

    while queue:
        if len(visited) >= max_states:
            print(f"  [!] Reached state limit ({max_states}), stopping BFS. "
                  f"Found {len(terminal_atoms)} terminal states so far.")
            break

        state = queue.popleft()
        if state in visited:
            continue
        visited.add(state)

        if len(visited) % 200 == 0:
            print(f"  ... explored {len(visited)} states, "
                  f"{len(terminal_atoms)} terminal, queue={len(queue)}")

        # Generate all joint action tuples
        action_ranges = [range(c) for c in agent_counts]
        any_successor = False
        for joint_action in _product(action_ranges):
            transitions = world_model.transition_probabilities(
                state, list(joint_action)
            )
            if transitions is None:
                # Terminal state
                terminal_atoms.add(state[1])  # the frozenset of atoms
                continue
            any_successor = True
            for _, succ in transitions:
                if succ not in visited:
                    queue.append(succ)

        if not any_successor:
            terminal_atoms.add(state[1])

    print(f"  BFS done: {len(visited)} states explored, "
          f"{len(terminal_atoms)} terminal states.")
    return list(terminal_atoms)


def _product(ranges):
    """Itertools.product for a list of ranges."""
    import itertools
    return itertools.product(*ranges)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY env var (free at https://aistudio.google.com/apikey)")
        sys.exit(1)

    # L2P's OPENAI class works with any OpenAI-SDK-compatible endpoint,
    # including Gemini's free tier.
    from l2p.llm.openai import OPENAI

    llm = OPENAI(
        model="gemini-2.0-flash",
        provider="google",
        config_path=L2P_CONFIG,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    from empo.llm_world_model import WorldModelBuilder

    builder = WorldModelBuilder(llm=llm, max_steps=9)  # tic-tac-toe: at most 9 moves

    scene = (
        "A robot (X) and a human (O) play tic-tac-toe on a 3x3 grid. "
        "Each player can place their mark on one empty cell per turn. "
        "The game proceeds in alternating turns, with X moving first. "
        "A player wins by getting three marks in a row, column, or diagonal."
    )

    print(f"\nScene: {scene}\n")
    print("Building world model (this makes several LLM calls)...\n")

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "tictactoe_pddl"
    )
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    world_model, pddl_paths = builder.build_and_export(
        scene, output_dir=output_dir,
    )

    # Inspect the result
    state = world_model.get_state()
    print(f"\nWorld model built successfully!")
    print(f"  Agents: {[a.name for a in builder.domain.agents]}")
    print(f"  Ground atoms: {world_model._num_atoms}")
    print(f"  Actions per agent: {world_model._agent_action_counts}")
    print(f"  State: {state}")
    print(f"  PDDL files written to: {output_dir}")
    for p in pddl_paths:
        print(f"    {p}")

    # ------------------------------------------------------------------
    # Collect terminal states and build goal generator
    # ------------------------------------------------------------------
    print("\nCollecting reachable terminal states...")
    terminal_states = collect_terminal_states(world_model)
    print(f"  Found {len(terminal_states)} terminal states as possible goals.")

    if not terminal_states:
        print("No terminal states found — cannot run backward induction.")
        sys.exit(1)

    goal_gen = TerminalStateGoalGenerator(world_model, terminal_states)

    # ------------------------------------------------------------------
    # Backward induction: Phase 1 (human policy prior)
    # ------------------------------------------------------------------
    from empo.backward_induction import (
        compute_human_policy_prior,
        compute_robot_policy,
    )

    human_indices = world_model.human_agent_indices
    robot_indices = [i for i in world_model.agents if i not in human_indices]

    print(f"\nPhase 1: computing human policy prior "
          f"(human agents: {human_indices})...")
    human_policy_prior = compute_human_policy_prior(
        world_model=world_model,
        human_agent_indices=human_indices,
        possible_goal_generator=goal_gen,
        beta_h=10.0,
        gamma_h=0.95,
        level_fct=lambda state: state[0],  # step count as level
    )
    print("  Phase 1 done.")

    # ------------------------------------------------------------------
    # Backward induction: Phase 2 (robot policy)
    # ------------------------------------------------------------------
    print(f"\nPhase 2: computing robot policy "
          f"(robot agents: {robot_indices})...")
    robot_policy = compute_robot_policy(
        world_model=world_model,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        possible_goal_generator=goal_gen,
        human_policy_prior=human_policy_prior,
        beta_r=10.0,
        gamma_h=0.95,
        gamma_r=0.95,
        level_fct=lambda state: state[0],
    )
    print("  Phase 2 done.")

    # ------------------------------------------------------------------
    # Roll out 20 episodes
    # ------------------------------------------------------------------
    N_RUNS = 20
    print(f"\nRolling out {N_RUNS} episodes...\n")

    for run in range(1, N_RUNS + 1):
        world_model.reset()
        trajectory = []

        # Pick a random goal for the human this episode
        goals_list = list(goal_gen.generate(world_model.get_state(), human_indices[0]))
        goal, _ = goals_list[np.random.randint(len(goals_list))]

        for step in range(world_model._max_steps):
            state = world_model.get_state()
            actions = [0] * len(world_model.agents)

            # Human: sample from policy prior
            for h_idx in human_indices:
                dist = human_policy_prior(state, h_idx, goal)
                if dist is not None:
                    actions[h_idx] = int(np.random.choice(len(dist), p=dist))

            # Robot: sample from computed robot policy
            robot_action = robot_policy.sample(state)
            if robot_action is not None:
                for i, r_idx in enumerate(robot_indices):
                    actions[r_idx] = robot_action[i]

            trajectory.append((state, list(actions)))

            obs, reward, terminated, truncated, info = world_model.step(actions)
            if terminated or truncated:
                break

        final_state = world_model.get_state()
        achieved = goal.is_achieved(final_state)
        print(
            f"  Run {run:2d}: {step + 1} steps, "
            f"goal achieved: {'yes' if achieved else 'no'}, "
            f"final atoms: {len(final_state[1])}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
