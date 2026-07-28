"""
Natural-language WorldModel built from an LLM-generated trajectory tree.

Converts the tree produced by :func:`build_tree` into a concrete
:class:`~empo.world_model.WorldModel` where:

* **States** are histories ending in an observation (or the root state).
  Each state is identified by a hashable tuple of event strings.
* **Actions** are ``(robot_action_index, humans_reaction_index)`` pairs – an
  action profile of size 2 (robot agent + aggregated human agent).
* **Transitions** carry the LLM-estimated probabilities from the consequence
  step.
* **Terminal states** store an empowerment estimate accessible via
  :meth:`V_r_estimate`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gymnasium as gym

from empo.world_model import WorldModel
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.simple_hierarchical_llm_modeler.tree_builder import TreeNode


class NLWorldModel(WorldModel):
    """A WorldModel whose states and actions are derived from NL descriptions.

    After construction, the model is fully tabular – no further LLM calls are
    needed.  Use :meth:`from_tree` to construct an instance from a
    :class:`TreeNode` root.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

        # state -> {action_profile -> [(prob, next_state), ...]}
        self._transitions: Dict[
            tuple, Dict[Tuple[int, int], List[Tuple[float, tuple]]]
        ] = defaultdict(lambda: defaultdict(list))

        # state -> empowerment estimate (terminal states only)
        self._empowerment: Dict[tuple, float] = {}

        # state descriptions for human-readable output
        self._state_descriptions: Dict[tuple, str] = {}

        # action label tables (per state for robot, per (state,ra) for humans)
        # These map indices back to textual descriptions.
        self._robot_action_labels: Dict[tuple, List[str]] = {}
        self._humans_reaction_labels: Dict[tuple, Dict[int, List[str]]] = {}

        # All reachable states (including terminal)
        self._states: List[tuple] = []

        # Max number of robot actions / humans reactions across all states
        self._n_robot_actions: int = 0
        self._n_humans_reactions: int = 0

        # Current state
        self._current_state: tuple = ()

        # Initial state
        self._initial_state: tuple = ()

        # Fake agents list (robot + aggregated-humans)
        self.agents = [
            type("Agent", (), {"name": "robot"})(),
            type("Agent", (), {"name": "humans"})(),
        ]
        self.action_space = gym.spaces.Discrete(1)  # placeholder, updated later

        # Source tree (set by from_tree)
        self._tree: Optional[TreeNode] = None

    # ------------------------------------------------------------------
    # Per-agent action counts
    # ------------------------------------------------------------------

    @property
    def num_actions_per_agent(self) -> List[int]:
        """Return per-agent action counts: [n_robot_actions, n_humans_reactions]."""
        return [self._n_robot_actions, self._n_humans_reactions]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_tree(
        cls, root: TreeNode, initial_state_description: str
    ) -> "NLWorldModel":
        """Build a :class:`NLWorldModel` from a trajectory tree.

        Args:
            root: Root node (type ``"state"``) as returned by :func:`build_tree`.
            initial_state_description: Human-readable description of the
                initial situation.

        Returns:
            A fully-initialised :class:`NLWorldModel`.
        """
        model = cls()
        model._tree = root
        model._initial_state = tuple(root.history) if root.history else ()
        model._current_state = model._initial_state
        model._state_descriptions[model._initial_state] = initial_state_description

        # Recursively walk the tree
        model._ingest_state_node(root, initial_state_description)

        # Collect all unique states
        all_states = set()
        all_states.add(model._initial_state)
        for s, ap_dict in model._transitions.items():
            all_states.add(s)
            for _, outcomes in ap_dict.items():
                for _, ns in outcomes:
                    all_states.add(ns)
        for s in model._empowerment:
            all_states.add(s)
        model._states = sorted(all_states, key=lambda s: (len(s), s))

        # Determine max action counts
        max_ra = 0
        max_hr = 0
        for s, ap_dict in model._transitions.items():
            ra_indices = {ap[0] for ap in ap_dict}
            max_ra = max(max_ra, max(ra_indices) + 1 if ra_indices else 0)
            for ap in ap_dict:
                hr_indices = {ap[1]}
                max_hr = max(max_hr, max(hr_indices) + 1 if hr_indices else 0)
        model._n_robot_actions = max(max_ra, 1)
        model._n_humans_reactions = max(max_hr, 1)
        # action_space.n is set to the max per-agent count for gym compatibility.
        # The actual per-agent counts are provided by num_actions_per_agent.
        model.action_space = gym.spaces.Discrete(
            max(model._n_robot_actions, model._n_humans_reactions)
        )

        return model

    def _ingest_state_node(
        self, node: TreeNode, initial_state_description: str
    ) -> None:
        """Recursively process a state node and its children."""
        state_key = tuple(node.history) if node.history else ()

        if node.empowerment_estimate is not None:
            self._empowerment[state_key] = node.empowerment_estimate

        if not node.children:
            return

        # Children of a state node are robotaction nodes
        robot_actions: List[str] = []
        humans_reactions_per_ra: Dict[int, List[str]] = {}

        for ra_idx, (ra_label, _, ra_node) in enumerate(node.children):
            robot_actions.append(ra_label)
            hr_labels: List[str] = []

            for hr_idx, (hr_label, _, hr_node) in enumerate(ra_node.children):
                hr_labels.append(hr_label)

                # Children of a humansreaction node are consequence/state nodes
                for cons_label, prob, cons_node in hr_node.children:
                    next_state_key = tuple(cons_node.history)
                    self._transitions[state_key][(ra_idx, hr_idx)].append(
                        (prob, next_state_key)
                    )
                    # Build description for the next state
                    desc = f"{initial_state_description} → " + " → ".join(
                        cons_node.history
                    )
                    self._state_descriptions[next_state_key] = desc

                    # Recurse
                    self._ingest_state_node(cons_node, initial_state_description)

            humans_reactions_per_ra[ra_idx] = hr_labels

        self._robot_action_labels[state_key] = robot_actions
        self._humans_reaction_labels[state_key] = humans_reactions_per_ra

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def get_state(self) -> Any:
        return self._current_state

    def set_state(self, state: Any) -> None:
        self._current_state = state

    def transition_probabilities(
        self, state: Any, actions: List[int]
    ) -> Optional[List[Tuple[float, Any]]]:
        if state not in self._transitions:
            return None
        ap_dict = self._transitions[state]
        if not ap_dict:
            return None

        ra_idx = actions[0] if len(actions) > 0 else 0
        hr_idx = actions[1] if len(actions) > 1 else 0
        key = (ra_idx, hr_idx)
        if key not in ap_dict:
            return None
        outcomes = ap_dict[key]
        if not outcomes:
            return None
        return outcomes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_state = self._initial_state
        return self._current_state, {}

    def V_r_estimate(self, state: Any) -> float:
        """Return the LLM-estimated empowerment value for a (terminal) state.

        If the state has no stored estimate, returns 0.0 (equivalent to
        ``log2(1)``).  The value is ``log2`` of the raw estimate so that it
        behaves like a channel capacity / empowerment measure.
        """
        raw = self._empowerment.get(state)
        if raw is None or raw <= 0:
            return 0.0
        return math.log2(raw)

    # ------------------------------------------------------------------
    # Human-readable helpers
    # ------------------------------------------------------------------

    def state_description(self, state: Any = None) -> str:
        """Return a human-readable description of a state."""
        if state is None:
            state = self._current_state
        return self._state_descriptions.get(state, str(state))

    def robot_action_labels(self, state: Any = None) -> List[str]:
        """Return robot action descriptions available in *state*."""
        if state is None:
            state = self._current_state
        return self._robot_action_labels.get(state, [])

    def humans_reaction_labels(
        self, state: Any = None, robot_action_index: int = 0
    ) -> List[str]:
        """Return humans' reaction descriptions for a given robot action."""
        if state is None:
            state = self._current_state
        per_ra = self._humans_reaction_labels.get(state, {})
        return per_ra.get(robot_action_index, [])

    @property
    def states(self) -> List[tuple]:
        """All known states."""
        return list(self._states)

    @property
    def terminal_states(self) -> List[tuple]:
        """States with empowerment estimates (terminal)."""
        return [s for s in self._states if s in self._empowerment]

    @property
    def human_agent_indices(self) -> List[int]:
        return [1]

    @property
    def robot_agent_indices(self) -> List[int]:
        return [0]


# ---------------------------------------------------------------------------
# Goal classes for NL world models
# ---------------------------------------------------------------------------


class SingleStateGoal(PossibleGoal):
    """A goal that is achieved iff the current state equals a target state."""

    def __init__(self, env: Any, target_state: tuple):
        super().__init__(env)
        self.target_state = target_state
        self._hash = hash(("SingleStateGoal", target_state))
        super()._freeze()

    def is_achieved(self, state) -> int:
        return 1 if state == self.target_state else 0

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SingleStateGoal)
            and self.target_state == other.target_state
        )

    def __repr__(self) -> str:
        return f"SingleStateGoal({self.target_state!r})"


class AllSingleStatesGoalGenerator(PossibleGoalGenerator):
    """Yields one :class:`SingleStateGoal` for every state in the world model.

    Each goal is weighted uniformly (``1 / n_states``), so the aggregate
    empowerment metric treats all states as equally important objectives.
    """

    def __init__(self, env: "NLWorldModel"):
        super().__init__(env)
        self._goals: List[SingleStateGoal] = [
            SingleStateGoal(env, s) for s in env.states
        ]
        self._weight = 1.0 / len(self._goals) if self._goals else 1.0

    def generate(
        self, state, human_agent_index: int
    ) -> Iterator[Tuple[PossibleGoal, float]]:
        for g in self._goals:
            yield g, self._weight
