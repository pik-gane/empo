"""
Hierarchical modeler: extends the tree builder with higher-level context
awareness, success/failure detection, and consequence matching.

This module produces :class:`~empo.hierarchical.HierarchicalWorldModel`
instances by building two (or more) levels of NL world models and connecting
them with an :class:`NLLevelMapper`.

The fine-level model is built *lazily*: only when a coarse-level action is
actually taken in ``step()`` does the code query the LLM to construct the
corresponding fine-level tree.  This avoids unnecessary detail for actions
not actually taken.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel
from empo.hierarchical.level_mapper import LevelMapper
from empo.simple_hierarchical_llm_modeler.llm_connector import LLMConnector
from empo.simple_hierarchical_llm_modeler.nl_world_model import NLWorldModel
from empo.simple_hierarchical_llm_modeler.prompts import (
    hierarchical_status_prompt,
    match_consequence_prompt,
)
from empo.simple_hierarchical_llm_modeler.tree_builder import (
    TreeNode,
    _parse_json_object,
    build_tree,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hierarchical status check
# ---------------------------------------------------------------------------


def check_hierarchical_status(
    llm: LLMConnector,
    higher_level_context: str,
    higher_level_action: str,
    initial_state: str,
    history: List[str],
) -> str:
    """Ask the LLM whether the current state is success / failure / in-progress.

    Returns:
        One of ``"success"``, ``"failure"``, or ``"still in progress"``.
    """
    prompt = hierarchical_status_prompt(
        higher_level_context, higher_level_action, initial_state, history
    )
    raw = llm.query(prompt)
    try:
        data = _parse_json_object(raw)
        status = data.get("status", "still in progress")
    except ValueError:
        status = "still in progress"

    if not isinstance(status, str):
        return "still in progress"
    normalised = status.strip().lower()
    if normalised in ("success", "failure", "still in progress"):
        return normalised
    return "still in progress"


def match_consequence(
    llm: LLMConnector,
    higher_level_context: str,
    higher_level_action: str,
    known_consequences: List[str],
    status: str,
    initial_state: str,
    history: List[str],
) -> Tuple[Optional[int], Optional[str]]:
    """Match current outcome to a known higher-level consequence.

    Returns:
        ``(match_index, new_description)`` where *match_index* is a 0-based
        index into *known_consequences* if there is a match (else ``None``),
        and *new_description* is a novel consequence string if no match
        (else ``None``).
    """
    prompt = match_consequence_prompt(
        higher_level_context,
        higher_level_action,
        known_consequences,
        status,
        initial_state,
        history,
    )
    raw = llm.query(prompt)
    try:
        data = _parse_json_object(raw)
    except ValueError:
        return None, f"Unmatched {status}"

    match_idx = data.get("match")
    new_cons = data.get("new_consequence")
    if match_idx is not None:
        # LLM returns 1-based index; convert to 0-based and validate.
        try:
            idx = int(match_idx) - 1
        except (TypeError, ValueError):
            idx = None
        if idx is not None and 0 <= idx < len(known_consequences):
            return idx, None
    return None, new_cons or f"Novel {status} outcome"


# ---------------------------------------------------------------------------
# NLLevelMapper
# ---------------------------------------------------------------------------


class NLLevelMapper(LevelMapper):
    """LevelMapper connecting two NL world models with LLM-based status checks.

    ``super_state`` returns the coarse root state while the higher-level
    activity is ``"still in progress"`` and switches to a terminal coarse
    state once the LLM reports ``"success"`` or ``"failure"``.

    ``return_control`` returns ``True`` exactly when the LLM says the
    higher-level activity is ``"success"`` or ``"failure"``.
    """

    def __init__(
        self,
        coarse_model: NLWorldModel,
        fine_model: NLWorldModel,
        *,
        llm: Optional[LLMConnector] = None,
        higher_level_context: str = "",
        higher_level_action: str = "",
        initial_state_description: str = "",
    ) -> None:
        super().__init__(coarse_model, fine_model)
        self._llm = llm
        self._higher_level_context = higher_level_context
        self._higher_level_action = higher_level_action
        self._initial_state_description = initial_state_description
        # Cache the last LLM status check result per fine state
        self._status_cache: Dict[tuple, str] = {}

    def _get_status(self, fine_state: Any) -> str:
        """Return the hierarchical status for *fine_state*, with caching."""
        key = fine_state if isinstance(fine_state, tuple) else (fine_state,)
        if key in self._status_cache:
            return self._status_cache[key]
        if self._llm is None:
            return "still in progress"
        history = list(key) if isinstance(fine_state, tuple) else []
        status = check_hierarchical_status(
            self._llm,
            self._higher_level_context,
            self._higher_level_action,
            self._initial_state_description,
            history,
        )
        self._status_cache[key] = status
        return status

    def super_state(self, fine_state: Any) -> Any:
        """Map fine state to coarse state.

        While the higher-level activity is ``"still in progress"``, returns
        the coarse root state (empty tuple).  On ``"success"`` or
        ``"failure"``, returns a distinct coarse terminal marker so that
        macro-level decisions can condition on progress.
        """
        if fine_state is None or fine_state == ():
            return ()
        status = self._get_status(fine_state)
        if status == "still in progress":
            return ()
        # Return a distinguishable coarse state for completed activities
        return ("_completed", status)

    def super_agent(self, fine_agent_index: int) -> int:
        return fine_agent_index

    def is_feasible(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
    ) -> bool:
        return True

    def is_abort(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
    ) -> bool:
        return False

    def return_control(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
        fine_successor_state: Any,
    ) -> bool:
        """Return control to the coarse level iff the LLM says success/failure."""
        status = self._get_status(fine_successor_state)
        return status in ("success", "failure")


# ---------------------------------------------------------------------------
# Lazy two-level world model
# ---------------------------------------------------------------------------


class LazyTwoLevelModel:
    """Two-level hierarchical model that builds fine models on demand.

    Only the coarse model is built upfront.  Fine-level models are
    constructed lazily the first time a coarse-level action is actually
    taken (via :meth:`get_fine_model`).

    This class intentionally does **not** subclass
    :class:`HierarchicalWorldModel` because the latter requires all levels
    to be provided at construction time.  Instead it provides a compatible
    interface and can produce a :class:`HierarchicalWorldModel` snapshot
    via :meth:`to_hierarchical` once a fine model has been built.
    """

    def __init__(
        self,
        llm: LLMConnector,
        initial_state_description: str,
        coarse_model: NLWorldModel,
        *,
        fine_n_steps: int = 2,
        n_robotactions: int = 3,
        n_humansreactions: int = 3,
        n_consequences: int = 2,
        fine_time_horizon: Optional[str] = None,
        on_update: Optional[Callable[[Optional["TreeNode"], str], None]] = None,
    ) -> None:
        self.llm = llm
        self.initial_state_description = initial_state_description
        self.coarse_model = coarse_model
        self._fine_n_steps = fine_n_steps
        self._n_robotactions = n_robotactions
        self._n_humansreactions = n_humansreactions
        self._n_consequences = n_consequences
        self._fine_time_horizon = fine_time_horizon
        self._on_update = on_update

        # Cache: (coarse action, human reaction) -> (fine_model, mapper)
        self._fine_cache: Dict[Tuple[str, Optional[str]], Tuple[NLWorldModel, NLLevelMapper]] = {}

        # Source trees (coarse set by build_two_level_model, fine set lazily)
        self._coarse_tree: Optional[TreeNode] = None
        self._fine_trees: Dict[Tuple[str, Optional[str]], TreeNode] = {}

        # Currently active fine model / mapper (set by get_fine_model)
        self._active_fine: Optional[NLWorldModel] = None
        self._active_mapper: Optional[NLLevelMapper] = None
        self._active_action: Optional[str] = None

    @property
    def num_levels(self) -> int:
        return 2

    def coarsest(self) -> NLWorldModel:
        return self.coarse_model

    def finest(self) -> Optional[NLWorldModel]:
        """Return the currently active fine model, or ``None`` if none built."""
        return self._active_fine

    def get_fine_model(
        self,
        coarse_action_label: str,
        coarse_human_reaction_label: Optional[str] = None,
        on_update: Optional[Callable[[Optional[TreeNode], str], None]] = None,
    ) -> Tuple[NLWorldModel, NLLevelMapper]:
        """Get (or lazily build) the fine model for *coarse_action_label*.

        Args:
            coarse_action_label: Human-readable description of the coarse
                robot action that was chosen.
            coarse_human_reaction_label: Human-readable description of the
                coarse human reaction that was selected.  When provided,
                the fine-level context states this as the *actual* human
                choice rather than listing all possible reactions.
            on_update: Optional callback override for live rendering of
                the fine-level tree.  If ``None``, uses the callback passed
                at construction time.

        Returns:
            ``(fine_model, mapper)`` pair.
        """
        cb = on_update if on_update is not None else self._on_update
        cache_key = (coarse_action_label, coarse_human_reaction_label)
        if cache_key in self._fine_cache:
            fine_model, mapper = self._fine_cache[cache_key]
        else:
            # Build a rich higher-level context including coarse human reactions
            human_context = ""
            if coarse_human_reaction_label:
                # Specific human reaction was selected
                human_context = (
                    f" Reacting to that decision, the humans have formed this plan: "
                    f"{coarse_human_reaction_label}."
                )
            elif self._coarse_tree is not None:
                for label, _, child in self._coarse_tree.children:
                    if label == coarse_action_label:
                        # child is the robotaction node; its children are humansreaction nodes
                        if child.children:
                            reactions = [lbl for lbl, _, _ in child.children]
                            human_context = (
                                " At the higher level, humans may react by: "
                                + "; OR ".join(reactions)
                                + "."
                            )
                        break

            higher_ctx = (
                f"The robot is executing a HIGHER-LEVEL PLAN: "
                f"{coarse_action_label}.{human_context}\n"
                f"You must now explore the SUB-ACTIVITIES within this "
                f"higher-level plan, on a SHORTER TIME HORIZON. The actions, "
                f"reactions, and consequences should be about WHAT HAPPENS "
                f"DURING this activity — not about the activity itself.\n"
                f"For example, if the higher-level plan is 'drive passengers "
                f"to destination', fine-level actions might be categories like "
                f"'handle luggage and boarding', 'navigate the route', "
                f"'manage in-ride communication'."
            )
            fine_tree = build_tree(
                llm=self.llm,
                initial_state_description=self.initial_state_description,
                n_steps=self._fine_n_steps,
                n_robotactions=self._n_robotactions,
                n_humansreactions=self._n_humansreactions,
                n_consequences=self._n_consequences,
                higher_level_context=higher_ctx,
                time_horizon=self._fine_time_horizon,
                on_update=cb,
            )
            fine_model = NLWorldModel.from_tree(
                fine_tree, self.initial_state_description
            )
            mapper = NLLevelMapper(
                self.coarse_model,
                fine_model,
                llm=self.llm,
                higher_level_context=self.initial_state_description,
                higher_level_action=coarse_action_label,
                initial_state_description=self.initial_state_description,
            )
            self._fine_cache[cache_key] = (fine_model, mapper)
            self._fine_trees[cache_key] = fine_tree

        self._active_fine = fine_model
        self._active_mapper = mapper
        self._active_action = coarse_action_label
        return fine_model, mapper

    def to_hierarchical(self) -> HierarchicalWorldModel:
        """Snapshot the current state as a :class:`HierarchicalWorldModel`.

        Requires that a fine model has already been built via
        :meth:`get_fine_model`.

        Raises:
            RuntimeError: If no fine model has been built yet.
        """
        if self._active_fine is None or self._active_mapper is None:
            raise RuntimeError("No fine model built yet.  Call get_fine_model() first.")
        return HierarchicalWorldModel(
            levels=[self.coarse_model, self._active_fine],
            mappers=[self._active_mapper],
        )


def build_two_level_model(
    llm: LLMConnector,
    initial_state_description: str,
    *,
    coarse_n_steps: int = 1,
    fine_n_steps: int = 2,
    n_robotactions: int = 3,
    n_humansreactions: int = 3,
    n_consequences: int = 2,
    coarse_time_horizon: Optional[str] = None,
    fine_time_horizon: Optional[str] = None,
    on_update: Optional[Callable[[Optional[TreeNode], str], None]] = None,
) -> LazyTwoLevelModel:
    """Build a lazy two-level hierarchical model from NL descriptions.

    Only the **coarse** (level-0) tree is built immediately.  Fine-level
    models are constructed on demand via
    :meth:`LazyTwoLevelModel.get_fine_model` when a coarse-level action
    is actually taken.

    Args:
        llm: LLM connector.
        initial_state_description: Starting situation in natural language.
        coarse_n_steps: Depth of the coarse tree.
        fine_n_steps: Depth of each fine tree (built lazily).
        n_robotactions: Robot actions per expansion.
        n_humansreactions: Human reactions per expansion.
        n_consequences: Consequences per expansion.

    Returns:
        A :class:`LazyTwoLevelModel` with the coarse model ready and fine
        models built on demand.
    """
    coarse_tree = build_tree(
        llm=llm,
        initial_state_description=initial_state_description,
        n_steps=coarse_n_steps,
        n_robotactions=n_robotactions,
        n_humansreactions=n_humansreactions,
        n_consequences=n_consequences,
        time_horizon=coarse_time_horizon,
        on_update=on_update,
    )
    coarse_model = NLWorldModel.from_tree(coarse_tree, initial_state_description)

    model = LazyTwoLevelModel(
        llm=llm,
        initial_state_description=initial_state_description,
        coarse_model=coarse_model,
        fine_n_steps=fine_n_steps,
        n_robotactions=n_robotactions,
        n_humansreactions=n_humansreactions,
        n_consequences=n_consequences,
        fine_time_horizon=fine_time_horizon,
        on_update=on_update,
    )
    model._coarse_tree = coarse_tree
    return model
