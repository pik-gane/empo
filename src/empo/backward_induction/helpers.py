"""
Helper functions for backward induction computations.

This module provides utility functions used by both Phase 1 (human policy prior)
and Phase 2 (robot policy) backward induction algorithms.
"""

import numpy as np
import numpy.typing as npt
import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any, TypeAlias, TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
else:
    # For runtime type aliases that need PossibleGoal
    from empo.possible_goal import PossibleGoal

T = TypeVar('T')

# Type aliases for complex types used throughout
State: TypeAlias = Any  # State is typically a hashable tuple from WorldModel.get_state()
ActionProfile = List[int]
TransitionData = Tuple[Tuple[int, ...], List[float], List[State]]  # (action_profile, probs, successor_states)
VhValues = List[List[Dict['PossibleGoal', float]]]  # Indexed as Vh_values[state_index][agent_index][goal]
VhValuesSmall = List[List[npt.NDArray[np.float16]]]  # Indexed as Vh_values[state_index][agent_index][goal_index]

# Slice cache for a worker's batch of states.
# Structure: slice_cache[state_index][action_profile_index][goal] -> array of attainment values.
# - Dict keyed by state_index (states in a batch are not consecutive)
# - List indexed by action_profile_index (consecutive 0..num_action_profiles-1, O(1) access)
# - Dict keyed by goal (flexible set of goals)
SliceCache: TypeAlias = Dict[int, List[Dict["PossibleGoal", npt.NDArray[np.int8]]]]

# Sliced attainment cache: stores worker slice caches indexed by slice_id.
# slice_id is typically a tuple identifying the batch (e.g., batch contents hash or first state index).
# This avoids merging overhead - each slice stays separate and Phase 2 workers access by slice_id.
SliceId: TypeAlias = Tuple[int, ...]  # Tuple of state indices in the slice


class SlicedAttainmentCache:
    """Cache for goal attainment values organized by worker slices.
    
    Each worker in parallel mode creates a SliceCache for its batch of states.
    These are stored separately (not merged) and accessed by slice_id.
    Both Phase 1 and Phase 2 use the same slice assignments for consistency.
    
    Can optionally use disk-based storage via DiskBasedDAG to avoid memory overhead.
    
    Structure:
        slices[slice_id] = SliceCache
        SliceCache[state_index][action_profile_index][goal] = attainment_array
    """
    
    def __init__(self, num_action_profiles: int):
        """Initialize empty sliced cache.
        
        Args:
            num_action_profiles: Number of action profiles (needed to create lists)
        """
        self.num_action_profiles = num_action_profiles
        self.slices: Dict[SliceId, SliceCache] = {}
        self._disk_dag: Optional[Any] = None  # Optional DiskBasedDAG for disk storage
    
    def create_slice_cache(self, state_indices: List[int]) -> SliceCache:
        """Create an empty slice cache for given state indices.
        
        Args:
            state_indices: List of state indices this slice will contain
            
        Returns:
            Empty SliceCache with pre-allocated lists for each state
        """
        return {
            state_idx: [{} for _ in range(self.num_action_profiles)]
            for state_idx in state_indices
        }
    
    def store_slice(self, slice_id: SliceId, cache: SliceCache) -> None:
        """Store a worker's completed slice cache.
        
        Args:
            slice_id: Identifier for this slice (tuple of state indices)
            cache: The worker's completed slice cache
        """
        self.slices[slice_id] = cache
    
    def get_slice(self, slice_id: SliceId) -> Optional[SliceCache]:
        """Get a slice cache by ID.
        
        Args:
            slice_id: The slice identifier
            
        Returns:
            The slice cache if found, None otherwise
        """
        return self.slices.get(slice_id)
    
    def get(self, state_index: int, action_profile_index: int, goal: "PossibleGoal") -> Optional[npt.NDArray[np.int8]]:
        """Look up a cached attainment array across all slices.
        
        Args:
            state_index: Global state index
            action_profile_index: Action profile index
            goal: The goal to look up
            
        Returns:
            Cached attainment array if found, None otherwise
        """
        # First check in-memory slices
        for slice_cache in self.slices.values():
            if state_index in slice_cache:
                result = slice_cache[state_index][action_profile_index].get(goal)
                if result is not None:
                    return result
        
        # If using disk storage, try loading from disk
        if self._disk_dag is not None and hasattr(self._disk_dag, 'load_cache_slice'):
            # Try to find which timestep contains this state
            # This requires level_fct - for now, just return None
            # Phase 2 will need to handle disk cache lookup differently
            pass
        
        return None
    
    def total_entries(self) -> int:
        """Count total number of cached entries across all slices."""
        count = 0
        for slice_cache in self.slices.values():
            for state_idx, ap_list in slice_cache.items():
                for goal_dict in ap_list:
                    count += len(goal_dict)
        return count
    
    def num_states(self) -> int:
        """Count total number of states across all slices."""
        return sum(len(slice_cache) for slice_cache in self.slices.values())


def make_slice_id(state_indices: List[int]) -> SliceId:
    """Create a slice ID from a list of state indices.
    
    Args:
        state_indices: List of state indices in the slice
        
    Returns:
        Tuple that uniquely identifies this slice
    """
    return tuple(sorted(state_indices))


class DefaultBelievedOthersPolicy:
    """Callable that returns uniform distribution over other humans' actions.
    
    Precomputes results for each human agent index at initialization time,
    since the result only depends on agent_index (not state or action).
    Robot agent positions are set to -1 (placeholder) since they will be
    overwritten by the caller when iterating over robot action profiles.
    
    Usage:
        # Create once at start of Phase 1
        believed_others = DefaultBelievedOthersPolicy(
            num_agents=4, num_actions_per_agent=[5, 5, 5, 5], 
            human_agent_indices=[0, 2], robot_agent_indices=[1, 3]
        )
        # Call like the old function (state and action are ignored)
        distribution = believed_others(state, agent_index=0, action=2, ...)
    """
    
    def __init__(
        self,
        num_agents: int,
        num_actions_per_agent: List[int],
        human_agent_indices: List[int],
        robot_agent_indices: List[int]
    ):
        """Precompute believed others policy for each human agent.
        
        Args:
            num_agents: Total number of agents
            num_actions_per_agent: Number of actions for each agent
            human_agent_indices: List of agent indices that are humans
            robot_agent_indices: List of agent indices that are robots
        """
        self.num_agents = num_agents
        self.num_actions_per_agent = num_actions_per_agent
        self.robot_agent_indices = robot_agent_indices
        
        # Precompute for each human agent index, None for non-humans
        self._cached: List[Optional[List[Tuple[float, npt.NDArray[np.int64]]]]] = [
            None for _ in range(num_agents)
        ]
        
        robot_set = set(robot_agent_indices)
        
        for agent_index in human_agent_indices:
            # Compute uniform probability over other humans' action profiles
            other_human_total = 1
            for idx in human_agent_indices:
                if idx != agent_index:
                    other_human_total *= num_actions_per_agent[idx]
            uniform_p = 1.0 / other_human_total if other_human_total > 1 else 1.0
            
            # Each action profile for the other human agents gets the same probability.
            # The agent's own action and robot actions are set to -1 since they will be overwritten.
            self._cached[agent_index] = [
                (uniform_p, np.array(action_profile, dtype=np.int64))
                for action_profile in product(*[
                    [-1] if (idx == agent_index or idx in robot_set) else list(range(num_actions_per_agent[idx]))
                    for idx in range(num_agents)
                ])
            ]
    
    def __call__(
        self,
        _unused_state: State,
        agent_index: int,
        _unused_action: int,
    ) -> List[Tuple[float, npt.NDArray[np.int64]]]:
        """Return precomputed distribution for the given agent index.
        
        Args match the old function signature for API compatibility.
        state and action are ignored (result doesn't depend on them).
        num_agents, num_actions, robot_agent_indices should match init values.
        
        Returns:
            List of (probability, action_profile) tuples for other agents.
        """
        result = self._cached[agent_index]
        if result is None:
            raise ValueError(f"Agent {agent_index} is not a human agent")
        return result


def collect_goals_by_human(
    goal_generator: "PossibleGoalGenerator",
    human_agent_indices: List[int],
    state: State
) -> Dict[int, List[Tuple["PossibleGoal", float]]]:
    """
    Collect all possible goals for each human agent from a generator.
    
    This function iterates through the goal generator for each human agent
    and collects all (goal, weight) pairs into a dictionary keyed by human
    agent index.
    
    Args:
        goal_generator: A PossibleGoalGenerator that yields (goal, weight) pairs.
        human_agent_indices: List of agent indices that are humans.
        state: Current world state (passed to the generator).
    
    Returns:
        Dict mapping each human agent index to a list of (PossibleGoal, weight)
        tuples as returned by the generator for that human.
    
    Example:
        >>> generator = AllCellsGenerator(env)
        >>> goals_by_human = collect_goals_by_human(generator, [0, 2], state)
        >>> # goals_by_human[0] contains all (goal, weight) pairs for human 0
        >>> # goals_by_human[2] contains all (goal, weight) pairs for human 2
    """
    return {
        human_idx: list(goal_generator.generate(state, human_idx))
        for human_idx in human_agent_indices
    }


def combine_action_profiles(
    human_action_profile: List[int] | npt.NDArray[np.int64],
    robot_action_profile: List[int] | npt.NDArray[np.int64],
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    num_agents: Optional[int] = None
) -> List[int]:
    """
    Combine human and robot action profiles into a full action profile for all agents.
    
    This function interleaves actions from humans and robots according to their
    respective agent indices to produce a complete action profile where
    full_profile[agent_index] contains the action for that agent.
    
    Args:
        human_action_profile: List/array of actions for human agents, in the order
                             specified by human_agent_indices.
        robot_action_profile: List/array of actions for robot agents, in the order
                             specified by robot_agent_indices.
        human_agent_indices: List of agent indices that are humans.
        robot_agent_indices: List of agent indices that are robots.
        num_agents: Total number of agents. If None, computed from indices.
    
    Returns:
        List[int]: Full action profile of length num_agents,
                  where each position corresponds to the agent at that index.
    
    Example:
        >>> # 4 agents: agents 0,2 are humans, agents 1,3 are robots
        >>> human_actions = [3, 1]  # human 0 does action 3, human 2 does action 1
        >>> robot_actions = [2, 0]  # robot 1 does action 2, robot 3 does action 0
        >>> combine_action_profiles(human_actions, robot_actions, [0, 2], [1, 3])
        [3, 2, 1, 0]  # full_profile[i] = action of agent i
    """
    if num_agents is None:
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
    
    # Use numpy for efficient scatter operation
    full_profile = np.empty(num_agents, dtype=np.int64)
    full_profile[human_agent_indices] = human_action_profile
    full_profile[robot_agent_indices] = robot_action_profile
    
    return full_profile.tolist()


def combine_action_profiles_batch(
    human_action_profiles: npt.NDArray[np.int64],
    robot_action_profile: List[int] | npt.NDArray[np.int64],
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    num_agents: Optional[int] = None
) -> npt.NDArray[np.int64]:
    """
    Combine multiple human action profiles with a single robot action profile.
    
    Vectorized version for efficiently processing many human action profiles
    at once (e.g., from profile_distribution) with a fixed robot action.
    
    Args:
        human_action_profiles: Array of shape (N, num_humans) containing N human
                              action profiles to combine.
        robot_action_profile: Single robot action profile to use for all combinations.
        human_agent_indices: List of agent indices that are humans.
        robot_agent_indices: List of agent indices that are robots.
        num_agents: Total number of agents. If None, computed from indices.
    
    Returns:
        np.ndarray: Array of shape (N, num_agents) containing the combined profiles.
    
    Example:
        >>> # 3 agents: agent 0 is human, agents 1,2 are robots
        >>> human_profiles = np.array([[0], [1], [2], [3]])  # 4 human action profiles
        >>> robot_actions = [5, 6]  # robots do actions 5 and 6
        >>> combine_action_profiles_batch(human_profiles, robot_actions, [0], [1, 2])
        array([[0, 5, 6],
               [1, 5, 6],
               [2, 5, 6],
               [3, 5, 6]])
    """
    if num_agents is None:
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
    
    n_profiles = len(human_action_profiles)
    
    # Allocate output array
    full_profiles = np.empty((n_profiles, num_agents), dtype=np.int64)
    
    # Scatter human actions (vectorized across all profiles)
    full_profiles[:, human_agent_indices] = human_action_profiles
    
    # Broadcast robot actions to all profiles
    full_profiles[:, robot_agent_indices] = robot_action_profile
    
    return full_profiles


def combine_profile_distributions(
    human_probs: npt.NDArray[np.floating[Any]],
    human_profiles: npt.NDArray[np.int64],
    robot_probs: npt.NDArray[np.floating[Any]],
    robot_profiles: npt.NDArray[np.int64],
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    num_agents: Optional[int] = None,
    min_prob: float = 0.0
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.int64]]:
    """
    Combine human and robot profile distributions into a joint distribution.
    
    Computes the outer product of independent human and robot distributions,
    returning the joint probability distribution over full action profiles.
    
    Args:
        human_probs: Array of shape (N_h,) with probabilities for each human profile.
        human_profiles: Array of shape (N_h, num_humans) with human action profiles.
        robot_probs: Array of shape (N_r,) with probabilities for each robot profile.
        robot_profiles: Array of shape (N_r, num_robots) with robot action profiles.
        human_agent_indices: List of agent indices that are humans.
        robot_agent_indices: List of agent indices that are robots.
        num_agents: Total number of agents. If None, computed from indices.
        min_prob: Minimum probability threshold. Profiles with prob <= min_prob
                 are filtered out. Default 0.0 keeps all non-zero entries.
    
    Returns:
        Tuple of:
        - joint_probs: Array of shape (M,) with joint probabilities, where
          M <= N_h * N_r (filtered by min_prob).
        - full_profiles: Array of shape (M, num_agents) with combined profiles.
    
    Example:
        >>> # 3 agents: agent 0 is human, agents 1,2 are robots
        >>> human_probs = np.array([0.3, 0.7])
        >>> human_profiles = np.array([[0], [1]])  # human does action 0 or 1
        >>> robot_probs = np.array([0.4, 0.6])
        >>> robot_profiles = np.array([[2, 3], [4, 5]])  # robots do (2,3) or (4,5)
        >>> probs, profiles = combine_profile_distributions(
        ...     human_probs, human_profiles, robot_probs, robot_profiles,
        ...     [0], [1, 2])
        >>> probs  # [0.3*0.4, 0.3*0.6, 0.7*0.4, 0.7*0.6]
        array([0.12, 0.18, 0.28, 0.42])
        >>> profiles
        array([[0, 2, 3],
               [0, 4, 5],
               [1, 2, 3],
               [1, 4, 5]])
    """
    if num_agents is None:
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
    
    n_human = len(human_probs)
    n_robot = len(robot_probs)
    
    # Compute outer product of probabilities: shape (N_h, N_r)
    joint_probs_2d = np.outer(human_probs, robot_probs)
    
    # Flatten to 1D
    joint_probs_flat = joint_probs_2d.ravel()
    
    # Filter by minimum probability if requested
    if min_prob > 0.0:
        mask = joint_probs_flat > min_prob
        joint_probs_flat = joint_probs_flat[mask]
        
        # Get indices of non-filtered entries
        valid_indices = np.where(mask)[0]
        human_idx = valid_indices // n_robot
        robot_idx = valid_indices % n_robot
    else:
        # All combinations
        human_idx = np.repeat(np.arange(n_human), n_robot)
        robot_idx = np.tile(np.arange(n_robot), n_human)
    
    n_combinations = len(joint_probs_flat)
    
    # Build full profiles array
    full_profiles = np.empty((n_combinations, num_agents), dtype=np.int64)
    full_profiles[:, human_agent_indices] = human_profiles[human_idx]
    full_profiles[:, robot_agent_indices] = robot_profiles[robot_idx]
    
    return joint_probs_flat, full_profiles


def combine_profile_distributions_to_indices(
    human_probs: npt.NDArray[np.floating[Any]],
    human_profiles: npt.NDArray[np.int64],
    robot_probs: npt.NDArray[np.floating[Any]],
    robot_profiles: npt.NDArray[np.int64],
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    action_powers: npt.NDArray[np.int64],
    num_agents: Optional[int] = None,
    min_prob: float = 0.0
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.int64]]:
    """
    Combine profile distributions and directly compute action profile indices.
    
    This is an optimized version that skips building the full profiles array
    and directly computes the transition indices needed for value lookups.
    
    Args:
        human_probs: Array of shape (N_h,) with probabilities for each human profile.
        human_profiles: Array of shape (N_h, num_humans) with human action profiles.
        robot_probs: Array of shape (N_r,) with probabilities for each robot profile.
        robot_profiles: Array of shape (N_r, num_robots) with robot action profiles.
        human_agent_indices: List of agent indices that are humans.
        robot_agent_indices: List of agent indices that are robots.
        action_powers: Precomputed array [1, num_actions, num_actions^2, ...] for
                      converting profiles to flat indices.
        num_agents: Total number of agents. If None, computed from indices.
        min_prob: Minimum probability threshold.
    
    Returns:
        Tuple of:
        - joint_probs: Array of shape (M,) with joint probabilities.
        - profile_indices: Array of shape (M,) with flat action profile indices,
          computed as sum(profile[i] * action_powers[i]).
    
    Example:
        >>> # Used in backward induction to directly index transitions
        >>> probs, indices = combine_profile_distributions_to_indices(
        ...     human_probs, human_profiles, robot_probs, robot_profiles,
        ...     human_agent_indices, robot_agent_indices, action_powers)
        >>> for prob, idx in zip(probs, indices):
        ...     _, next_probs, next_states = transitions[state_index][idx]
        ...     # ... compute expected values
    """
    # Validate or infer the total number of agents.
    if num_agents is not None:
        inferred_num_agents = len(human_agent_indices) + len(robot_agent_indices)
        if num_agents != inferred_num_agents:
            raise ValueError(
                f"Inconsistent num_agents: expected {inferred_num_agents} "
                f"from agent indices, but got {num_agents}."
            )
    else:
        # Infer num_agents for potential future use or debugging.
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
    
    n_human = len(human_probs)
    n_robot = len(robot_probs)
    
    # Compute outer product of probabilities
    joint_probs_2d = np.outer(human_probs, robot_probs)
    joint_probs_flat = joint_probs_2d.ravel()
    
    # Precompute contribution of each human profile to the final index
    # human_contrib[i] = sum(human_profiles[i, j] * action_powers[human_agent_indices[j]])
    human_powers = action_powers[human_agent_indices]
    human_contrib = human_profiles @ human_powers  # shape (N_h,)
    
    # Precompute contribution of each robot profile
    robot_powers = action_powers[robot_agent_indices]
    robot_contrib = robot_profiles @ robot_powers  # shape (N_r,)
    
    # Filter by minimum probability if requested
    if min_prob > 0.0:
        mask = joint_probs_flat > min_prob
        joint_probs_flat = joint_probs_flat[mask]
        
        valid_indices = np.where(mask)[0]
        human_idx = valid_indices // n_robot
        robot_idx = valid_indices % n_robot
    else:
        human_idx = np.repeat(np.arange(n_human), n_robot)
        robot_idx = np.tile(np.arange(n_robot), n_human)
    
    # Compute profile indices by adding contributions
    profile_indices = human_contrib[human_idx] + robot_contrib[robot_idx]
    
    return joint_probs_flat, profile_indices


def compute_dependency_levels_general(successors: List[List[int]]) -> List[List[int]]:
    """Compute dependency levels using general topological approach."""
    levels: List[List[int]] = []
    remaining_states = set(range(len(successors)))
    
    while remaining_states:
        # Find states with no dependencies to remaining states (terminal in remaining subgraph)
        current_level: List[int] = []
        for state_idx in remaining_states:
            if not any(succ in remaining_states for succ in successors[state_idx]):
                current_level.append(state_idx)
        
        if not current_level:
            raise ValueError("Circular dependency detected in state graph")
        
        levels.append(current_level)
        remaining_states -= set(current_level)
    
    return levels


def compute_dependency_levels_fast(
    states: List[State], 
    level_fct: Callable[[State], int],
    successors: Optional[List[List[int]]] = None
) -> Tuple[List[List[int]], Optional[Dict[int, int]], Optional[List[int]]]:
    """Compute dependency levels using level function for faster computation.
    
    Args:
        states: List of states to compute levels for
        level_fct: Function mapping state to its level value
        successors: Optional list of successor state indices for each state.
                   If provided, also computes max successor levels.
    
    Returns:
        Tuple of:
        - levels: List of levels, where each level is a list of state indices
        - max_successor_levels: If successors provided, dict mapping level value
          to the max level value of any successor of states at that level.
          None if successors not provided.
        - level_values: If successors provided, list of level values parallel to levels.
          None if successors not provided.
    """
    # Compute level values for all states
    level_values_array: List[int] = [level_fct(state) for state in states]
    
    # Group states by level value (descending order for backward induction)
    level_groups: Dict[int, List[int]] = defaultdict(list)
    for state_idx, level_val in enumerate(level_values_array):
        level_groups[level_val].append(state_idx)
    
    # Sort by level value (highest first for backward induction)
    # Note: state indices within each level are already sorted by construction
    sorted_levels = sorted(level_groups.keys(), reverse=True)
    levels = [level_groups[level_val] for level_val in sorted_levels]
    
    # Compute max successor level for each level if successors provided
    max_successor_levels: Optional[Dict[int, int]] = None
    if successors is not None:
        max_successor_levels = {}
        for level_val in sorted_levels:
            # Initialize to level_val + 1 for terminal states (no successors)
            # This ensures terminal states follow the k→k+1 pattern
            max_succ_level = level_val + 1
            for state_idx in level_groups[level_val]:
                for succ_idx in successors[state_idx]:
                    max_succ_level = max(max_succ_level, level_values_array[succ_idx])
            max_successor_levels[level_val] = max_succ_level
        
        return levels, max_successor_levels, sorted_levels
    
    return levels, None, None


def split_into_batches(items: List[int], num_batches: Optional[int]) -> List[List[int]]:
    """Split a list into approximately equal batches."""
    if num_batches is None or num_batches <= 1:
        return [items]
    
    batch_size = max(1, len(items) // num_batches)
    batches: List[List[int]] = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if batch:
            batches.append(batch)
    return batches


class SlicedList(Generic[T]):
    """List-like container that organizes values by level for memory-efficient archival.
    
    Values are stored in slices grouped by level value (from level_fct). When a slice
    is no longer needed, it can be archived to disk and removed from memory.
    
    Structure:
        _slices[level_value] = list of values for states at that level
        _state_to_slice[state_index] = (level_value, offset_in_slice)
    
    Usage:
        # Create from level information
        sliced_list = SlicedList.from_levels(states, level_fct)
        
        # Access by state index
        sliced_list[state_idx] = value
        value = sliced_list[state_idx]
        
        # Archive a level when no longer needed
        sliced_list.archive_slice(level_value, filepath, clear_memory=True)
    """
    
    def __init__(self):
        """Initialize empty sliced list."""
        self._slices: Dict[int, Optional[List[T]]] = {}
        self._state_to_slice: List[Tuple[int, int]] = []  # [(level_value, offset), ...]
        self._level_values: List[int] = []  # All level values in order
    
    @classmethod
    def from_levels(
        cls,
        states: List[State],
        level_fct: Callable[[State], int]
    ) -> 'SlicedList[T]':
        """Create a SlicedList organized by level values.
        
        Args:
            states: List of all states
            level_fct: Function mapping state to its level value
            
        Returns:
            SlicedList instance with allocated slices (values initialized to None)
        """
        sliced_list = cls()
        
        # Compute level values for all states
        level_values = [level_fct(state) for state in states]
        
        # Group state indices by level value
        level_groups: Dict[int, List[int]] = defaultdict(list)
        for state_idx, level_val in enumerate(level_values):
            level_groups[level_val].append(state_idx)
        
        # Store all unique level values
        sliced_list._level_values = sorted(level_groups.keys())
        
        # Create slices and build state_to_slice mapping
        sliced_list._state_to_slice = [(0, 0)] * len(states)
        
        for level_val, state_indices in level_groups.items():
            # Create slice for this level
            slice_size = len(state_indices)
            sliced_list._slices[level_val] = [None] * slice_size  # type: ignore
            
            # Map each state to its position in the slice
            for offset, state_idx in enumerate(state_indices):
                sliced_list._state_to_slice[state_idx] = (level_val, offset)
        
        return sliced_list
    
    def __getitem__(self, state_idx: int) -> T:
        """Get value for a state index."""
        level_val, offset = self._state_to_slice[state_idx]
        slice_data = self._slices[level_val]
        if slice_data is None:
            raise ValueError(f"Slice for level {level_val} has been archived")
        return slice_data[offset]
    
    def __setitem__(self, state_idx: int, value: T) -> None:
        """Set value for a state index."""
        level_val, offset = self._state_to_slice[state_idx]
        slice_data = self._slices[level_val]
        if slice_data is None:
            raise ValueError(f"Slice for level {level_val} has been archived")
        slice_data[offset] = value
    
    def __len__(self) -> int:
        """Return total number of states."""
        return len(self._state_to_slice)
    
    def archive_slice(
        self,
        level_value: int,
        filepath: Path,
        clear_memory: bool = True
    ) -> None:
        """Archive a slice to disk and optionally clear from memory.
        
        Args:
            level_value: The level value to archive
            filepath: Path to append the slice data to
            clear_memory: If True, set slice to None to free memory
        """
        if level_value not in self._slices:
            raise ValueError(f"Level {level_value} not found in slices")
        
        slice_data = self._slices[level_value]
        if slice_data is None:
            return  # Already archived
        
        # Append to file using pickle protocol 5 for zero-copy of buffers
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'ab') as f:
            pickle.dump({'level_value': level_value, 'data': slice_data}, f, protocol=5)
        
        # Clear from memory if requested
        if clear_memory:
            self._slices[level_value] = None
    
    def get_slice(self, level_value: int) -> Optional[List[T]]:
        """Get the slice for a level value (may be None if archived)."""
        return self._slices.get(level_value)
    
    def has_level(self, level_value: int) -> bool:
        """Check if a level exists (even if archived)."""
        return level_value in self._slices
    
    def is_archived(self, level_value: int) -> bool:
        """Check if a level has been archived."""
        return self._slices.get(level_value) is None


def detect_archivable_levels(
    current_level_value: int,
    max_successor_levels: Dict[int, int],
    quiet: bool = False
) -> List[int]:
    """Detect which levels can be archived based on successor dependencies.
    
    A level k can be archived when max_successor_levels[l] < k for all l < current_level_value.
    This means no future computations will need to read from level k.
    
    Args:
        current_level_value: The level value just completed
        max_successor_levels: Dict mapping level value to max successor level value
        quiet: If True, suppress debug output
        
    Returns:
        List of level values that can be safely archived
    """
    # Get all level values that are < current level value (not yet processed)
    # Note: backward induction processes in descending order, so smaller values are future
    future_levels = [l for l in max_successor_levels.keys() if l < current_level_value]
    
    if not future_levels:
        return []
    
    # Find the maximum level value among all future successors
    max_future_successor = max(max_successor_levels[l] for l in future_levels)
    
    # All levels > max_future_successor can be archived
    return sorted([l for l in max_successor_levels.keys() if l > max_future_successor], reverse=True)


def archive_value_slices(
    values: List[Any],
    states: List[State],
    level_fct: Callable[[State], int],
    archivable_levels: List[int],
    filepath: Path,
    return_values: bool = True,
    quiet: bool = False,
    archive_dir: Optional[Path] = None
) -> None:
    """Archive value function slices to disk and optionally clear from memory.
    
    This function archives slices of the value function (Vh_values, Vr_values, Vhe_values, etc.)
    organized by level value. Used for memory-efficient backward induction
    with disk slicing.
    
    Memory optimization: Uses pickle protocol 5 and incremental writing (one state at a time)
    to avoid creating large intermediate copies during serialization. This prevents
    memory spikes when archiving large value function slices.
    
    Args:
        values: Value function array indexed by state. Can be List[List[Dict[goal, float]]]
                for vh_values/vr_values or List[Dict[goal, float]] for vhe_values.
        states: List of all states
        level_fct: Function mapping state to its level value
        archivable_levels: List of level values to archive
        filepath: Path to append the archived slices
        return_values: If False, clear archived slices from memory after saving
        quiet: If True, suppress progress output
    """
    if not archivable_levels:
        return
    
    # Use archive_dir if provided, otherwise use filepath's parent
    if archive_dir is not None:
        filepath = archive_dir / filepath.name
    
    # Group state indices by level value
    level_to_states: Dict[int, List[int]] = defaultdict(list)
    for state_idx, state in enumerate(states):
        level_val = level_fct(state)
        if level_val in archivable_levels:
            level_to_states[level_val].append(state_idx)
    
    # Archive each level
    total_states = 0
    for level_val in archivable_levels:
        state_indices = level_to_states.get(level_val, [])
        if not state_indices:
            continue
        
        # Append to file using protocol 5 for zero-copy of buffers
        # Write incrementally to avoid creating large intermediate dict
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'ab') as f:
            # Write header with metadata
            pickle.dump({
                'level_value': level_val,
                'state_indices': state_indices,
                'num_states': len(state_indices)
            }, f, protocol=5)
            
            # Write data incrementally (one state at a time) to avoid copying entire slice
            for state_idx in state_indices:
                pickle.dump(values[state_idx], f, protocol=5)
        
        total_states += len(state_indices)
        
        # Clear from memory if not returning values
        if not return_values:
            for state_idx in state_indices:
                # Replace with empty structure to free memory
                # Handle both List[List[Dict]] and List[Dict] structures
                if isinstance(values[state_idx], list):
                    values[state_idx] = [{} for _ in range(len(values[state_idx]))]
                else:
                    values[state_idx] = {}
    
    if not quiet:
        clear_msg = " (cleared from memory)" if not return_values else ""
        print(f"    Archived {total_states} states from {len(archivable_levels)} level(s) to {filepath.name}{clear_msg}")


def load_archived_value_slices(filepath: Path):
    """Generator that yields (level_value, state_indices, data) from archived file.
    
    Supports both old format (monolithic 'data' field) and new format (incremental writes).
    
    Args:
        filepath: Path to the archived value slices file
        
    Yields:
        Tuple of (level_value, state_indices, data) where data is a list of values per state
    """
    if not filepath.exists():
        return
    
    with open(filepath, 'rb') as f:
        while True:
            try:
                # Load metadata header
                header = pickle.load(f)
                level_value = header['level_value']
                state_indices = header['state_indices']
                
                # Check if old format (monolithic) or new format (incremental)
                if 'data' in header:
                    # Old format: all data in header
                    yield level_value, state_indices, header['data']
                else:
                    # New format: read num_states individual pickles
                    num_states = header['num_states']
                    data = []
                    for _ in range(num_states):
                        data.append(pickle.load(f))
                    yield level_value, state_indices, data
            except EOFError:
                break

