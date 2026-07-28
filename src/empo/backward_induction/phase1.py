"""
Phase 1: Human Policy Prior Computation via Backward Induction.

This module implements backward induction on the state DAG to compute
goal-conditioned Boltzmann policies for human agents.

Main function:
    compute_human_policy_prior: Compute tabular human policy prior via backward induction.

The algorithm works by:
1. Building the DAG of reachable states and transitions
2. Processing states in reverse topological order (from terminal to initial)

Key features:
- Supports parallel computation for large state spaces
- Returns both the policy (prior) and optionally value functions
- Automatically stores attainment cache on world_model for reuse in Phase 2

Parameters:
    beta_h: Human inverse temperature (β). Higher = more deterministic, inf = argmax.
    gamma_h: Human discount factor (γ).
"""

import gc
import math
import numpy as np
import numpy.typing as npt
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any, Union, overload, Literal, Set

import cloudpickle
from tqdm import tqdm

from empo.util.memory_monitor import MemoryMonitor, deep_sizeof, get_process_memory_mb
from empo.backward_induction.dag_slicing import (
    DiskBasedDAG, 
    estimate_dag_memory,
    convert_transitions_to_float16
)
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.world_model import WorldModel
from empo.backward_induction.shared_dag import (
    init_shared_dag, get_shared_dag, attach_shared_dag, cleanup_shared_dag
)
from empo.backward_induction.shared_attainment_cache import (
    cleanup_shared_attainment_cache
)

from .helpers import (
    State, TransitionData, SliceCache, SliceId, SlicedAttainmentCache,
    DefaultBelievedOthersPolicy,
    compute_dependency_levels_general,
    compute_dependency_levels_fast,
    split_into_batches,
    make_slice_id,
    SlicedList,
    detect_archivable_levels,
    archive_value_slices,
    VhValues,
    VhValuesSmall,
)

# Type aliases
HumanPolicyDict = Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]]  # state -> agent -> goal -> probs

DEBUG = False  # Set to True for verbose debugging output
PROFILE_PARALLEL = os.environ.get('PROFILE_PARALLEL', '').lower() in ('1', 'true', 'yes')

# Module-level globals for shared memory in forked processes
# These are set before spawning workers and inherited copy-on-write
_shared_states: Optional[List[State]] = None
_shared_transitions: Optional[List[List[TransitionData]]] = None
_shared_Vh_values: Optional[VhValues] = None
_shared_sliced_cache: Optional[SlicedAttainmentCache] = None  # Sliced cache for goal attainment arrays
_shared_num_action_profiles: int = 0  # Needed to create slice caches in workers
_shared_params: Optional[Tuple[List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], List[int], List[List[int]], float, float, bool, float]] = None
_shared_believed_others_policy_pickle: Optional[bytes] = None  # cloudpickle'd believed_others_policy function
_shared_disk_dag: Optional['DiskBasedDAG'] = None  # For parallel disk slicing
_shared_world_model: Optional['WorldModel'] = None  # For duration-aware discounting in workers


def _hpp_process_single_state(
    state_index: int,
    state: State,
    states: List[State],
    transitions: Union[List[List[TransitionData]], List[TransitionData]],  # Can be full list or single state's transitions
    Vh_values: Union[VhValues, VhValuesSmall],
    human_agent_indices: List[int],
    possible_goal_generator: PossibleGoalGenerator,
    num_actions_per_agent: List[int],
    action_powers: npt.NDArray[np.int64],
    believed_others_policy: Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]],
    robot_agent_indices: List[int],
    robot_action_profiles: List[List[int]],
    beta_h: float,
    gamma_h: float,
    slice_cache: Optional[SliceCache] = None,
    use_indexed: bool = False,
    vres0: Union[Dict, npt.NDArray] = None,
    pres0: Union[Dict, npt.NDArray] = None,
    optimistic: bool = False,
    rho_h: float = 0.0,
    world_model: Optional['WorldModel'] = None,
) -> Tuple[Dict[int, Dict[PossibleGoal, float]], Optional[Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]]]:
    """Process a single state, returning (v_results, p_results).
    
    Unified implementation for sequential, parallel batch, and inline fallback.
    Handles both terminal and non-terminal states correctly.
    
    Args:
        state_index: Index of the state in the states list
        state: The state to process
        transitions: Either full transitions list (indexed by state_index) OR 
                    the transitions for just this state (when using disk slicing)
        Vh_values: Value function (reads from successors, may write to state_index)
        human_agent_indices: List of agent indices to compute for
        possible_goal_generator: Generator for possible goals
        num_actions_per_agent: Number of actions for each agent
        action_powers: Precomputed powers for action profile indexing
        believed_others_policy: Function for beliefs about other agents
        beta_h: Inverse temperature (can be float('inf') for argmax)
        gamma_h: Discount factor
        slice_cache: Optional slice cache to WRITE to (worker's local cache for this batch).
            Structure: Dict[state_index, List[Dict[goal, array]]]
        rho_h: Continuous-time discount rate = -ln(gamma_h). When > 0, per-transition
            duration-aware discounting is used via world_model.transition_durations().
        world_model: WorldModel instance for querying transition_durations(). Required
            when rho_h > 0.
    
    Returns:
        Tuple of:
        - v_results: Dict[agent_index, Dict[goal, float]] - V-values for this state
        - p_results: Dict[agent_index, Dict[goal, ndarray]] - policies (None for terminal states)
    """
    if slice_cache is not None and state_index in slice_cache:
        this_state_cache = slice_cache[state_index]
    else:
        this_state_cache = None

    v_results: Dict[int, Any] = {}
    p_results: Optional[Dict[int, Any]] = None
    
    # Support both full transitions list and single-state transitions
    if transitions and isinstance(transitions[0], list):
        # Full list of transitions - index into it
        state_transitions = transitions[state_index]
    else:
        # Single state's transitions passed directly (or empty list for terminal)
        state_transitions = transitions
    
    is_terminal = not state_transitions
    
    if is_terminal:
        # Terminal state: V_h^m = 0 for all goals
        if DEBUG:
            print(f"  Terminal state {state_index}")
        return v_results, None
    
    # Non-terminal state: compute both V-values and policies
    p_results = {}
    
    # Cache duration arrays per action_profile_index to avoid repeated world_model queries.
    # Durations depend on (state, action_profile, transitions) which are uniquely identified
    # by action_profile_index for a given state. The cache is reused across goal iterations.
    _duration_cache: Dict[int, npt.NDArray] = {}
    
    for agent_index in human_agent_indices:
        agent_num_actions = num_actions_per_agent[agent_index]
        vres = v_results[agent_index] = vres0.copy() if use_indexed else {}
        pres = p_results[agent_index] = pres0.copy() if use_indexed else {}
        
        for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
            key = possible_goal.index if use_indexed else possible_goal
            if possible_goal.is_achieved(state):
                # Goal achieved already in this state: remain still, V=0
                # Sparse storage: don't store zero values
                # vres[possible_goal] = np.float16(0)  # Omitted for sparse storage
                ps = np.zeros(agent_num_actions)
                ps[0] = 1.0  # Assume action 0 is 'stay still'
                pres[key] = ps
                if DEBUG:
                    print(f"    Goal achieved in state, agent {agent_index}: V = 0, remain still policy")
            else:
                # Compute Q values as expected future V values
                q = np.zeros(agent_num_actions)
                for action in range(agent_num_actions):
                    v_accum = 0.0
                    for action_profile_prob, action_profile in believed_others_policy(state, agent_index, action):
                        anticipated_expectation = -math.inf if optimistic else math.inf
                        action_profile[agent_index] = action
                        for robot_action_profile in robot_action_profiles:
                            action_profile[robot_agent_indices] = robot_action_profile
                            action_profile_index = (action_profile @ action_powers).item()
                            _, next_state_probabilities, next_state_indices = state_transitions[action_profile_index]

                            # Compute attainment values
                            attainment_values_array = np.fromiter(
                                (possible_goal.is_achieved(states[next_state_index]) 
                                    for next_state_index in next_state_indices),
                                dtype=np.int8,
                                count=len(next_state_indices)
                            )
                            
                            # Store in slice_cache if available
                            if this_state_cache is not None:
                                this_state_cache[action_profile_index][possible_goal] = attainment_values_array

                            # Get V values from successors (use .get for parallel safety)
                            if use_indexed:
                                # VhValuesSmall: indexed by goal.index
                                v_values_array = np.fromiter(
                                    (Vh_values[next_state_index][agent_index][key]
                                     for next_state_index in next_state_indices),
                                    dtype=np.float16,
                                    count=len(next_state_indices)
                                )
                            else:
                                # VhValues: dict keyed by goal
                                v_values_array = np.fromiter(
                                    (Vh_values[next_state_index][agent_index].get(possible_goal, 0)
                                     for next_state_index in next_state_indices),
                                    dtype=np.float16,
                                    count=len(next_state_indices)
                                )
                            # Use np.where to avoid intermediate array allocation
                            # NumPy automatically promotes float16 to float64 during computation
                            if rho_h > 0.0:
                                # Duration-aware discounting: e^{-rho_h * D(s, a, s')} per transition
                                # Only discount non-achieved successors (achieved stay at 1.0)
                                if world_model is None:
                                    raise ValueError("world_model is required for duration-aware discounting (rho_h > 0)")
                                # Use cached durations if available, otherwise compute and cache
                                if action_profile_index in _duration_cache:
                                    durations_arr = _duration_cache[action_profile_index]
                                else:
                                    transitions_list = [(float(p), states[i]) for p, i in zip(next_state_probabilities, next_state_indices)]
                                    durations_arr = np.array(world_model.transition_durations(state, action_profile.tolist(), transitions_list))
                                    if len(durations_arr) != len(next_state_indices):
                                        raise ValueError(
                                            f"transition_durations() returned {len(durations_arr)} durations but expected "
                                            f"{len(next_state_indices)} (state_index={state_index}, action_profile_index={action_profile_index})")
                                    _duration_cache[action_profile_index] = durations_arr
                                discount_factors = np.exp(-rho_h * durations_arr)
                                successor_values = np.where(attainment_values_array, 1.0, discount_factors * v_values_array)
                            else:
                                successor_values = np.where(attainment_values_array, 1.0, v_values_array)
                            expectation = np.dot(
                                next_state_probabilities,
                                successor_values
                            )
                            if ((expectation > anticipated_expectation) if optimistic 
                                else (expectation < anticipated_expectation)):
                                anticipated_expectation = expectation
                        v_accum += action_profile_prob * anticipated_expectation
                    q[action] = v_accum
                
                # Boltzmann policy (numerically stable softmax)
                if beta_h == math.inf:
                    # Infinite beta: deterministic argmax policy
                    max_q = np.max(q)
                    p = np.zeros_like(q)
                    max_indices = np.where(q == max_q)[0]
                    p[max_indices] = 1.0 / len(max_indices)  # Uniform over max actions
                else:
                    scaled_q = beta_h * (q - np.max(q))  # Subtract max for numerical stability
                    p = np.exp(scaled_q)
                    p /= np.sum(p)
                
                v_value = float(np.sum(p * q))
                # Sparse storage: only store non-zero values (convert to float16 only when storing)
                if v_value != 0.0:
                    vres[key] = np.float16(v_value)
                pres[key] = p
                
                if DEBUG:
                    print(f"    Agent {agent_index}, goal {possible_goal}: V = {v_results[agent_index].get(possible_goal, 0):.4f}")
    
    return v_results, p_results


def _hpp_compute_sequential(
    states: List[State], 
    Vh_values: Union[VhValues, VhValuesSmall],  # result is inserted into this!
    system2_policies: HumanPolicyDict,  # result is inserted into this! 
    transitions: Optional[List[List[Tuple[Tuple[int, ...], List[float], List[int]]]]],
    human_agent_indices: List[int], 
    possible_goal_generator: PossibleGoalGenerator,
    num_agents: int, 
    num_actions_per_agent: List[int], 
    action_powers: npt.NDArray[np.int64],
    believed_others_policy: Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]], 
    robot_agent_indices: List[int],
    robot_action_profiles: List[List[int]],
    beta_h: float, 
    gamma_h: float,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    memory_monitor: Optional[MemoryMonitor] = None,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    disk_dag: Optional[DiskBasedDAG] = None,
    level_fct: Optional[Callable[[State], int]] = None,
    archive_dir: Optional[str] = None,
    return_Vh: bool = True,
    quiet: bool = False,
    memory_profile: bool = False,
    optimistic: bool = False,
    rho_h: float = 0.0,
    world_model: Optional['WorldModel'] = None,
) -> None:
    """Sequential backward induction algorithm.
    
    Processes states in reverse topological order using the unified
    _process_single_state helper.
    
    Supports disk-based slicing: if disk_dag is provided, loads transitions
    from disk timestep-by-timestep instead of keeping all in memory.
    """
    total_states = len(states)
    
    # Memory tracking interval
    memory_report_interval = max(1, total_states // 10)  # Report ~10 times during computation
    
    # Determine if using indexed goals and initialize templates
    use_indexed = False # hasattr(possible_goal_generator, 'indexed') and possible_goal_generator.indexed
    if use_indexed:
        n_goals = possible_goal_generator.n_goals
        vres0 = np.zeros(n_goals, dtype=np.float16)
        pres0 = np.zeros((n_goals, max(num_actions_per_agent)), dtype=np.float16)
        print(f"Using indexed goals: VhValuesSmall with numpy arrays (n_goals={n_goals})")
    else:
        vres0 = {}
        pres0 = {}
    
    # In sequential mode, we use a single slice containing all states
    # Create slice cache for all states, but only allocate entries for non-terminal states
    slice_cache: Optional[SliceCache] = None
    if sliced_cache is not None:
        # Only create cache entries for non-terminal states (those with transitions)
        # If using disk_dag, we can't determine terminal states yet, so create for all states
        if transitions is not None:
            non_terminal_indices = [i for i in range(len(states)) if transitions[i]]
            slice_cache = sliced_cache.create_slice_cache(non_terminal_indices)
        else:
            # disk_dag mode or transitions not yet loaded - create for all states
            all_state_indices = list(range(len(states)))
            slice_cache = sliced_cache.create_slice_cache(all_state_indices)
    
    # Compute max_successor_levels for archival if level_fct and archive_dir provided
    max_successor_levels: Optional[Dict[int, int]] = None
    archived_levels: Set[int] = set()  # Track already-archived levels
    if level_fct is not None and archive_dir is not None:
        from .helpers import compute_dependency_levels_fast, detect_archivable_levels, archive_value_slices
        from pathlib import Path
    
    # Disk-based slicing: process by timesteps
    if disk_dag is not None:
        if level_fct is None:
            raise ValueError("disk_dag requires level_fct")
        
        # Compute max successor levels for archival if archive_dir provided
        if archive_dir is not None:
            # Get successors from disk slices to compute max_successor_levels
            successors_map: Dict[int, List[int]] = {}
            for timestep in range(disk_dag.max_timestep, -1, -1):
                dag_slice = disk_dag.load_slice(timestep)
                for state_idx in dag_slice.state_indices:
                    state_transitions = dag_slice.get_transitions(state_idx)
                    # Extract unique successor indices from all transitions
                    succ_set = set()
                    for action_profile, probs, succ_states in state_transitions:
                        succ_set.update(succ_states)
                    successors_map[state_idx] = list(succ_set)
                disk_dag.unload_slice(timestep)
            
            # Convert to list format for compute_dependency_levels_fast
            successors = [successors_map.get(i, []) for i in range(len(states))]
            _, max_successor_levels, _ = compute_dependency_levels_fast(states, level_fct, successors)
        
        # Track progress accurately
        states_processed_count = 0
        
        # Process timesteps in reverse order
        for timestep in range(disk_dag.max_timestep, -1, -1):
            # Load transitions for this timestep
            dag_slice = disk_dag.load_slice(timestep)
            
            # Create cache slice for this timestep (will be saved to disk)
            timestep_cache = disk_dag.create_cache_slice_for_states(dag_slice.state_indices)
            
            # Process all states in this timestep
            for global_state_idx in dag_slice.state_indices:
                states_processed_count += 1
                if memory_monitor is not None:
                    memory_monitor.check(states_processed_count)
                if progress_callback is not None:
                    progress_callback(states_processed_count, total_states)
                
                state = states[global_state_idx]
                
                # Get transitions from disk slice
                state_transitions = dag_slice.get_transitions(global_state_idx)
                
                # Process state with disk-based cache
                v_results, p_results = _hpp_process_single_state(
                    global_state_idx, state, states, state_transitions, Vh_values,
                    human_agent_indices, possible_goal_generator,
                    num_actions_per_agent, action_powers, believed_others_policy,
                    robot_agent_indices, robot_action_profiles,
                    beta_h, gamma_h,
                    slice_cache=timestep_cache,
                    use_indexed=use_indexed,
                    vres0=vres0,
                    pres0=pres0,
                    optimistic=optimistic,
                    rho_h=rho_h,
                    world_model=world_model,
                )
                
                # Store results
                for agent_index, agent_v in v_results.items():
                    Vh_values[global_state_idx][agent_index] = agent_v
                if p_results is not None:
                    system2_policies[state] = p_results
            
            # Save cache slice to disk and free memory
            disk_dag.save_cache_slice(timestep, timestep_cache)
            del timestep_cache
            
            # Unload DAG slice to free memory
            disk_dag.unload_slice(timestep)
            
            # Archive completed timestep if archive_dir is set
            if archive_dir is not None and max_successor_levels is not None:
                archivable = detect_archivable_levels(timestep, max_successor_levels, quiet=False)
                # Only archive NEW levels (not already archived)
                new_archivable = [lvl for lvl in archivable if lvl not in archived_levels]
                if new_archivable:
                    archive_value_slices(
                        Vh_values, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vh_values.pkl",
                        return_values=return_Vh,
                        quiet=False
                    )
                    archived_levels.update(new_archivable)
                # Only archive levels not yet archived
                new_archivable = [l for l in archivable if l not in archived_levels]
                if new_archivable:
                    archive_filepath = Path(disk_dag.cache_dir) / "vh_values.pkl"
                    archive_value_slices(
                        Vh_values, states, level_fct, new_archivable,
                        archive_filepath, return_values=return_Vh, quiet=False,
                        archive_dir=Path(archive_dir)
                    )
                    archived_levels.update(new_archivable)
        
        # Store disk_dag reference for Phase 2 cache reuse
        if sliced_cache is not None:
            sliced_cache._disk_dag = disk_dag  # type: ignore
        
        return  # Done with disk-based processing
    
    # Standard in-memory processing
    if transitions is None:
        raise ValueError("transitions must be provided when disk_dag is None")
    
    # Compute max_successor_levels for archival if level_fct and archive_dir provided
    archived_levels: Set[int] = set()  # Track already-archived levels
    previous_level: Optional[int] = None  # Track level transitions
    if level_fct is not None and archive_dir is not None:
        # Build successors list from transitions
        successors = []
        for state_transitions in transitions:
            succ_set = set()
            for action_profile, probs, succ_indices in state_transitions:
                succ_set.update(succ_indices)
            successors.append(list(succ_set))
        _, max_successor_levels, level_values_list = compute_dependency_levels_fast(states, level_fct, successors)
        
        # Pre-compute state levels to avoid calling level_fct for every state
        state_levels = [level_fct(s) for s in states]
    
    # loop over the nodes in reverse topological order:
    for state_index in range(len(states)-1, -1, -1):
        # Check for level transition BEFORE processing state (for archival)
        if archive_dir is not None and level_fct is not None and max_successor_levels is not None:
            current_state_level = state_levels[state_index]
            if previous_level is not None and current_state_level != previous_level:
                # We just completed processing previous_level, check what can be archived
                archivable = detect_archivable_levels(current_state_level, max_successor_levels, quiet=False)
                # Only archive NEW levels (not already archived)
                new_archivable = [lvl for lvl in archivable if lvl not in archived_levels]
                if new_archivable:
                    archive_value_slices(
                        Vh_values, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vh_values.pkl",
                        return_values=return_Vh,
                        quiet=False
                    )
                    archived_levels.update(new_archivable)
            previous_level = current_state_level
        
        # Check memory periodically (using step = states processed)
        states_processed = total_states - state_index
        if memory_monitor is not None:
            memory_monitor.check(states_processed)
        # Update progress bar
        if progress_callback is not None:
            progress_callback(states_processed, total_states)
        
        # Periodic memory reporting
        if not quiet and states_processed > 0 and states_processed % memory_report_interval == 0:
            print(f"\n[Memory @ {states_processed}/{total_states} states] RSS: {get_process_memory_mb():.1f} MB")
            if memory_profile:
                # Detailed profiling with deep_sizeof (adds O(total_size) overhead)
                vh_mb = deep_sizeof(Vh_values) / (1024**2)
                pol_mb = deep_sizeof(system2_policies) / (1024**2)
                cache_mb = deep_sizeof(slice_cache) / (1024**2) if slice_cache is not None else 0.0
                print(f"  Vh_values: {vh_mb:.1f} MB, policies: {pol_mb:.1f} MB, cache: {cache_mb:.1f} MB")
        
        if DEBUG:
            print(f"Processing state {state_index}")
        
        state = states[state_index]
        
        # Use unified helper
        v_results, p_results = _hpp_process_single_state(
            state_index, state, states, transitions, Vh_values,
            human_agent_indices, possible_goal_generator,
            num_actions_per_agent, action_powers, believed_others_policy,
            robot_agent_indices, robot_action_profiles,
            beta_h, gamma_h,
            slice_cache=slice_cache,
            use_indexed=use_indexed,
            vres0=vres0,
            pres0=pres0,
            optimistic=optimistic,
            rho_h=rho_h,
            world_model=world_model,
        )
        
        # Store results
        for agent_index, agent_v in v_results.items():
            Vh_values[state_index][agent_index] = agent_v
        
        if p_results is not None:
            system2_policies[state] = p_results
    
    # Final archival check: archive any remaining levels after loop completes
    if archive_dir is not None and level_fct is not None and max_successor_levels is not None:
        # Check if there are any levels we haven't archived yet
        # At this point, all states have been processed, so all levels should be archivable
        all_levels = sorted(set(level_fct(s) for s in states))
        remaining_levels = [lvl for lvl in all_levels if lvl not in archived_levels]
        if remaining_levels:
            archive_value_slices(
                Vh_values, states, level_fct, remaining_levels,
                filepath=Path(archive_dir) / "vh_values.pkl",
                return_values=return_Vh,
                quiet=False
            )
            archived_levels.update(remaining_levels)
    
    # Store the slice in the sliced cache
    if sliced_cache is not None and slice_cache is not None:
        slice_id = make_slice_id(list(range(len(states))))
        sliced_cache.store_slice(slice_id, slice_cache)


def _hpp_init_shared_data(
    states: List[State], 
    transitions: List[List[TransitionData]], 
    Vh_values: VhValues, 
    params: Tuple[List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float, bool],
    believed_others_policy_pickle: Optional[bytes] = None,
    use_shared_memory: bool = False,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    num_action_profiles: int = 0,
    world_model: Optional['WorldModel'] = None,
) -> None:
    """Initialize shared data for worker processes.
    
    Args:
        states: List of states (will be stored in shared memory if use_shared_memory=True)
        transitions: List of transitions (will be stored in shared memory if use_shared_memory=True)
        Vh_values: Value function (always passed via globals, updated by workers)
        params: Parameters tuple
        believed_others_policy_pickle: Pickled policy function
        use_shared_memory: If True, store states and transitions in shared memory
        sliced_cache: Optional SlicedAttainmentCache for goal attainment arrays
        num_action_profiles: Number of action profiles (needed to create slice caches)
        world_model: WorldModel for duration-aware discounting (needed when rho_h > 0)
    """
    global _shared_states, _shared_transitions, _shared_Vh_values, _shared_params
    global _shared_believed_others_policy_pickle, _shared_sliced_cache, _shared_num_action_profiles
    global _shared_world_model
    
    if use_shared_memory:
        # Store DAG in shared memory to avoid copy-on-write
        init_shared_dag(states, transitions)
        _shared_states = None  # Will be loaded from shared memory in workers
        _shared_transitions = None
    else:
        _shared_states = states
        _shared_transitions = transitions
    
    _shared_Vh_values = Vh_values
    _shared_params = params
    _shared_believed_others_policy_pickle = believed_others_policy_pickle
    _shared_sliced_cache = sliced_cache
    _shared_num_action_profiles = num_action_profiles
    _shared_world_model = world_model


def _hpp_process_state_batch(
    state_indices: List[int]
) -> Tuple[Dict[int, Dict[int, Dict[PossibleGoal, float]]], 
           Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]], 
           SliceId,  # slice_id for storing the cache
           SliceCache,  # slice cache for this batch
           float]:
    """Process a batch of states that can be computed in parallel.
    
    Uses module-level shared data (inherited via fork) to avoid copying.
    Returns V-values, policies, slice_id, slice_cache, plus timing.
    """
    batch_start = time.perf_counter()
    
    # Access shared data - these are guaranteed to be set when called from parallel context
    assert _shared_Vh_values is not None
    assert _shared_params is not None
    
    # Try to get states/transitions from shared memory first, fall back to globals
    shared_dag = get_shared_dag()
    if shared_dag is None:
        # Try to attach to shared memory (first call in this worker)
        shared_dag = attach_shared_dag()
    
    if shared_dag is not None:
        states = shared_dag.get_states()
        transitions = shared_dag.get_transitions()
    else:
        # Fall back to module-level globals
        assert _shared_states is not None
        assert _shared_transitions is not None
        states = _shared_states
        transitions = _shared_transitions
    
    assert transitions is not None
    
    Vh_values = _shared_Vh_values
    (human_agent_indices, possible_goal_generator, num_agents, num_actions_per_agent, 
     action_powers, robot_agent_indices, robot_action_profiles, beta_h, gamma_h, optimistic, rho_h) = _shared_params
    
    v_results: Dict[int, Dict[int, Dict[PossibleGoal, float]]] = {}
    p_results: Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]] = {}
    
    # Create slice cache for this batch
    # Structure: Dict[state_index, List[Dict[goal, array]]]
    slice_id = make_slice_id(state_indices)
    slice_cache: SliceCache = {
        state_idx: [{} for _ in range(_shared_num_action_profiles)]
        for state_idx in state_indices
    }
    
    # Deserialize believed_others_policy if custom one was provided via cloudpickle
    if _shared_believed_others_policy_pickle is not None:
        believed_others_policy = cloudpickle.loads(_shared_believed_others_policy_pickle)
    else:
        # Create precomputed default believed others policy
        believed_others_policy = DefaultBelievedOthersPolicy(
            num_agents, num_actions_per_agent, human_agent_indices, robot_agent_indices)
    
    for state_index in state_indices:
        state = states[state_index]
        
        # Use unified helper with:
        # - slice_cache for writing (this worker's cache for current batch)
        # - _shared_sliced_cache for reading (populated by previous levels)
        state_v_results, state_p_results = _hpp_process_single_state(
            state_index, state, states, transitions, Vh_values,
            human_agent_indices, possible_goal_generator,
            num_actions_per_agent, action_powers, believed_others_policy,
            robot_agent_indices, robot_action_profiles,
            beta_h, gamma_h,
            slice_cache=slice_cache,
            optimistic=optimistic,
            rho_h=rho_h,
            world_model=_shared_world_model,
        )
        
        v_results[state_index] = state_v_results
        if state_p_results is not None:
            p_results[state] = state_p_results
    
    batch_time = time.perf_counter() - batch_start
    return v_results, p_results, slice_id, slice_cache, batch_time


def _hpp_process_timestep_batch_disk(
    timestep: int,
    state_indices: List[int]
) -> Tuple[Dict[int, Dict[int, Dict[PossibleGoal, float]]], 
           Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]], 
           int,  # timestep
           float]:  # timing
    """
    Process a batch of states from a timestep using disk-based slicing.
    
    Worker loads DAG slice and cache slice from disk, processes states,
    saves cache slice back to disk, all independently of master process.
    
    Args:
        timestep: The timestep number
        state_indices: List of state indices in this timestep (sorted in Kahn order)
    
    Returns:
        Tuple of (v_results, p_results, timestep, batch_time)
    """
    batch_start = time.perf_counter()
    
    # Access shared data
    assert _shared_disk_dag is not None, "disk_dag must be initialized for parallel disk slicing"
    assert _shared_Vh_values is not None
    assert _shared_params is not None
    assert _shared_states is not None
    
    disk_dag = _shared_disk_dag
    states = _shared_states
    Vh_values = _shared_Vh_values
    (human_agent_indices, possible_goal_generator, num_agents, num_actions_per_agent, 
     action_powers, robot_agent_indices, robot_action_profiles, beta_h, gamma_h, optimistic, rho_h) = _shared_params
    
    # Deserialize believed_others_policy if custom one was provided via cloudpickle
    if _shared_believed_others_policy_pickle is not None:
        believed_others_policy = cloudpickle.loads(_shared_believed_others_policy_pickle)
    else:
        # Create precomputed default believed others policy
        believed_others_policy = DefaultBelievedOthersPolicy(
            num_agents, num_actions_per_agent, human_agent_indices, robot_agent_indices)
    
    # Load DAG slice from disk (transitions for this timestep)
    dag_slice = disk_dag.load_slice(timestep)
    
    # Create empty cache structure for this timestep
    timestep_cache = disk_dag.create_cache_slice_for_states(state_indices)
    
    # Process all states in this timestep
    v_results: Dict[int, Dict[int, Dict[PossibleGoal, float]]] = {}
    p_results: Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]] = {}
    
    for global_state_idx in state_indices:
        state = states[global_state_idx]
        
        # Get transitions from disk slice
        state_transitions = dag_slice.get_transitions(global_state_idx)
        
        # Process state
        state_v_results, state_p_results = _hpp_process_single_state(
            global_state_idx, state, states, state_transitions, Vh_values,
            human_agent_indices, possible_goal_generator,
            num_actions_per_agent, action_powers, believed_others_policy,
            robot_agent_indices, robot_action_profiles,
            beta_h, gamma_h,
            slice_cache=timestep_cache,
            optimistic=optimistic,
            rho_h=rho_h,
            world_model=_shared_world_model,
        )
        
        v_results[global_state_idx] = state_v_results
        if state_p_results is not None:
            p_results[state] = state_p_results
    
    # Save cache slice to disk (worker writes directly)
    disk_dag.save_cache_slice(timestep, timestep_cache)
    
    # Unload DAG slice to free memory
    disk_dag.unload_slice(timestep)
    
    batch_time = time.perf_counter() - batch_start
    return v_results, p_results, timestep, batch_time


@overload
def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: Optional[PossibleGoalGenerator] = None, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]]] = None, 
    *,
    beta_h: float = 10.0, 
    gamma_h: Optional[float] = None, 
    rho_h: Optional[float] = None,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: Literal[False] = False,
    return_attainment_cache: Literal[False] = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    optimistic: bool = False,
) -> TabularHumanPolicyPrior: ...


@overload
def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: Optional[PossibleGoalGenerator] = None, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]]] = None, 
    *, 
    beta_h: float = 10.0, 
    gamma_h: Optional[float] = None, 
    rho_h: Optional[float] = None,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: Literal[True],
    return_attainment_cache: Literal[False] = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    optimistic: bool = False,
) -> Tuple[TabularHumanPolicyPrior, Dict[State, Dict[int, Dict[PossibleGoal, float]]]]: ...


@overload
def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: Optional[PossibleGoalGenerator] = None, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]]] = None, 
    *, 
    beta_h: float = 10.0, 
    gamma_h: Optional[float] = None, 
    rho_h: Optional[float] = None,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: Literal[False] = False,
    return_attainment_cache: Literal[True],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    optimistic: bool = False,
) -> Tuple[TabularHumanPolicyPrior, SlicedAttainmentCache]: ...


@overload
def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: Optional[PossibleGoalGenerator] = None, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]]] = None, 
    *, 
    beta_h: float = 10.0, 
    gamma_h: Optional[float] = None, 
    rho_h: Optional[float] = None,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: Literal[True],
    return_attainment_cache: Literal[True],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    optimistic: bool = False,
) -> Tuple[TabularHumanPolicyPrior, Dict[State, Dict[int, Dict[PossibleGoal, float]]], SlicedAttainmentCache]: ...


def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: Optional[PossibleGoalGenerator] = None, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]]] = None, 
    *,
    beta_h: float = 10.0, 
    gamma_h: Optional[float] = None, 
    rho_h: Optional[float] = None,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: bool = False,
    return_attainment_cache: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    use_disk_slicing: bool = False,
    disk_cache_dir: Optional[str] = None,
    use_float16: bool = True,
    use_compression: bool = False,
    archive_dir: Optional[str] = None,
    use_attainment_cache: bool = True,
    optimistic: bool = False,
) -> Union[TabularHumanPolicyPrior, 
           Tuple[TabularHumanPolicyPrior, Dict[State, Dict[int, Dict[PossibleGoal, float]]]],
           Tuple[TabularHumanPolicyPrior, SlicedAttainmentCache],
           Tuple[TabularHumanPolicyPrior, Dict[State, Dict[int, Dict[PossibleGoal, float]]], SlicedAttainmentCache]]:
    """
    Compute human policy prior via backward induction on the state DAG.
    
    This function builds the complete state DAG of the world model and computes
    goal-conditioned Boltzmann policies for all human agents in all non-terminal
    states. The result is a TabularHumanPolicyPrior that can be used to query
    action distributions.
    
    Algorithm overview:
        1. Build the DAG of reachable states using world_model.get_dag()
        2. Compute dependency levels for topological ordering
        3. Process states in reverse topological order:
           - Terminal states (and states where the goal is already achieved): V(s, g) = 0
             Achievement is handled as reward-on-transition via the successor term achievement(s'),
             i.e., goal attainment is credited when entering s rather than by adding value at s.
           - Non-terminal states (with D = transition duration from world_model):
             * Q(s, a, g) = E[achievement(s') + (1-achievement(s')) · e^{-ρ_h·D} · V(s', g)]
             * π(a|s,g) = softmax(β * Q(s, *, g))
             * V(s, g) = Σ_a π(a|s,g) * Q(s, a, g)
           When ρ_h = 0 (γ_h = 1.0), per-transition discounting is skipped (no duration queries).
    
    Args:
        world_model: A WorldModel (or MultiGridEnv) with get_state(), set_state(),
                    and transition_probabilities() methods.
        human_agent_indices: List of agent indices to compute policies for.
                            Other agents are modeled by believed_others_policy.
        possible_goal_generator: Generator that yields (goal, weight) pairs for
                                each state and agent. See PossibleGoalGenerator.
                                If None, uses world_model.possible_goal_generator
                                (which can be set via config file 'possible_goals' key).
        believed_others_policy: Function(state, agent_index, action) -> List[(prob, action_profile)]
                               specifying beliefs about other agents' actions.
                               action_profile must be a numpy array of int64.
                               If None, uses uniform distribution over all action profiles.
        beta_h: Inverse temperature for Boltzmann policy. Higher = more deterministic.
                Use float('inf') for pure argmax (greedy) policy.
        gamma_h: Discount factor for future rewards (0 < gamma_h ≤ 1). At most one of
                gamma_h or rho_h may be provided. If gamma_h is given, rho_h is computed
                as -ln(gamma_h). Defaults to 1.0 (no discounting) if neither is provided.
        rho_h: Continuous-time discount rate (rho_h ≥ 0). At most one of gamma_h or
               rho_h may be provided. If rho_h is given, gamma_h is computed as exp(-rho_h).
        parallel: If True, use multiprocessing for parallel computation.
                 Requires 'fork' context (works on Linux, may not work on macOS/Windows).
        num_workers: Number of parallel workers. If None, uses mp.cpu_count().
        level_fct: Optional function(state) -> int that returns the "level" of a state
                  for fast dependency computation. States at higher levels are processed
                  first. If None, uses general topological sort (slower for large DAGs).
        return_Vh: If True, also return the computed value function.
        return_attainment_cache: If True, also return the attainment cache containing
            precomputed goal achievement values for each (state_index, action_profile_index, goal)
            combination. This cache can be passed to compute_robot_policy() to avoid
            recomputing these values in Phase 2.
        progress_callback: Optional callback(done, total) for progress updates.
        quiet: If True, suppress progress output.
        min_free_memory_fraction: Minimum free memory as fraction of total (0.0-1.0).
            When free memory falls below this threshold, computation pauses for
            memory_pause_duration seconds, then checks again. If still low, raises
            KeyboardInterrupt for graceful shutdown. Set to 0.0 to disable (default).
        memory_check_interval: How often to check memory, in states processed.
        memory_pause_duration: How long to pause (seconds) when memory is low.
        memory_profile: If True, enable detailed memory profiling using deep_sizeof
            to measure actual sizes of Vh_values, policies, and cache structures.
            This adds overhead (O(total_size) traversal) so defaults to False.
            When False, only RSS (process memory) is logged.
        use_disk_slicing: If True, slice DAG by timestep and save to disk, loading
            only the necessary slice during backward induction. This reduces memory
            usage by 10-20x but requires level_fct to be provided. Recommended for
            large state spaces (>10K states).
        disk_cache_dir: Directory to store DAG slices (None = use temp directory).
        use_float16: If True (default), convert transition probabilities to float16
            for 50% memory savings with negligible precision loss.
        use_compression: If True, compress disk slices with gzip (slower but smaller).
        archive_dir: Directory to save archived value slices (None = don't archive).
            Separate from disk_cache_dir - this is where final archives go.
        use_attainment_cache: If True (default), cache is_achieved() results for reuse
            in Phase 2. Set to False to save ~3GB memory at the cost of recomputing
            is_achieved() in Phase 2 (negligible for simple goals like ReachCellGoal).
        optimistic: If False (default), compute the min (worst-case) over robot action
            profiles. If True, compute the max (best-case) over robot action profiles.
    
    Returns:
        TabularHumanPolicyPrior: Policy prior that can be called as prior(state, agent, goal).
        
        If return_Vh=True, returns tuple (policy_prior, Vh_values_dict) where
        Vh_values_dict maps state -> agent_idx -> goal -> float.
        
        If return_attainment_cache=True, returns tuple with attainment_cache as
        the last element. The cache maps state_index -> action_profile_index -> 
        goal -> ndarray of 0/1 values for successor states.
    
    Performance notes:
        - State space must be finite and acyclic (DAG structure)
        - Time complexity: O(|S| * |A|^n * |G|) where |S|=states, |A|=actions,
          n=agents, |G|=goals per state
        - Memory: O(|S| * n * |G|) for storing V-values and policies
        - Parallel mode provides ~linear speedup for large state spaces
    
    Example:
        >>> env = SmallOneOrTwoChambersMapEnv()
        >>> 
        >>> class ReachGoal(PossibleGoalGenerator):
        ...     def generate(self, state, agent_idx):
        ...         yield (MyGoal(env, target=(3, 7)), 1.0)
        >>> 
        >>> policy = compute_human_policy_prior(
        ...     env, 
        ...     human_agent_indices=[0], 
        ...     possible_goal_generator=ReachGoal(env),
        ...     beta_h=5.0
        ... )
        >>> 
        >>> state = env.get_state()
        >>> action_dist = policy(state, 0, my_goal)  # numpy array of probabilities
    """
    # Use world_model's goal generator if none provided
    if possible_goal_generator is None:
        possible_goal_generator = getattr(world_model, 'possible_goal_generator', None)
        if possible_goal_generator is None:
            raise ValueError(
                "possible_goal_generator must be provided either as an argument "
                "or via world_model.possible_goal_generator (set in config file)"
            )
    
    human_policy_priors: HumanPolicyDict = {}  # these will be a mixture of system-1 and system-2 policies

    # Q_vectors = {}
    system2_policies: HumanPolicyDict = {}  # these will be Boltzmann policies with fixed inverse temperature beta for now
    # V_values will be indexed as V_values[state_index][agent_index][possible_goal]
    # Using nested lists for faster access on first two levels

    num_agents: int = len(world_model.agents)  # type: ignore[attr-defined]
    num_actions_per_agent: List[int] = world_model.num_actions_per_agent

    # Compute robot agent indices and all possible robot action profiles
    robot_agent_indices: List[int] = [i for i in range(num_agents) if i not in human_agent_indices]
    robot_action_profiles: List[List[int]] = [list(profile) for profile in product(*[range(num_actions_per_agent[i]) for i in robot_agent_indices])] if robot_agent_indices else [[]]

    if believed_others_policy is None:
        # Create precomputed default believed others policy (fast for sequential execution)
        believed_others_policy = DefaultBelievedOthersPolicy(
            num_agents, num_actions_per_agent, human_agent_indices, robot_agent_indices)
        believed_others_policy_pickle: Optional[bytes] = None  # No need to pickle the default
    else:
        # Serialize custom believed_others_policy using cloudpickle for parallel mode
        # cloudpickle can serialize lambdas, closures, and other functions that
        # standard pickle cannot handle
        believed_others_policy_pickle = cloudpickle.dumps(believed_others_policy)

    # Precompute powers for action profile indexing
    action_powers: npt.NDArray[np.int64] = np.cumprod([1] + num_actions_per_agent[:-1]).astype(np.int64)

    # Resolve gamma_h / rho_h: at most one may be provided (neither → default 1.0)
    if gamma_h is not None and rho_h is not None:
        raise ValueError("Specify at most one of gamma_h or rho_h, not both.")
    if gamma_h is not None:
        if not (0.0 < gamma_h <= 1.0):
            raise ValueError(f"gamma_h must be in (0, 1], got {gamma_h}")
        rho_h = 0.0 if gamma_h == 1.0 else -math.log(gamma_h)
    elif rho_h is not None:
        if rho_h < 0.0:
            raise ValueError(f"rho_h must be >= 0, got {rho_h}")
        gamma_h = math.exp(-rho_h)
    else:
        # Default: no discounting
        gamma_h = 1.0
        rho_h = 0.0

    # first get the dag of the world model:
    states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True, quiet=quiet)
    
    # Free state_to_idx immediately - it's not used anywhere after this
    del state_to_idx
    gc.collect()
    
    # Print memory estimate
    if not quiet:
        mem_stats = estimate_dag_memory(states, transitions, num_agents, 
                                       len(list(possible_goal_generator.generate(states[0], human_agent_indices[0]))),
                                       max(num_actions_per_agent),
                                       include_cache=use_attainment_cache,
                                       num_humans=len(human_agent_indices),
                                       num_robots=len(robot_agent_indices))
        print(f"\nMemory estimate:")
        print(f"  States: {mem_stats['num_states']:,} ({mem_stats['states_mb']:.1f} MB)")
        print(f"  Transitions: {mem_stats['num_transitions']:,} ({mem_stats['transitions_mb']:.1f} MB)")
        print(f"  Phase 1 Vh_values: {mem_stats['phase1_vh_mb']:.1f} MB")
        print(f"  Phase 1 policies: {mem_stats['phase1_policies_mb']:.1f} MB")
        print(f"  Phase 2 Vh_values: {mem_stats['phase2_vh_mb']:.1f} MB")
        print(f"  Phase 2 Vr_values: {mem_stats['phase2_vr_mb']:.1f} MB")
        print(f"  Phase 2 robot_policy: {mem_stats['phase2_robot_policy_mb']:.1f} MB")
        if 'attainment_cache_mb' in mem_stats:
            print(f"  Attainment cache: {mem_stats['attainment_cache_mb']:.1f} MB")
        else:
            print(f"  Attainment cache: DISABLED (use_attainment_cache=False)")
        print(f"  Python baseline: {mem_stats['python_baseline_mb']:.0f} MB")
        print(f"  TOTAL (recommended allocation): {mem_stats['total_mb']:.0f} MB")
        
        # Measure ACTUAL memory of DAG structures
        print(f"\nActual memory usage (after DAG build):")
        print(f"  Process RSS: {get_process_memory_mb():.1f} MB")
        states_actual = deep_sizeof(states) / (1024**2)
        print(f"  states: {states_actual:.1f} MB (estimate: {mem_stats['states_mb']:.1f} MB)")
        transitions_actual = deep_sizeof(transitions) / (1024**2)
        print(f"  transitions: {transitions_actual:.1f} MB (estimate: {mem_stats['transitions_mb']:.1f} MB)")
        successors_actual = deep_sizeof(successors) / (1024**2)
        print(f"  successors: {successors_actual:.1f} MB (not in estimate)")
        print(f"  Total DAG: {states_actual + transitions_actual + successors_actual:.1f} MB")
    
    # Handle disk-based slicing if requested
    disk_dag: Optional[DiskBasedDAG] = None
    if use_disk_slicing:
        if level_fct is None:
            raise ValueError("use_disk_slicing=True requires level_fct to be provided")
        
        # Calculate num_action_profiles for cache initialization
        num_action_profiles_for_cache = int(np.prod(num_actions_per_agent))
        
        if not quiet:
            print(f"\nCreating disk-based DAG slices...")
        
        disk_dag = DiskBasedDAG.from_dag(
            states, transitions, level_fct,
            cache_dir=disk_cache_dir,
            use_compression=use_compression,
            use_float16=use_float16,
            num_action_profiles=num_action_profiles_for_cache,
            quiet=quiet
        )
        
        # Free the full transitions from memory (will load slices on demand)
        del transitions
        transitions = None  # type: ignore
        
        # Free successors too if not needed for parallel archival
        # (will be reconstructed from disk if needed for sequential archival)
        if not parallel or archive_dir is None:
            del successors
        
        # CRITICAL: Clear the world_model's DAG cache to actually free the memory
        # (otherwise Phase 2 will get the cached DAG defeating disk slicing)
        world_model.clear_dag_cache()
        
        # Store disk_dag on world_model for Phase 2 to reuse
        world_model._disk_dag = disk_dag  # type: ignore
        
        gc.collect()  # Force immediate garbage collection
        
        if not quiet:
            freed_msg = "Transitions"
            if not parallel or archive_dir is None:
                freed_msg += " and successors"
            freed_msg += " freed from memory."
            print(f"  Disk slicing complete. {freed_msg}")
            print(f"  Loaded DAG memory: {disk_dag.get_loaded_memory_mb():.1f} MB")
            print(f"  Attainment cache will also be saved to disk during computation.")
            est_per_timestep = mem_stats['total_mb'] / (disk_dag.max_timestep + 1)
            print(f"  Estimated peak memory per timestep: {est_per_timestep:.1f} MB")
    elif use_float16:
        # Just convert to float16 without disk slicing
        if not quiet:
            print(f"\nConverting transitions to float16...")
        transitions = convert_transitions_to_float16(transitions)
    
    
    # Set up default tqdm progress bar if no callback provided
    _pbar: Optional[tqdm[int]] = None
    if progress_callback is None and not quiet:
        _pbar = tqdm(total=len(states), desc="Backward induction", unit="states", leave=False)
        def progress_callback(done: int, total: int) -> None:
            if _pbar is not None:
                _pbar.n = done
                _pbar.refresh()
    
    # Initialize V_values as nested lists for faster access
    # Use VhValuesSmall (ndarray-indexed) if goal generator is indexed, otherwise VhValues (dict-based)
    use_indexed_goals = False # hasattr(possible_goal_generator, 'indexed') and possible_goal_generator.indexed
    if use_indexed_goals:
        # VhValuesSmall: List[List[NDArray[float16]]] indexed by [state][agent][goal_index]
        n_goals = getattr(possible_goal_generator, 'n_goals', None)
        if n_goals is None:
            raise ValueError("indexed goal generator must have n_goals attribute")
        Vh_values: Union[VhValues, VhValuesSmall] = [
            [np.zeros(n_goals, dtype=np.float16) for _ in range(num_agents)] 
            for _ in range(len(states))
        ]
    else:
        # VhValues: List[List[Dict[goal, float]]] indexed by [state][agent][goal]
        Vh_values = [[{} for _ in range(num_agents)] for _ in range(len(states))]
    
    # Measure Vh_values initial structure
    if not quiet:
        vh_initial = deep_sizeof(Vh_values) / (1024**2)
        print(f"\nActual memory (after Vh_values init): Process RSS = {get_process_memory_mb():.1f} MB")
        print(f"  Vh_values (empty): {vh_initial:.1f} MB")
    
    # ============================================================================
    # WARN if parallel mode was requested
    # ============================================================================
    if parallel:
        if not quiet:
            print("WARNING: Parallel mode is currently disabled due to bugs.")
            print("         Running in sequential mode instead.")
            print("         See docs/plans/bwind_parallel.md for status.")
        parallel = False  # Force to sequential
        # The read below ensures the forced value is observably used, keeping
        # static analyzers satisfied while preserving the current behavior.
        if parallel:
            raise RuntimeError(
                "Internal error: parallel should be disabled in Phase 1 backward induction."
            )
    
    # Create sliced attainment cache if enabled - it will be stored on world_model for reuse in Phase 2.
    # This avoids redundant is_achieved() computation between phases.
    # Set use_attainment_cache=False to save ~3GB memory for simple goals.
    # Structure: SlicedAttainmentCache holds slices indexed by slice_id
    # Each slice is Dict[state_index, List[Dict[goal, array]]] for efficient access.
    num_action_profiles = int(np.prod(num_actions_per_agent))
    sliced_cache: Optional[SlicedAttainmentCache] = None
    if use_attainment_cache:
        existing_cache = getattr(world_model, '_attainment_cache', None)
        if existing_cache is not None and isinstance(existing_cache, SlicedAttainmentCache):
            # Reuse existing sliced cache
            sliced_cache = existing_cache
        else:
            # Create new sliced cache
            sliced_cache = SlicedAttainmentCache(num_action_profiles)
    
    # Measure memory after sliced_cache is initialized
    if not quiet:
        cache_initial = deep_sizeof(sliced_cache) / (1024**2) if sliced_cache else 0.0
        print(f"\nActual memory (after sliced_cache init): Process RSS = {get_process_memory_mb():.1f} MB")
        print(f"  sliced_cache (empty): {cache_initial:.1f} MB")
    
    # ============================================================================
    # PARALLEL CODE TEMPORARILY DISABLED
    # The parallel implementation is currently broken and needs refactoring.
    # See docs/plans/bwind_parallel.md for the refactoring plan.
    # The code below is kept for reference but will not execute.
    # ============================================================================
    if False:  # Was: if parallel and len(states) > 1:
        # DISABLED: Parallel execution using shared memory via fork
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        if not quiet:
            print(f"Using parallel execution with {num_workers} workers")
        
        # Track already-archived levels
        archived_levels: Set[int] = set()
        
        # Compute dependency levels and max successor levels for archival
        dependency_levels: List[List[int]]
        max_successor_levels: Optional[Dict[int, int]] = None
        level_values_list: Optional[List[int]] = None
        if level_fct is not None:
            if not quiet:
                print("Using fast level computation with provided level function")
            # Pass successors to get max_successor_levels for archival
            dependency_levels, max_successor_levels, level_values_list = compute_dependency_levels_fast(
                states, level_fct, successors
            )
        else:
            if not quiet:
                print("Using general level computation")
            dependency_levels = compute_dependency_levels_general(successors)
            max_successor_levels = None
            level_values_list = None
        
        if not quiet:
            print(f"Computed {len(dependency_levels)} dependency levels")
            
            # Initialize shared data for worker processes
            # On Linux (fork), workers inherit these as copy-on-write
            params: Tuple[List[int], PossibleGoalGenerator, int, List[int], npt.NDArray[np.int64], List[int], List[List[int]], float, float, bool, float] = (
                human_agent_indices, possible_goal_generator, num_agents, num_actions_per_agent,
                action_powers, robot_agent_indices, robot_action_profiles, beta_h, gamma_h, optimistic, rho_h
            )
            
            # Use 'fork' context explicitly to ensure shared memory works
            ctx = mp.get_context('fork')
            
            # Initialize DAG in shared memory to avoid copy-on-write overhead
            # Skip if using disk_dag - we'll load slices per level instead
            if disk_dag is None:
                if not quiet:
                    print("Storing DAG in shared memory...")
                init_shared_dag(states, transitions)
            
            # Profiling counters
            if PROFILE_PARALLEL:
                prof_states_parallel: int = 0
                prof_states_sequential: int = 0
                prof_batches: int = 0
                prof_fork_time = 0.0
            prof_submit_time = 0.0
            prof_wait_time = 0.0
            prof_merge_v_time = 0.0
            prof_merge_p_time = 0.0
            prof_seq_in_par_time = 0.0
            prof_total_parallel_time = 0.0
            prof_batch_times: List[float] = []  # Worker timing for each batch
        
            # Create memory monitor if enabled (for parallel mode - check at each level)
            memory_monitor: Optional[MemoryMonitor] = None
            if min_free_memory_fraction > 0.0:
                memory_monitor = MemoryMonitor(
                    min_free_fraction=min_free_memory_fraction,
                    check_interval=1,  # Check every level in parallel mode
                    pause_duration=memory_pause_duration,
                    verbose=not quiet,
                    enabled=True
                )
            
            # For inline fallback, we need a slice cache for small levels processed sequentially
            inline_slice_cache: Optional[SliceCache] = None
            
            # Process each level sequentially, but parallelize within each level
            for level_idx, level in enumerate(dependency_levels):
                # Load disk slice for this level if using disk_dag
                if disk_dag is not None and level_values_list is not None:
                    current_level_value = level_values_list[level_idx]
                    dag_slice = disk_dag.load_slice(current_level_value)
                    # Build transitions list for this level
                    transitions = []
                    for i in range(len(states)):
                        if i in dag_slice.state_indices:
                            transitions.append(dag_slice.get_transitions(i))
                        else:
                            transitions.append([])  # Empty for states not in this level
                    # Share the loaded slice with workers
                    init_shared_dag(states, transitions)
                
                # Check memory at the start of each level
                if memory_monitor is not None:
                    memory_monitor.check(level_idx)
                
                if DEBUG:
                    print(f"Processing level {level_idx} with {len(level)} states")
                
                if len(level) <= num_workers:
                    # Few states - process sequentially to avoid overhead
                    if PROFILE_PARALLEL:
                        _t0 = time.perf_counter()
                        prof_states_sequential += len(level)
                    
                    # Create/extend inline slice cache for these states
                    if inline_slice_cache is None:
                        inline_slice_cache = {}
                    for state_index in level:
                        if state_index not in inline_slice_cache:
                            inline_slice_cache[state_index] = [{} for _ in range(num_action_profiles)]
                    
                    for state_index in level:
                        state = states[state_index]
                        
                        # Use unified helper with inline slice cache
                        v_results, p_results = _hpp_process_single_state(
                            state_index, state, states, transitions, Vh_values,
                            human_agent_indices, possible_goal_generator,
                            num_actions_per_agent, action_powers, believed_others_policy,
                            robot_agent_indices, robot_action_profiles,
                            beta_h, gamma_h,
                            slice_cache=inline_slice_cache,
                            optimistic=optimistic,
                            rho_h=rho_h,
                            world_model=world_model,
                        )
                        
                        # Store results
                        for agent_index, agent_v in v_results.items():
                            Vh_values[state_index][agent_index].update(agent_v)
                        
                        if p_results is not None:
                            system2_policies[state] = p_results
                    
                    if PROFILE_PARALLEL:
                        prof_seq_in_par_time += time.perf_counter() - _t0
                else:
                    # Many states - parallelize
                    if PROFILE_PARALLEL:
                        _level_t0 = time.perf_counter()
                        prof_states_parallel += len(level)
                    
                    # Re-initialize shared data so new workers see updated Vh_values from previous levels
                    # Also pass the cloudpickle'd believed_others_policy for custom policy support
                    # DAG (states, transitions) is already in shared memory, only update Vh_values
                    _hpp_init_shared_data(states, transitions, Vh_values, params, believed_others_policy_pickle, 
                                          use_shared_memory=True, sliced_cache=sliced_cache, 
                                          num_action_profiles=num_action_profiles,
                                          world_model=world_model)
                    
                    # Only pass state indices - workers access shared data via globals
                    batches = split_into_batches(level, num_workers)
                    if PROFILE_PARALLEL:
                        prof_batches += len(batches)
                    
                    # Create executor per level to ensure workers fork with current Vh_values
                    if PROFILE_PARALLEL:
                        _fork_t0 = time.perf_counter()
                    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                        if PROFILE_PARALLEL:
                            prof_fork_time += time.perf_counter() - _fork_t0
                        
                        # Submit all batches (just the indices, not the data!)
                        if PROFILE_PARALLEL:
                            _submit_t0 = time.perf_counter()
                        futures = [executor.submit(_hpp_process_state_batch, batch) 
                                   for batch in batches if batch]
                        if PROFILE_PARALLEL:
                            prof_submit_time += time.perf_counter() - _submit_t0
                        
                        # Collect results and store slices
                        batches_completed = 0
                        for future in as_completed(futures):
                            # Check memory BEFORE collecting result to catch pressure early
                            # Use force=True to bypass interval check since we check per-batch
                            if memory_monitor is not None:
                                memory_monitor.check(batches_completed, force=True)
                            
                            if PROFILE_PARALLEL:
                                _wait_t0 = time.perf_counter()
                            v_results, p_results, slice_id, slice_cache, batch_time = future.result()
                            batches_completed += 1
                            if PROFILE_PARALLEL:
                                prof_wait_time += time.perf_counter() - _wait_t0
                                prof_batch_times.append(batch_time)
                            
                            # Merge Vh-values back into shared Vh_values
                            if PROFILE_PARALLEL:
                                _merge_v_t0 = time.perf_counter()
                            for state_idx, state_results in v_results.items():
                                for agent_idx, agent_results in state_results.items():
                                    Vh_values[state_idx][agent_idx].update(agent_results)
                            if PROFILE_PARALLEL:
                                prof_merge_v_time += time.perf_counter() - _merge_v_t0
                            
                            # Merge policies back
                            if PROFILE_PARALLEL:
                                _merge_p_t0 = time.perf_counter()
                            for state, state_policies in p_results.items():
                                if state not in system2_policies:
                                    system2_policies[state] = {}
                                for agent_idx, agent_policies in state_policies.items():
                                    if agent_idx not in system2_policies[state]:
                                        system2_policies[state][agent_idx] = {}
                                    system2_policies[state][agent_idx].update(agent_policies)
                            if PROFILE_PARALLEL:
                                prof_merge_p_time += time.perf_counter() - _merge_p_t0
                            
                            # Store slice cache (no merging needed - each slice is stored separately)
                            if sliced_cache is not None:
                                sliced_cache.store_slice(slice_id, slice_cache)
                            
                            # Check memory AFTER merging results (this is when memory actually increases)
                            if memory_monitor is not None:
                                memory_monitor.check(batches_completed, force=True)
                
                if PROFILE_PARALLEL:
                    prof_total_parallel_time += time.perf_counter() - _level_t0
                
                # Unload disk slice and cleanup shared memory if using disk_dag
                if disk_dag is not None and level_values_list is not None:
                    disk_dag.unload_slice(current_level_value)
                    cleanup_shared_dag()  # Free shared memory for this level's transitions
            
            # Report progress after each level
            if progress_callback:
                states_processed = sum(len(lvl) for lvl in dependency_levels[:level_idx + 1])
                progress_callback(states_processed, len(states))
            
            # Archive completed levels if archive_dir is set
            if archive_dir is not None and max_successor_levels is not None and level_values_list is not None:
                current_level_value = level_values_list[level_idx]
                archivable = detect_archivable_levels(current_level_value, max_successor_levels, quiet=quiet)
                # Only archive NEW levels (not already archived)
                new_archivable = [lvl for lvl in archivable if lvl not in archived_levels]
                if new_archivable:
                    archive_value_slices(
                        Vh_values, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vh_values.pkl",
                        return_values=return_Vh,
                        quiet=quiet
                    )
                    archived_levels.update(new_archivable)
        
        # Print profiling results
        if PROFILE_PARALLEL:
            overhead = prof_fork_time + prof_submit_time + prof_wait_time + prof_merge_v_time + prof_merge_p_time
            print("\n=== Parallelization Overhead Profile ===")
            print(f"States processed in parallel: {prof_states_parallel}")
            print(f"States processed sequentially: {prof_states_sequential}")
            print(f"Batches submitted: {prof_batches}")
            print(f"\nTime breakdown (parallel levels only):")
            print(f"  Total parallel level time:  {prof_total_parallel_time:.4f}s")
            print(f"  Fork overhead:              {prof_fork_time:.4f}s")
            print(f"  Submit overhead:            {prof_submit_time:.4f}s")
            print(f"  Wait + unpickle (result):   {prof_wait_time:.4f}s")
            print(f"  Merge Vh-values:            {prof_merge_v_time:.4f}s")
            print(f"  Merge policies:             {prof_merge_p_time:.4f}s")
            print(f"  Sequential in parallel mode:{prof_seq_in_par_time:.4f}s")
            print(f"\n  Overhead total:             {overhead:.4f}s")
            if prof_total_parallel_time > 0:
                print(f"  Overhead percentage:        {100*overhead/prof_total_parallel_time:.1f}%")
            
            # Worker load balance analysis
            if prof_batch_times:
                sum_batch = sum(prof_batch_times)
                min_batch = min(prof_batch_times)
                max_batch = max(prof_batch_times)
                mean_batch = sum_batch / len(prof_batch_times)
                print(f"\nWorker batch times ({len(prof_batch_times)} batches):")
                print(f"  Min batch time:             {min_batch:.4f}s")
                print(f"  Max batch time:             {max_batch:.4f}s")
                print(f"  Mean batch time:            {mean_batch:.4f}s")
                print(f"  Sum of all batch times:     {sum_batch:.4f}s")
                if prof_total_parallel_time > 0:
                    theoretical_speedup = sum_batch / prof_total_parallel_time
                    print(f"  Theoretical max speedup:    {theoretical_speedup:.2f}x")
                    print(f"  Load imbalance (max/mean):  {max_batch/mean_batch:.2f}x")
            print("=========================================")
        
            # Store inline slice cache in sliced_cache (states processed sequentially in parallel mode)
            if inline_slice_cache:
                inline_slice_id = make_slice_id(list(inline_slice_cache.keys()))
                sliced_cache.store_slice(inline_slice_id, inline_slice_cache)
            
            # Final archival check for parallel mode: archive any remaining levels
            if archive_dir is not None and level_fct is not None and max_successor_levels is not None:
                all_levels = sorted(set(level_fct(s) for s in states))
                remaining_levels = [lvl for lvl in all_levels if lvl not in archived_levels]
                if remaining_levels:
                    archive_value_slices(
                        Vh_values, states, level_fct, remaining_levels,
                        filepath=Path(archive_dir) / "vh_values.pkl",
                        return_values=return_Vh,
                        quiet=quiet
                    )
                    archived_levels.update(remaining_levels)
            
            # Clean up shared memory after parallel processing
            cleanup_shared_dag()
    
    else:
        # Sequential execution (original algorithm)
        # Create memory monitor if enabled
        memory_monitor: Optional[MemoryMonitor] = None
        if min_free_memory_fraction > 0.0:
            memory_monitor = MemoryMonitor(
                min_free_fraction=min_free_memory_fraction,
                check_interval=memory_check_interval,
                pause_duration=memory_pause_duration,
                verbose=not quiet,
                enabled=True
            )
        
        try:
            _hpp_compute_sequential(states, Vh_values, system2_policies, transitions,
                             human_agent_indices, possible_goal_generator,
                             num_agents, num_actions_per_agent, action_powers,
                             believed_others_policy, robot_agent_indices, robot_action_profiles,
                             beta_h, gamma_h,
                             progress_callback, memory_monitor,
                             sliced_cache, disk_dag, level_fct, archive_dir, return_Vh,
                             quiet=quiet, memory_profile=memory_profile, optimistic=optimistic,
                             rho_h=rho_h, world_model=world_model)
        except KeyboardInterrupt:
            if not quiet:
                print("\n[Phase1] Computation interrupted (KeyboardInterrupt).")
                print("         Partial results have been computed.")
            # Re-raise to allow caller to handle
            raise
    
    # Store sliced attainment cache on world_model for automatic reuse in Phase 2.
    # This avoids redundant is_achieved() computation between phases.
    world_model._attainment_cache = sliced_cache  # type: ignore[attr-defined]
    
    human_policy_priors = system2_policies # TODO: mix with system-1 policies!

    policy_prior = TabularHumanPolicyPrior(
        world_model=world_model, human_agent_indices=human_agent_indices, 
        possible_goal_generator=possible_goal_generator, values=human_policy_priors
    )
    
    if _pbar is not None:
        _pbar.close()
    
    # Clean up shared attainment cache memory (parallel mode only)
    cleanup_shared_attainment_cache()
    
    # Print actual memory after backward induction
    if not quiet:
        print(f"\nActual memory usage (after Phase 1 backward induction):")
        print(f"  Process RSS: {get_process_memory_mb():.1f} MB")
        vh_actual = deep_sizeof(Vh_values) / (1024**2)
        policies_actual = deep_sizeof(system2_policies) / (1024**2)
        cache_actual = deep_sizeof(sliced_cache) / (1024**2) if sliced_cache is not None else 0.0
        print(f"  Vh_values: {vh_actual:.1f} MB")
        print(f"  system2_policies: {policies_actual:.1f} MB")
        if sliced_cache is not None:
            print(f"  sliced_cache: {cache_actual:.1f} MB")
        else:
            print(f"  sliced_cache: DISABLED")
        print(f"  Total measured Phase 1 structures: {vh_actual + policies_actual + cache_actual:.1f} MB")
    
    if return_Vh:
        # Convert V_values from list-indexed to state-indexed dict
        Vh_values_dict = {}
        for state_idx, state in enumerate(states):
            if any(Vh_values[state_idx][agent_idx] for agent_idx in range(num_agents)):
                Vh_values_dict[state] = {agent_idx: Vh_values[state_idx][agent_idx] 
                                        for agent_idx in human_agent_indices
                                        if Vh_values[state_idx][agent_idx]}
        if return_attainment_cache:
            return policy_prior, Vh_values_dict, sliced_cache
        return policy_prior, Vh_values_dict
    
    if return_attainment_cache:
        return policy_prior, sliced_cache
    
    return policy_prior
