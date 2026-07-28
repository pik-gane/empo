"""
Phase 2: Robot Policy Computation via Backward Induction.

This module implements backward induction on the state DAG to compute
robot policies that maximize human empowerment.

Main function:
    compute_robot_policy: Compute tabular robot policy via backward induction.

The algorithm computes the robot's power-law policy by:
1. Building the DAG of reachable states and transitions
2. Processing states in reverse topological order (from terminal to initial)
3. Computing robot Q-values based on expected future robot values
4. Computing robot policy as power-law distribution over Q-values
5. Computing human expected goal achievement values under robot policy

Attainment Cache:
    Phase 2 automatically reuses the attainment cache computed in Phase 1 if it's
    stored on the world_model (via world_model._attainment_cache). This avoids
    redundant is_achieved() computation between phases.
"""

import math
import numpy as np
import numpy.typing as npt
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any, Union, overload, Literal, Set

import cloudpickle
from tqdm import tqdm
from scipy.special import logsumexp

from empo.util.memory_monitor import MemoryMonitor, deep_sizeof, get_process_memory_mb
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.robot_policy import RobotPolicy
from empo.world_model import WorldModel
from empo.backward_induction.shared_dag import (
    init_shared_dag, get_shared_dag, attach_shared_dag, cleanup_shared_dag
)

from .helpers import (
    State, TransitionData,
    SliceCache, SliceId, SlicedAttainmentCache, make_slice_id,
    compute_dependency_levels_general,
    compute_dependency_levels_fast,
    split_into_batches,
    SlicedList,
    detect_archivable_levels,
    archive_value_slices,
    VhValues,
    VhValuesSmall,
)
from .phase1 import compute_human_policy_prior

# Type aliases
VrValues = npt.NDArray[np.floating[Any]]  # Indexed as Vr_values[state_index]
RobotActionProfile = Tuple[int, ...]
RobotPolicyDict = Dict[State, Dict[RobotActionProfile, float]]  # state -> robot_action_profile -> prob
MarkovChain = List[Dict[int, float]]  # state_index -> {successor_state_index -> transition_probability}

DEBUG = False  # Set to True for verbose debugging output

# Module-level globals for shared memory in forked processes (Phase 2)
_shared_states: Optional[List[State]] = None
_shared_transitions: Optional[List[List[TransitionData]]] = None
_shared_Vh_values: Optional[Union[VhValues, VhValuesSmall]] = None
_shared_Vr_values: Optional[VrValues] = None
_shared_robot_agent_indices: Optional[List[int]] = None
_shared_human_policy_prior_pickle: Optional[bytes] = None
_shared_sliced_cache: Optional[SlicedAttainmentCache] = None
_shared_num_action_profiles: int = 0
_shared_rp_params: Optional[Tuple[List[int], List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float, float, float, float, float, float, float, float]] = None
_shared_world_model: Optional['WorldModel'] = None  # For duration-aware discounting in workers


def _rp_process_single_state(
    state_index: int,
    state: State,
    states: List[State],
    state_transitions: List[TransitionData],
    Vh_values: Union[VhValues, VhValuesSmall],
    Vr_values: VrValues,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    robot_action_profiles: List[RobotActionProfile],
    possible_goal_generator: PossibleGoalGenerator,
    num_agents: int,
    num_actions: int,
    action_powers: npt.NDArray[np.int64],
    human_policy_prior: TabularHumanPolicyPrior,
    beta_r: float,
    gamma_h: float,
    gamma_r: float,
    zeta: float,
    xi: float,
    eta: float,
    terminal_Vr: float,
    slice_cache: Optional[SliceCache] = None,
    use_indexed: bool = False,
    vres0: Union[Dict, npt.NDArray] = None,
    compute_successor_probs: bool = False,
    rho_h: float = 0.0,
    rho_r: float = 0.0,
    world_model: Optional['WorldModel'] = None,
) -> Tuple[
    Dict[int, Dict[PossibleGoal, float]],  # vh_results: agent -> goal -> value
    float,  # vr_result
    Optional[Dict[RobotActionProfile, float]],  # robot_policy (None for terminal)
    Dict[int, float],  # successor_probs: successor state_index -> transition probability
]:
    """Process a single state for Phase 2, returning (vh_results, vr_result, robot_policy, successor_probs).
    
    Unified implementation for sequential, parallel batch, and inline fallback.
    Handles both terminal and non-terminal states correctly.
    
    Args:
        state_index: Index of the state in the states list
        state: The state to process
        states: Full states list (needed for goal achievement checks)
        state_transitions: Transitions for this state only
        Vh_values: Human value function (reads from successors)
        Vr_values: Robot value function (reads from successors)
        human_agent_indices: List of human agent indices
        robot_agent_indices: List of robot agent indices
        robot_action_profiles: Precomputed list of robot action profiles
        possible_goal_generator: Generator for possible goals
        num_agents: Total number of agents
        num_actions: Number of actions available
        action_powers: Precomputed powers for action profile indexing
        human_policy_prior: Human policy prior for computing expectations
        beta_r: Robot inverse temperature (power-law parameter)
        gamma_h: Human discount factor
        gamma_r: Robot discount factor
        zeta: Risk-aversion parameter
        xi: Inter-human power-inequality aversion
        eta: Intertemporal power-inequality aversion
        terminal_Vr: Value for terminal states
        slice_cache: Optional SliceCache for this worker's batch (for writing).
            Structure: Dict[state_index, List[Dict[goal, array]]]
    
    Returns:
        Tuple of:
        - vh_results: Dict[agent_index, Dict[goal, float]] - V_h^e values for this state
        - vr_result: float - V_r value for this state
        - robot_policy: Dict[RobotActionProfile, float] or None (None for terminal states)
        - successor_probs: Dict[successor_state_index, float] - aggregate transition
          probabilities to successor states under the joint robot+human policy
          (empty dict for terminal states)
    """
    if slice_cache is not None and state_index in slice_cache:
        this_state_cache = slice_cache[state_index]
    else:
        this_state_cache = None
    
    is_terminal = not state_transitions
    
    if is_terminal:
        # Terminal state: V_h^e = 0 for all goals (dict defaults to 0), V_r = terminal_Vr, no robot policy
        vh_results: Dict[int, Dict[PossibleGoal, float]] = {}
        if DEBUG:
            print(f"  Terminal state {state_index}")
        return vh_results, terminal_Vr, None, {}
    
    # Non-terminal state: compute Q_r, pi_r, V_h^e, X_h, U_r, V_r
    vh_results = {}
    action_profile: npt.NDArray[np.int64] = np.zeros(num_agents, dtype=np.int64)
    
    # Cache duration arrays per action_profile_index to avoid repeated world_model queries.
    # Durations depend on (state, action_profile, transitions) which are uniquely identified
    # by action_profile_index for a given state. The cache is reused across Q_r and V_h^e loops.
    _duration_cache: Dict[int, npt.NDArray] = {}
    
    if DEBUG:
        print(f"  Transient state {state_index}")
    
    # Compute Q_r values for all robot action profiles
    Qr_values = np.zeros(len(robot_action_profiles))
    # Track expected duration weight per robot action profile (for duration-weighted reward)
    duration_weights_per_rap = np.zeros(len(robot_action_profiles)) if rho_r > 0.0 else None
    for robot_action_profile_index, robot_action_profile in enumerate(robot_action_profiles):
        action_profile[robot_agent_indices] = robot_action_profile
        v = 0.0
        dw = 0.0
        for human_action_profile_prob, human_action_profile in human_policy_prior.profile_distribution(state):
            action_profile[human_agent_indices] = human_action_profile
            action_profile_index = (action_profile @ action_powers).item()
            _, next_state_probabilities, next_state_indices = state_transitions[action_profile_index]
            if rho_r > 0.0:
                # Duration-aware discounting: e^{-rho_r * D(s, a, s')} per transition
                if world_model is None:
                    raise ValueError("world_model is required for duration-aware discounting (rho_r > 0)")
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
                discount_factors_r = np.exp(-rho_r * durations_arr)
                v += human_action_profile_prob * np.dot(next_state_probabilities, discount_factors_r * Vr_values[next_state_indices])
                # Duration weight: (1 - e^{-rho*D}) / rho for reward term
                # Use -expm1(-x) = 1 - e^{-x} for numerical stability when rho_r*D is small
                duration_weight_factors = -np.expm1(-rho_r * durations_arr) / rho_r
                dw += human_action_profile_prob * np.dot(next_state_probabilities, duration_weight_factors)
            else:
                v += human_action_profile_prob * np.dot(next_state_probabilities, Vr_values[next_state_indices])
        Qr_values[robot_action_profile_index] = v  # discounting already applied per-transition
        if rho_r > 0.0:
            duration_weights_per_rap[robot_action_profile_index] = dw
    
    # Compute robot policy as power-law policy
    # Use log-space computation for numerical stability:
    # pi_r(a) ∝ (-Q_r(a))^{-beta_r} = exp(-beta_r * log(-Q_r(a)))
    log_neg_Qr = np.log(-Qr_values)  # Q_r values are always negative
    log_powers = -beta_r * log_neg_Qr
    log_normalizer = logsumexp(log_powers)
    ps = np.exp(log_powers - log_normalizer)
    robot_policy = {robot_action_profile: ps[idx] 
                   for idx, robot_action_profile in enumerate(robot_action_profiles)}
    
    # Compute aggregate transition probabilities under joint robot+human policy
    # (only when requested, to avoid unnecessary overhead):
    # P(s'|s) = sum_{a_r} pi_r(a_r|s) * sum_{a_h} pi_h(a_h|s) * T(s'|s, a)
    successor_probs: Dict[int, float] = {}
    if compute_successor_probs:
        for robot_action_profile_index, robot_action_profile in enumerate(robot_action_profiles):
            robot_weight = ps[robot_action_profile_index]
            if robot_weight == 0.0:
                continue
            action_profile[robot_agent_indices] = robot_action_profile
            for human_action_profile_prob, human_action_profile in human_policy_prior.profile_distribution(state):
                joint_weight = robot_weight * human_action_profile_prob
                if joint_weight == 0.0:
                    continue
                action_profile[human_agent_indices] = human_action_profile
                action_profile_index = (action_profile @ action_powers).item()
                _, next_state_probabilities, next_state_indices = state_transitions[action_profile_index]
                for prob, succ_idx in zip(next_state_probabilities, next_state_indices):
                    p = joint_weight * prob
                    if p > 0.0:
                        successor_probs[succ_idx] = successor_probs.get(succ_idx, 0.0) + p
    
    # Compute V_h^e, X_h, and U_r values
    powersum = 0.0  # sum over humans of X_h^(-xi)
    for agent_index in human_agent_indices:
        vh_agent = vh_results[agent_index] = vres0.copy() if use_indexed else {}
        
        if DEBUG:
            print(f"   Human agent {agent_index}")
            # Check if at least one goal is achieved in this state
            goals_achieved = []
            for pg, _ in possible_goal_generator.generate(state, agent_index):
                achieved = pg.is_achieved(state)
                goals_achieved.append((pg, achieved))
            if not any(a for _, a in goals_achieved):
                print(f"   WARNING: No goal achieved in state {state_index}!")
                for pg, a in goals_achieved:
                    print(f"     {pg}: is_achieved={a}")
        
        xh = 0.0
        some_goal_achieved_with_positive_prob = False
        
        for possible_goal, possible_goal_weight in possible_goal_generator.generate(state, agent_index):
            if DEBUG:
                print(f"    Possible goal: {possible_goal}")
            
            key = possible_goal.index if use_indexed else possible_goal
            vh = 0.0
            for robot_action_profile_index, robot_action_profile in enumerate(robot_action_profiles):
                action_profile[robot_agent_indices] = robot_action_profile
                v = 0.0
                for human_action_profile_prob, human_action_profile in human_policy_prior.profile_distribution_with_fixed_goal(state, agent_index, possible_goal):
                    action_profile[human_agent_indices] = human_action_profile
                    action_profile_index = (action_profile @ action_powers).item()
                    _, next_state_probabilities, next_state_indices = state_transitions[action_profile_index]
                    
                    # Look up attainment values from Phase 1 cache
                    # The slice_cache is pre-populated with all values for this batch from Phase 1
                    cached = None
                    
                    if slice_cache is not None:
                        cached = this_state_cache[action_profile_index].get(possible_goal)
                    
                    if cached is not None:
                        attainment_values_array = cached
                    else:
                        # Cache miss - compute attainment values
                        # This should rarely happen if Phase 1 populated the cache correctly
                        attainment_values_array = np.fromiter(
                            (possible_goal.is_achieved(states[next_state_index]) 
                             for next_state_index in next_state_indices),
                            dtype=np.int8,
                            count=len(next_state_indices)
                        )
                    
                    if np.dot(next_state_probabilities, attainment_values_array) > 0.0:
                        some_goal_achieved_with_positive_prob = True
                    
                    # Read successor values - use goal.index if indexed, else goal as key
                    if use_indexed:
                        try:
                            vhe_values_array = np.fromiter(
                                (Vh_values[next_state_index][agent_index][key]
                                for next_state_index in next_state_indices),
                                dtype=np.float16,
                                count=len(next_state_indices)
                            )
                        except KeyError:
                            raise KeyError(f"Key error {Vh_values[next_state_indices[0]][agent_index]}")
                    else:
                        vhe_values_array = np.fromiter(
                            (Vh_values[next_state_index][agent_index].get(possible_goal, 0)
                             for next_state_index in next_state_indices),
                            dtype=np.float16,
                            count=len(next_state_indices)
                        )
                    # Use np.where to avoid intermediate array allocation
                    # NumPy automatically promotes float16 to float64 during computation
                    if rho_h > 0.0:
                        # Duration-aware discounting for human achievement values
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
                        discount_factors_h = np.exp(-rho_h * durations_arr)
                        successor_values = np.where(attainment_values_array, 1.0, discount_factors_h * vhe_values_array)
                    else:
                        # Standard gamma discounting (not duration-aware)
                        successor_values = np.where(attainment_values_array, 1.0, gamma_h * vhe_values_array)
                    v += human_action_profile_prob * np.dot(
                        next_state_probabilities,
                        successor_values
                    )
                vh += ps[robot_action_profile_index] * v
            
            # Store computed value - use goal.index if indexed, else goal as key
            if vh != 0.0:
                vh_agent[key] = np.float16(vh)
            xh += possible_goal_weight * vh**zeta
            
            if DEBUG:
                print(f"      ...Vh = {vh:.4f}")
        
        assert some_goal_achieved_with_positive_prob, \
            f"No goal achievable with positive probability for agent {agent_index} in state {state_index}!"
        
        if xh == 0:
            # xh is zero means no goal has positive expected achievement value
            raise ValueError(
                f"xh=0 for agent {agent_index} in state {state_index}: "
                f"no goal is reachable! State: {state}"
            )
        
        if DEBUG:
            print(f"   ...Xh = {xh:.4f}")
        
        powersum += xh**(-xi)
    
    y = powersum / len(human_agent_indices)  # average over humans
    ur = -(y**eta)
    if rho_r > 0.0:
        # Duration-weighted reward: expected (1-e^{-rho*D})/rho under the joint policy
        expected_duration_weight = float(np.dot(ps, duration_weights_per_rap))
        vr = expected_duration_weight * ur + float(np.dot(ps, Qr_values))
    else:
        vr = ur + float(np.dot(ps, Qr_values))
    
    if DEBUG:
        print(f"  ...Ur = {ur:.4f}, Vr = {vr:.4f}")
    
    return vh_results, vr, robot_policy, successor_probs


def _rp_compute_sequential(
    states: List[State], 
    Vh_values: Union[VhValues, VhValuesSmall],  # result is inserted into this!
    Vr_values: VrValues,  # result is inserted into this!
    robot_policy: RobotPolicyDict,  # result is inserted into this! 
    transitions: Optional[List[List[Tuple[Tuple[int, ...], List[float], List[int]]]]],
    human_agent_indices: List[int], 
    robot_agent_indices: List[int], # the AI coordinates all robots 
    possible_goal_generator: PossibleGoalGenerator,
    num_agents: int, 
    num_actions: int, 
    action_powers: npt.NDArray[np.int64],
    human_policy_prior: TabularHumanPolicyPrior, 
    beta_r: float, # softmax parameter for robots' power-law softmax policies
    gamma_h: float, # humans' discount factor
    gamma_r: float, # robots' discount factor
    zeta: float, # robots' risk-aversion
    xi: float, # robots' inter-human power-inequality aversion
    eta: float, # robots' additional intertemporal power-inequality aversion
    terminal_Vr: float = -1e-10,  # must be strictly negative !
    progress_callback: Optional[Callable[[int, int], None]] = None,
    memory_monitor: Optional[MemoryMonitor] = None,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    level_fct: Optional[Callable[[State], int]] = None,
    return_values: bool = False,
    archive_dir: Optional[str] = None,
    disk_dag: Optional[Any] = None,
    quiet: bool = False,
    memory_profile: bool = False,
    markov_chain: Optional[MarkovChain] = None,
    rho_h: float = 0.0,
    rho_r: float = 0.0,
    world_model: Optional['WorldModel'] = None,
) -> None:
    """Sequential Phase 2 backward induction algorithm.
    
    Processes states in reverse topological order using the unified
    _process_single_state_phase2 helper.
    
    Args:
        markov_chain: Optional pre-allocated list (len = num_states) to fill with
            aggregate transition probabilities. Each entry markov_chain[state_index]
            will be set to a dict mapping successor state_index to transition
            probability under the joint robot+human policy.
    """
    # Generate all possible robot action profiles (cartesian product of actions for each robot)
    robot_action_profiles: List[RobotActionProfile] = [
        tuple(actions) for actions in product(range(num_actions), repeat=len(robot_agent_indices))
    ]
    
    total_states = len(states)
    
    # Memory tracking interval
    memory_report_interval = max(1, total_states // 10)  # Report ~10 times during computation
    
    # Determine if using indexed goals and initialize templates
    use_indexed = False #hasattr(possible_goal_generator, 'indexed') and possible_goal_generator.indexed
    if use_indexed:
        n_goals = possible_goal_generator.n_goals
        vres0 = np.zeros(n_goals, dtype=np.float16)
        print(f"Using indexed goals: VhValuesSmall with numpy arrays (n_goals={n_goals})")
    else:
        vres0 = {}

    # In sequential mode, we use a single slice containing all states
    # Retrieve the slice cache populated by Phase 1 (only has non-terminal states)
    slice_cache: Optional[SliceCache] = None
    if sliced_cache is not None:
        # Phase 1 only created entries for non-terminal states
        all_state_indices = list(range(len(states)))
        slice_id = make_slice_id(all_state_indices)
        slice_cache = sliced_cache.get_slice(slice_id)
    
    # Compute max_successor_levels for archival if level_fct and archive_dir provided
    max_successor_levels: Optional[Dict[int, int]] = None
    archived_levels: Set[int] = set()  # Track already-archived levels
    previous_level: Optional[int] = None  # Track level transitions
    if level_fct is not None and archive_dir is not None:
        if not quiet:
            print("Computing dependency levels for archival...")
        from .helpers import compute_dependency_levels_fast, detect_archivable_levels, archive_value_slices
        from pathlib import Path
        # Build successors list - handle disk_dag case
        successors = []
        if disk_dag is not None:
            # Load first slice to get successors for max_successor_levels computation
            first_slice = disk_dag.load_slice(disk_dag.max_timestep)
            for i in range(len(states)):
                if i in first_slice.state_indices:
                    state_transitions = first_slice.get_transitions(i)
                    succ_set = set()
                    for action_profile, probs, succ_indices in state_transitions:
                        succ_set.update(succ_indices)
                    successors.append(list(succ_set))
                else:
                    successors.append([])
            # Continue loading remaining slices to build full successors list
            for timestep in range(disk_dag.max_timestep - 1, -1, -1):
                dag_slice = disk_dag.load_slice(timestep)
                for i in dag_slice.state_indices:
                    state_transitions = dag_slice.get_transitions(i)
                    succ_set = set()
                    for action_profile, probs, succ_indices in state_transitions:
                        succ_set.update(succ_indices)
                    successors[i] = list(succ_set)
                disk_dag.unload_slice(timestep)
        else:
            # Build from in-memory transitions
            assert transitions is not None, "transitions must be provided if disk_dag is None"
            for state_transitions in transitions:
                succ_set = set()
                for action_profile, probs, succ_indices in state_transitions:
                    succ_set.update(succ_indices)
                successors.append(list(succ_set))
        _, max_successor_levels, _ = compute_dependency_levels_fast(states, level_fct, successors)
    
    # Pre-compute state levels if archival is enabled (to avoid calling level_fct for every state)
    state_levels: Optional[List[int]] = None
    if level_fct is not None and archive_dir is not None:
        state_levels = [level_fct(s) for s in states]
    
    # loop over the nodes in reverse topological order:
    for state_index in range(len(states)-1, -1, -1):
        state = states[state_index]
        
        # Check for level transition BEFORE processing state (for archival)
        if state_levels is not None and max_successor_levels is not None:
            current_state_level = state_levels[state_index]
            if previous_level is not None and current_state_level != previous_level:
                # We just completed processing previous_level, check what can be archived
                archivable = detect_archivable_levels(current_state_level, max_successor_levels, quiet=quiet)
                # Only archive NEW levels (not already archived)
                new_archivable = [lvl for lvl in archivable if lvl not in archived_levels]
                if new_archivable:
                    # Archive vhe_values (Vh_values in Phase 2 is expected human achievement)
                    archive_value_slices(
                        Vh_values, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vhe_values.pkl",
                        return_values=return_values,
                        quiet=quiet
                    )
                    # Archive vr_values (robot values) - convert to list structure for archival
                    vr_list = [[Vr_values[i]] for i in range(len(Vr_values))]
                    archive_value_slices(
                        vr_list, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vr_values.pkl",
                        return_values=return_values,
                        quiet=quiet
                    )
                    archived_levels.update(new_archivable)
            previous_level = current_state_level
        
        # Load disk slice if using disk_dag
        if disk_dag is not None and level_fct is not None:
            current_level = level_fct(state)
            # Check if we need to load a new slice
            if previous_level is None or current_level != previous_level:
                # Unload previous slice
                if previous_level is not None:
                    disk_dag.unload_slice(previous_level)
                # Load new slice
                dag_slice = disk_dag.load_slice(current_level)
                previous_level = current_level
            
            # Get transitions for current state only (phase1 approach)
            if state_index in dag_slice.state_indices:
                state_transitions = dag_slice.get_transitions(state_index)
            else:
                state_transitions = []
        else:
            # Normal mode: get from transitions list
            assert transitions is not None, "transitions must be loaded"
            state_transitions = transitions[state_index]
        
        # Use unified helper
        vh_results, vr_result, p_result, successor_probs = _rp_process_single_state(
            state_index, state, states, state_transitions, Vh_values, Vr_values,
            human_agent_indices, robot_agent_indices, robot_action_profiles,
            possible_goal_generator, num_agents, num_actions, action_powers,
            human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
            slice_cache=slice_cache,
            use_indexed=use_indexed,
            vres0=vres0,
            compute_successor_probs=(markov_chain is not None),
            rho_h=rho_h,
            rho_r=rho_r,
            world_model=world_model,
        )
        
        # Store results
        for agent_index, agent_vh in vh_results.items():
            Vh_values[state_index][agent_index] = agent_vh
        
        Vr_values[state_index] = vr_result
        
        if p_result is not None:
            robot_policy[state] = p_result
        
        if markov_chain is not None:
            markov_chain[state_index] = successor_probs
        
        # Update progress first (lightweight)
        states_processed = total_states - state_index
        if progress_callback is not None:
            progress_callback(states_processed, total_states)
        
        # Periodic memory reporting
        if not quiet and states_processed > 0 and states_processed % memory_report_interval == 0:
            print(f"\n[Phase2 Memory @ {states_processed}/{total_states} states] RSS: {get_process_memory_mb():.1f} MB")
            if memory_profile:
                # Detailed profiling with deep_sizeof (adds O(total_size) overhead)
                vh_mb = deep_sizeof(Vh_values) / (1024**2)
                vr_mb = deep_sizeof(Vr_values) / (1024**2)
                pol_mb = deep_sizeof(robot_policy) / (1024**2)
                cache_mb = deep_sizeof(slice_cache) / (1024**2) if slice_cache is not None else 0.0
                print(f"  Vh_values: {vh_mb:.1f} MB, Vr_values: {vr_mb:.1f} MB, robot_policy: {pol_mb:.1f} MB, cache: {cache_mb:.1f} MB")
        
        # Check memory less frequently to reduce overhead
        if memory_monitor is not None:
            memory_monitor.check(states_processed)
    
    # Final archival check: archive any remaining levels after loop completes
    if archive_dir is not None and level_fct is not None:
        # Check if there are any levels we haven't archived yet
        # At this point, all states have been processed, so all levels should be archivable
        all_levels = sorted(set(level_fct(s) for s in states))
        remaining_levels = [lvl for lvl in all_levels if lvl not in archived_levels]
        if remaining_levels:
            # Archive vhe_values (Vh_values in Phase 2 is expected human achievement)
            archive_value_slices(
                Vh_values, states, level_fct, remaining_levels,
                filepath=Path(archive_dir) / "vhe_values.pkl",
                return_values=return_values,
                quiet=quiet
            )
            # Archive vr_values (robot values) - convert to list structure for archival
            vr_list = [[Vr_values[i]] for i in range(len(Vr_values))]
            archive_value_slices(
                vr_list, states, level_fct, remaining_levels,
                filepath=Path(archive_dir) / "vr_values.pkl",
                return_values=return_values,
                quiet=quiet
            )
            archived_levels.update(remaining_levels)
    
    # Note: slice_cache was retrieved from Phase 1, no need to store it again


def _rp_init_shared_data(
    states: List[State], 
    transitions: List[List[TransitionData]], 
    Vh_values: Union[VhValues, VhValuesSmall], 
    Vr_values: VrValues,
    params: Tuple[List[int], List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float, float, float, float, float, float],
    human_policy_prior_pickle: bytes,
    use_shared_memory: bool = False,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    num_action_profiles: int = 0,
    world_model: Optional['WorldModel'] = None,
) -> None:
    """Initialize shared data for robot policy worker processes.
    
    Args:
        states: List of states (will be stored in shared memory if use_shared_memory=True)
        transitions: List of transitions (will be stored in shared memory if use_shared_memory=True)
        Vh_values: Human value function (always passed via globals)
        Vr_values: Robot value function (always passed via globals)
        params: Parameters tuple
        human_policy_prior_pickle: Pickled human policy prior
        use_shared_memory: If True, states and transitions are already in shared memory
        sliced_cache: Optional SlicedAttainmentCache from Phase 1 for reading
        num_action_profiles: Number of action profiles (needed for slice cache creation)
        world_model: WorldModel for duration-aware discounting (needed when rho_h or rho_r > 0)
    """
    global _shared_states, _shared_transitions, _shared_Vh_values, _shared_Vr_values
    global _shared_rp_params, _shared_human_policy_prior_pickle
    global _shared_sliced_cache, _shared_num_action_profiles
    global _shared_world_model
    
    if use_shared_memory:
        # DAG is already in shared memory, just store refs as None
        _shared_states = None
        _shared_transitions = None
    else:
        _shared_states = states
        _shared_transitions = transitions
    
    _shared_Vh_values = Vh_values
    _shared_Vr_values = Vr_values
    _shared_rp_params = params
    _shared_human_policy_prior_pickle = human_policy_prior_pickle
    _shared_sliced_cache = sliced_cache
    _shared_num_action_profiles = num_action_profiles
    _shared_world_model = world_model


def _rp_process_state_batch(
    state_indices: List[int]
) -> Tuple[Dict[int, Dict[int, Dict[PossibleGoal, float]]], 
           Dict[int, float],
           Dict[State, Dict[RobotActionProfile, float]],
           SliceId,  # slice_id for this batch
           Optional[SliceCache],  # slice cache for this batch (None if not available)
           float,
           Dict[int, Dict[int, float]]]:
    """Process a batch of states for robot policy computation.
    
    Uses module-level shared data (inherited via fork) to avoid copying.
    Returns Vh-values, Vr-values, robot policies, slice_id, slice_cache, timing,
    and aggregate transition probabilities per state.
    """
    batch_start = time.perf_counter()
    
    # Access shared data - these are guaranteed to be set when called from parallel context
    assert _shared_Vh_values is not None
    assert _shared_Vr_values is not None
    assert _shared_rp_params is not None
    assert _shared_human_policy_prior_pickle is not None
    
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
    Vr_values = _shared_Vr_values
    
    (human_agent_indices, robot_agent_indices, possible_goal_generator, 
     num_agents, num_actions, action_powers, beta_r, gamma_h, gamma_r, 
     zeta, xi, eta, terminal_Vr, rho_h, rho_r) = _shared_rp_params
    
    # Deserialize human_policy_prior
    human_policy_prior = cloudpickle.loads(_shared_human_policy_prior_pickle)
    # The world_model is excluded from pickling, so we need to set num_actions directly
    # for profile_distribution to work. Use a mock attribute access pattern.
    human_policy_prior._num_actions_override = num_actions
    
    # Generate all possible robot action profiles
    robot_action_profiles: List[RobotActionProfile] = [
        tuple(actions) for actions in product(range(num_actions), repeat=len(robot_agent_indices))
    ]
    
    vh_results: Dict[int, Dict[int, Dict[PossibleGoal, float]]] = {}
    vr_results: Dict[int, float] = {}
    p_results: Dict[State, Dict[RobotActionProfile, float]] = {}
    mc_results: Dict[int, Dict[int, float]] = {}
    
    # Retrieve slice cache pre-populated by Phase 1 for this batch
    # Structure: Dict[state_index, List[Dict[goal, array]]]
    slice_id = make_slice_id(state_indices)
    slice_cache: Optional[SliceCache] = None
    if _shared_sliced_cache is not None:
        slice_cache = _shared_sliced_cache.get_slice(slice_id)
    
    for state_index in state_indices:
        state = states[state_index]
        state_transitions = transitions[state_index]
        
        # Use unified helper with slice_cache pre-populated by Phase 1
        # The sliced_cache is no longer needed for lookup since slice_cache has all data
        vh_results_state, vr_result, p_result, successor_probs = _rp_process_single_state(
            state_index, state, states, state_transitions, Vh_values, Vr_values,
            human_agent_indices, robot_agent_indices, robot_action_profiles,
            possible_goal_generator, num_agents, num_actions, action_powers,
            human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
            slice_cache=slice_cache,
            rho_h=rho_h,
            rho_r=rho_r,
            world_model=_shared_world_model,
        )
        
        vh_results[state_index] = vh_results_state
        vr_results[state_index] = vr_result
        if p_result is not None:
            p_results[state] = p_result
        mc_results[state_index] = successor_probs
    
    batch_time = time.perf_counter() - batch_start
    return vh_results, vr_results, p_results, slice_id, slice_cache, batch_time, mc_results


class TabularRobotPolicy(RobotPolicy):
    """
    Tabular (lookup-table) implementation of robot policy.
    
    This implementation stores precomputed robot policy distributions in a dictionary
    structure, indexed by state. The policy maps each state to a distribution over
    robot action profiles (joint actions for all robot agents).
    
    Computed via backward induction on the state DAG in Phase 2.
    
    Attributes:
        world_model: The world model (environment) this policy applies to.
        robot_agent_indices: List of agent indices controlled as robots.
        values: Dict mapping state -> robot_action_profile -> probability.
    """
    
    def __init__(
        self, 
        world_model: WorldModel, 
        robot_agent_indices: List[int], 
        values: RobotPolicyDict
    ):
        """
        Initialize the tabular robot policy.
        
        Args:
            world_model: The world model (environment) this policy applies to.
            robot_agent_indices: List of indices of robot agents.
            values: Precomputed policy lookup table (state -> action_profile -> prob).
        """
        self.world_model = world_model
        self.robot_agent_indices = robot_agent_indices
        self.values = values
        self.num_actions: int = world_model.action_space.n  # type: ignore[attr-defined]
    
    def __call__(self, state) -> Dict[RobotActionProfile, float]:
        """
        Get the robot action profile distribution for a state.
        
        Args:
            state: Current world state.
        
        Returns:
            Dict mapping robot action profiles to probabilities.
        """
        return self.values.get(state, {})
    
    def sample(self, state) -> RobotActionProfile:
        """
        Sample a robot action profile from the policy.
        
        Args:
            state: Current world state.
        
        Returns:
            A tuple of actions, one for each robot agent.
        """
        dist = self(state)
        if not dist:
            # No policy for this state (terminal state?), return random
            return tuple(np.random.randint(0, self.num_actions) for _ in self.robot_agent_indices)
        
        profiles = list(dist.keys())
        probs = np.fromiter((dist[p] for p in profiles), dtype=np.float64, count=len(profiles))
        probs = probs / probs.sum()  # normalize
        idx = np.random.choice(len(profiles), p=probs)
        return profiles[idx]
    
    def reset(self, world_model: WorldModel) -> None:
        """
        Reset the policy at the start of an episode.
        
        Updates the world model reference. For tabular policies, this allows
        the same policy to be used across different instances of the same
        environment type.
        
        Args:
            world_model: The environment/world model for this episode.
        """
        self.world_model = world_model
    
    def get_action(self, state, robot_agent_index: int) -> int:
        """
        Get the action for a specific robot agent.
        
        Samples from the joint policy and returns the action for the specified robot.
        
        Args:
            state: Current world state.
            robot_agent_index: Index of the robot agent.
        
        Returns:
            The action for the specified robot.
        """
        profile = self.sample(state)
        # Find position of robot_agent_index in robot_agent_indices
        pos = self.robot_agent_indices.index(robot_agent_index)
        return profile[pos]


@overload
def compute_robot_policy(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    robot_agent_indices: List[int],
    possible_goal_generator: Optional[PossibleGoalGenerator] = None,
    human_policy_prior: Optional[TabularHumanPolicyPrior] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: Optional[float] = None, 
    gamma_r: Optional[float] = None,
    rho_h: Optional[float] = None,
    rho_r: Optional[float] = None,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    terminal_Vr: float = -1e-10,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_values: Literal[False] = False,
    return_markov_chain: Literal[False] = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
) -> TabularRobotPolicy: ...


@overload
def compute_robot_policy(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    robot_agent_indices: List[int],
    possible_goal_generator: Optional[PossibleGoalGenerator] = None,
    human_policy_prior: Optional[TabularHumanPolicyPrior] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: Optional[float] = None, 
    gamma_r: Optional[float] = None,
    rho_h: Optional[float] = None,
    rho_r: Optional[float] = None,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    terminal_Vr: float = -1e-10,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_values: Literal[True],
    return_markov_chain: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    archive_dir: Optional[str] = None,
    use_disk_slicing: bool = False,
    use_compression: bool = False,
    use_float16: bool = True,
) -> Tuple[TabularRobotPolicy, Dict[State, float], Dict[State, Dict[int, Dict[PossibleGoal, float]]]]: ...


@overload
def compute_robot_policy(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    robot_agent_indices: List[int],
    possible_goal_generator: Optional[PossibleGoalGenerator] = None,
    human_policy_prior: Optional[TabularHumanPolicyPrior] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: Optional[float] = None, 
    gamma_r: Optional[float] = None,
    rho_h: Optional[float] = None,
    rho_r: Optional[float] = None,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    terminal_Vr: float = -1e-10,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_values: Literal[False] = False,
    return_markov_chain: Literal[True] = ...,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
) -> Tuple[TabularRobotPolicy, MarkovChain]: ...


def compute_robot_policy(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    robot_agent_indices: List[int],
    possible_goal_generator: Optional[PossibleGoalGenerator] = None,
    human_policy_prior: Optional[TabularHumanPolicyPrior] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: Optional[float] = None, 
    gamma_r: Optional[float] = None,
    rho_h: Optional[float] = None,
    rho_r: Optional[float] = None,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    terminal_Vr: float = -1e-10,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_values: bool = False,
    return_markov_chain: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    memory_profile: bool = False,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    archive_dir: Optional[str] = None,
    use_disk_slicing: bool = False,
    use_compression: bool = False,
    use_float16: bool = True,
) -> Union[
    TabularRobotPolicy,
    Tuple[TabularRobotPolicy, MarkovChain],
    Tuple[TabularRobotPolicy, Dict[State, float], Dict[State, Dict[int, Dict[PossibleGoal, float]]]],
    Tuple[TabularRobotPolicy, Dict[State, float], Dict[State, Dict[int, Dict[PossibleGoal, float]]], MarkovChain],
]:
    """
    Compute robot policy via backward induction on the state DAG.
    
    This function builds the complete state DAG of the world model and computes
    the robot's power-law policy that aims to maximize human empowerment.
    It simultaneously computes the expected human goal achievement values (V_h^e).
    
    Algorithm overview:
        1. Build the DAG of reachable states using world_model.get_dag()
        2. Compute dependency levels for topological ordering
        3. Process states in reverse topological order:
           - Terminal states: V_h^e(s, g) = 0, V_r(s) = terminal_Vr (strictly negative; default -1e-10)
           - Non-terminal states (with D = transition duration from world_model):
             * Q_r(s, a_r) = E[e^{-ρ_r·D} V_r(s')] under human_policy_prior
             * π_r(a_r|s) = power-law policy based on Q_r
             * V_h^e(s, g) = E[achievement(s') + (1-achievement(s')) · e^{-ρ_h·D} · V_h^e(s', g)]
             * X_h(s) = E[V_h^e(s, g)^ζ] (aggregate goal ability)
             * U_r(s) = -E[(1-e^{-ρ_r·D})/ρ_r] · K(s)^η  (duration-weighted intrinsic cost; negative)
             * V_r(s) = U_r(s) + Q_r(s, π_r)
           When ρ = 0 (γ = 1.0), per-transition discounting is skipped (no duration queries).
    
    Args:
        world_model: A WorldModel (or MultiGridEnv) with get_state(), set_state(),
                    and transition_probabilities() methods.
        human_agent_indices: List of agent indices representing humans.
        robot_agent_indices: List of agent indices representing robots.
        possible_goal_generator: Generator that yields (goal, weight) pairs for
                                each state and agent. See PossibleGoalGenerator.
                                If None, uses world_model.possible_goal_generator.
        human_policy_prior: Precomputed human policy prior from compute_human_policy_prior().
                           If None, will be computed automatically using the goal generator.
        beta_r: Power-law concentration parameter. Higher = more deterministic.
        gamma_h: Discount factor for human goal achievement values (0 < gamma_h ≤ 1).
                At most one of gamma_h or rho_h may be provided. Defaults to 1.0 if
                neither is given.
        gamma_r: Discount factor for robot values (0 < gamma_r ≤ 1). At most one of
                gamma_r or rho_r may be provided. Defaults to 1.0 if neither is given.
        rho_h: Continuous-time discount rate for humans (rho_h ≥ 0). If given, gamma_h
               is computed as exp(-rho_h).
        rho_r: Continuous-time discount rate for robots (rho_r ≥ 0). If given, gamma_r
               is computed as exp(-rho_r).
        zeta: Risk-aversion parameter for aggregate goal ability.
        xi: Inter-human power-inequality aversion parameter.
        eta: Additional intertemporal power-inequality aversion parameter.
        terminal_Vr: Value for V_r at terminal states. Must be strictly negative
                    to ensure power-law policy is well-defined. Default: -1e-10.
        parallel: If True, use multiprocessing for parallel computation.
                 Requires 'fork' context (works on Linux, may not work on macOS/Windows).
        num_workers: Number of parallel workers. If None, uses mp.cpu_count().
        level_fct: Optional function(state) -> int for fast dependency computation.
        return_values: If True, also return V_r and V_h^e value functions.
        return_markov_chain: If True, also return the induced Markov chain as a
            list indexed by state_index, where each entry is a dict mapping
            successor state_index to aggregate transition probability under the
            joint robot+human policy. Terminal states have empty dicts.
        progress_callback: Optional callback(done, total) for progress updates.
        quiet: If True, suppress progress output.
        min_free_memory_fraction: Minimum free memory as fraction of total (0.0-1.0).
            When free memory falls below this threshold, computation pauses for
            memory_pause_duration seconds, then checks again. If still low, raises
            KeyboardInterrupt for graceful shutdown. Set to 0.0 to disable (default).
        memory_check_interval: How often to check memory, in states processed.
        memory_pause_duration: How long to pause (seconds) when memory is low.
        memory_profile: If True, enable detailed memory profiling using deep_sizeof
            to measure actual sizes of Vh_values, Vr_values, robot_policy, and cache structures.
            This adds overhead (O(total_size) traversal) so defaults to False.
            When False, only RSS (process memory) is logged.
        sliced_cache: Optional SlicedAttainmentCache of precomputed goal attainment arrays.
            If not provided, Phase 2 automatically looks for the cache on world_model
            (stored automatically by Phase 1 at world_model._attainment_cache).
            
            **Automatic caching**: Phase 1 now automatically stores its sliced attainment 
            cache on the world_model, so Phase 2 will reuse it without any extra configuration.
            You don't need to pass return_attainment_cache=True to Phase 1 anymore.
            
            The sliced cache structure allows efficient read access without merging overhead.
    
    Returns:
        TabularRobotPolicy: Robot policy that can be called as policy(state).
        
        If return_values=True, returns tuple (robot_policy, Vr_dict, Vh_dict) where:
        - Vr_dict maps state -> float (robot value function)
        - Vh_dict maps state -> agent_idx -> goal -> float (human goal achievement values)
        
        If return_markov_chain=True, a MarkovChain is appended to the return tuple.
        MarkovChain is a list indexed by state_index where each entry is a dict
        mapping successor state_index to transition probability.
        
        Combined return patterns:
        - return_values=False, return_markov_chain=False: TabularRobotPolicy
        - return_values=False, return_markov_chain=True: (TabularRobotPolicy, MarkovChain)
        - return_values=True, return_markov_chain=False: (TabularRobotPolicy, Vr_dict, Vh_dict)
        - return_values=True, return_markov_chain=True: (TabularRobotPolicy, Vr_dict, Vh_dict, MarkovChain)
    
    Example:
        >>> # Phase 1 automatically stores attainment cache on world_model
        >>> human_policy = compute_human_policy_prior(env, [0], goal_gen)
        >>>
        >>> # Phase 2 automatically reuses the cache - no extra parameters needed!
        >>> robot_policy = compute_robot_policy(
        ...     env,
        ...     human_agent_indices=[0],
        ...     robot_agent_indices=[1],
        ...     possible_goal_generator=goal_gen,
        ...     human_policy_prior=human_policy,
        ...     beta_r=5.0
        ... )
        >>> 
        >>> state = env.get_state()
        >>> robot_actions = robot_policy.sample(state)  # tuple of actions
    """
    # Use world_model's goal generator if none provided
    if possible_goal_generator is None:
        possible_goal_generator = getattr(world_model, 'possible_goal_generator', None)
        if possible_goal_generator is None:
            raise ValueError(
                "possible_goal_generator must be provided either as an argument "
                "or via world_model.possible_goal_generator (set in config file)"
            )
    
    # Compute human policy prior if not provided
    if human_policy_prior is None:
        human_policy_prior = compute_human_policy_prior(
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            possible_goal_generator=possible_goal_generator,
            parallel=parallel,
            num_workers=num_workers,
            quiet=quiet
        )
    
    robot_policy_values: RobotPolicyDict = {}

    num_agents: int = len(world_model.agents)  # type: ignore[attr-defined]
    num_actions: int = world_model.action_space.n  # type: ignore[attr-defined]

    # Precompute powers for action profile indexing
    action_powers: npt.NDArray[np.int64] = num_actions ** np.arange(num_agents)

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
        gamma_h = 1.0
        rho_h = 0.0

    # Resolve gamma_r / rho_r: at most one may be provided (neither → default 1.0)
    if gamma_r is not None and rho_r is not None:
        raise ValueError("Specify at most one of gamma_r or rho_r, not both.")
    if gamma_r is not None:
        if not (0.0 < gamma_r <= 1.0):
            raise ValueError(f"gamma_r must be in (0, 1], got {gamma_r}")
        rho_r = 0.0 if gamma_r == 1.0 else -math.log(gamma_r)
    elif rho_r is not None:
        if rho_r < 0.0:
            raise ValueError(f"rho_r must be >= 0, got {rho_r}")
        gamma_r = math.exp(-rho_r)
    else:
        gamma_r = 1.0
        rho_r = 0.0

    # Serialize human_policy_prior using cloudpickle for parallel mode
    human_policy_prior_pickle = cloudpickle.dumps(human_policy_prior)

    # Check if disk_dag already exists from Phase 1
    disk_dag: Optional[Any] = getattr(world_model, '_disk_dag', None)
    
    if disk_dag is not None:
        # Reuse disk_dag from Phase 1
        if not quiet:
            print("\n=== Reusing Disk-Based DAG from Phase 1 ===")
            print(f"  Loaded DAG memory: {disk_dag.get_loaded_memory_mb():.1f} MB")
        # Get states and state_to_idx from world_model (cached during Phase 1 DAG build)
        # We need to rebuild these since they were not stored in disk_dag
        states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True, quiet=True)
        # Clear the cache again to free transitions
        world_model.clear_dag_cache()
        # Free transitions and successors - we'll load from disk_dag instead
        del transitions
        transitions = None  # type: ignore
        if archive_dir is None:
            del successors
    elif use_disk_slicing:
        # Create new disk_dag (Phase 1 didn't create one)
        if level_fct is None:
            raise ValueError("use_disk_slicing requires level_fct to be provided")
        
        from .dag_slicing import DiskBasedDAG, estimate_dag_memory
        import gc
        
        # Get the DAG first
        states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True, quiet=quiet)
        
        if not quiet:
            print("\n=== Creating Disk-Based DAG ===")
        
        num_action_profiles_for_cache = num_actions ** num_agents

        # Calculate memory stats before slicing
        # Note: num_goals not available here, use num_action_profiles as proxy
        mem_stats = estimate_dag_memory(states, transitions, num_agents,
                                        num_goals=num_action_profiles_for_cache,  # proxy
                                        num_actions=num_actions)
        if not quiet:
            print(f"Estimated memory: {mem_stats['total_mb']:.0f} MB")
        
        # Create disk-based DAG
        disk_dag = DiskBasedDAG.from_dag(
            states, transitions, level_fct,
            cache_dir=None,  # Auto-select optimal location
            use_compression=use_compression,
            use_float16=use_float16,
            num_action_profiles=num_action_profiles_for_cache,
            quiet=quiet
        )
        
        # Free the full transitions from memory (will load slices on demand)
        del transitions
        transitions = None  # type: ignore
        
        # Free successors too if not needed for archival
        if archive_dir is None:
            del successors
        
        # CRITICAL: Clear the world_model's DAG cache to actually free the memory
        world_model.clear_dag_cache()
        
        # Store on world_model for potential future use
        world_model._disk_dag = disk_dag  # type: ignore
        
        gc.collect()  # Force immediate garbage collection
        
        if not quiet:
            freed_msg = "Transitions"
            if archive_dir is None:
                freed_msg += " and successors"
            freed_msg += " freed from memory."
            print(f"  Disk slicing complete. {freed_msg}")
            print(f"  Loaded DAG memory: {disk_dag.get_loaded_memory_mb():.1f} MB")
    else:
        # Normal mode: get DAG from world_model
        states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True, quiet=quiet)
    
    # Set up default tqdm progress bar if no callback provided
    _pbar: Optional[tqdm[int]] = None
    if progress_callback is None and not quiet:
        _pbar = tqdm(total=len(states), desc="Robot policy backward induction", unit="states", leave=False)
        def progress_callback(done: int, total: int) -> None:
            if _pbar is not None:
                _pbar.n = done
                _pbar.refresh()
    
    # Initialize value arrays - use VhValuesSmall for indexed goals, VhValues for non-indexed
    if False and hasattr(possible_goal_generator, 'indexed') and possible_goal_generator.indexed:
        n_goals = possible_goal_generator.n_goals
        if not quiet:
            print(f"Using indexed goals: VhValuesSmall with numpy arrays (n_goals={n_goals})")
        Vh_values: Union[VhValues, VhValuesSmall] = [
            [np.zeros(n_goals, dtype=np.float16) for _ in range(num_agents)]
            for _ in range(len(states))
        ]
    else:
        Vh_values: Union[VhValues, VhValuesSmall] = [[{} for _ in range(num_agents)] for _ in range(len(states))]
    Vr_values: VrValues = np.zeros(len(states))
    
    # ============================================================================
    # WARN if parallel mode was requested
    # ============================================================================
    if parallel:
        if not quiet:
            print("WARNING: Parallel mode is currently disabled due to bugs.")
            print("         Running in sequential mode instead.")
            print("         See docs/plans/bwind_parallel.md for status.")
    
    # Get sliced attainment cache: prioritize explicit parameter, then world_model cache, then create new
    num_action_profiles = num_actions ** num_agents
    if sliced_cache is None:
        # Try to get cache from world_model (automatically stored by Phase 1)
        sliced_cache = getattr(world_model, '_attainment_cache', None)
        if sliced_cache is not None and isinstance(sliced_cache, SlicedAttainmentCache):
            if not quiet:
                print(f"Using sliced attainment cache from world_model ({sliced_cache.num_states()} state entries)")
        else:
            sliced_cache = None  # Wrong type or not set
    if sliced_cache is None:
        # Create empty sliced cache for Phase 2 internal use
        sliced_cache = SlicedAttainmentCache(num_action_profiles)
    
    # ============================================================================
    # PARALLEL CODE TEMPORARILY DISABLED
    # The parallel implementation is currently broken and needs refactoring.
    # See docs/plans/bwind_parallel.md for the refactoring plan.
    # The code below is kept for reference but will not execute.
    # ============================================================================
    if False:  # Was: if parallel and len(states) > 1:
        # DISABLED: Parallel execution using shared memory via fork
        if disk_dag is not None:
            raise ValueError("Parallel mode does not support disk slicing yet. Use parallel=False with use_disk_slicing=True.")
        
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        if not quiet:
            print(f"Using parallel execution with {num_workers} workers")
        
        # Compute dependency levels and max successor levels for archival
        dependency_levels: List[List[int]]
        max_successor_levels: Optional[Dict[int, int]] = None
        level_values_list: Optional[List[int]] = None
        archived_levels: Set[int] = set()  # Track already-archived levels
        if level_fct is not None:
            if not quiet:
                print("Using fast level computation with provided level function")
            # Pass successors for archival max_successor_levels computation
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
        params: Tuple[List[int], List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float, float, float, float, float, float, float, float] = (
            human_agent_indices, robot_agent_indices, possible_goal_generator, 
            num_agents, num_actions, action_powers, beta_r, gamma_h, gamma_r, 
            zeta, xi, eta, terminal_Vr, rho_h, rho_r
        )
        
        # Use 'fork' context explicitly to ensure shared memory works
        ctx = mp.get_context('fork')
        
        # Initialize shared memory for DAG data to avoid copy-on-write overhead
        if not quiet:
            print("Storing DAG in shared memory...")
        init_shared_dag(states, transitions)
        
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
        
        # Process each level sequentially, but parallelize within each level
        for level_idx, level in enumerate(dependency_levels):
            # Check memory at the start of each level
            if memory_monitor is not None:
                memory_monitor.check(level_idx)
            
            if DEBUG:
                print(f"Processing level {level_idx} with {len(level)} states")
            
            # Generate all possible robot action profiles (needed for sequential fallback)
            robot_action_profiles: List[RobotActionProfile] = [
                tuple(actions) for actions in product(range(num_actions), repeat=len(robot_agent_indices))
            ]
            
            if len(level) <= num_workers:
                # Few states - process sequentially to avoid overhead
                # Create a slice cache for inline processing, but only for non-terminal states
                inline_slice_cache: SliceCache = {
                    state_idx: [{} for _ in range(num_action_profiles)]
                    for state_idx in level
                    if transitions[state_idx]  # Skip terminal states (no transitions)
                }
                
                for state_index in level:
                    state = states[state_index]
                    state_transitions = transitions[state_index]
                    
                    # Use unified helper
                    vh_results, vr_result, p_result, _successor_probs = _rp_process_single_state(
                        state_index, state, states, state_transitions, Vh_values, Vr_values,
                        human_agent_indices, robot_agent_indices, robot_action_profiles,
                        possible_goal_generator, num_agents, num_actions, action_powers,
                        human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
                        slice_cache=inline_slice_cache,
                        rho_h=rho_h,
                        rho_r=rho_r,
                        world_model=world_model,
                    )
                    
                    # Store results
                    for agent_index, agent_vh in vh_results.items():
                        Vh_values[state_index][agent_index].update(agent_vh)
                    
                    Vr_values[state_index] = vr_result
                    
                    if p_result is not None:
                        robot_policy_values[state] = p_result
                
                # Store inline slice cache in sliced_cache (states processed sequentially in parallel mode)
                if inline_slice_cache:
                    inline_slice_id = make_slice_id(list(inline_slice_cache.keys()))
                    sliced_cache.store_slice(inline_slice_id, inline_slice_cache)
            else:
                # Many states - parallelize
                # Re-initialize shared data so new workers see updated values from previous levels
                _rp_init_shared_data(states, transitions, Vh_values, Vr_values, params, human_policy_prior_pickle, use_shared_memory=True, sliced_cache=sliced_cache, num_action_profiles=num_action_profiles, world_model=world_model)
                
                batches = split_into_batches(level, num_workers)
                
                # Create executor per level to ensure workers fork with current values
                with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                    futures = [executor.submit(_rp_process_state_batch, batch) 
                               for batch in batches if batch]
                    
                    batches_completed = 0
                    for future in as_completed(futures):
                        # Check memory BEFORE collecting result to catch pressure early
                        # Use force=True to bypass interval check since we check per-batch
                        if memory_monitor is not None:
                            memory_monitor.check(batches_completed, force=True)
                        
                        vh_results, vr_results, p_results, slice_id, slice_cache, batch_time = future.result()
                        batches_completed += 1
                        
                        # Merge Vh-values back
                        for state_idx, state_results in vh_results.items():
                            for agent_idx, agent_results in state_results.items():
                                Vh_values[state_idx][agent_idx].update(agent_results)
                        
                        # Merge Vr-values back
                        for state_idx, vr_val in vr_results.items():
                            Vr_values[state_idx] = vr_val
                        
                        # Merge robot policies back
                        robot_policy_values.update(p_results)
                        
                        # Check memory AFTER merging results (this is when memory actually increases)
                        if memory_monitor is not None:
                            memory_monitor.check(batches_completed, force=True)
                        
                        # Note: slice_cache is retrieved from Phase 1, no need to store it again
            
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
                    # Archive vhe_values (Vh_values in Phase 2 is expected human achievement)
                    archive_value_slices(
                        Vh_values, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vhe_values.pkl",
                        return_values=return_values,
                        quiet=quiet
                    )
                    # Archive vr_values (robot values) - convert to list structure for archival
                    vr_list = [[Vr_values[i]] for i in range(len(Vr_values))]
                    archive_value_slices(
                        vr_list, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vr_values.pkl",
                        return_values=return_values,
                        quiet=quiet
                    )
                    archived_levels.update(new_archivable)
        
        # Final archival check for parallel mode: archive any remaining levels
        if archive_dir is not None and level_fct is not None:
            all_levels = sorted(set(level_fct(s) for s in states))
            remaining_levels = [lvl for lvl in all_levels if lvl not in archived_levels]
            if remaining_levels:
                # Archive vhe_values (Vh_values in Phase 2 is expected human achievement)
                archive_value_slices(
                    Vh_values, states, level_fct, remaining_levels,
                    filepath=Path(archive_dir) / "vhe_values.pkl",
                    return_values=return_values,
                    quiet=quiet
                )
                # Archive vr_values (robot values) - convert to list structure for archival
                vr_list = [[Vr_values[i]] for i in range(len(Vr_values))]
                archive_value_slices(
                    vr_list, states, level_fct, remaining_levels,
                    filepath=Path(archive_dir) / "vr_values.pkl",
                    return_values=return_values,
                    quiet=quiet
                )
                archived_levels.update(remaining_levels)
        
        # Clean up shared memory after parallel processing
        cleanup_shared_dag()
    
    else:
        # Sequential execution
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
        
        # Allocate markov chain storage if requested
        markov_chain_data: Optional[MarkovChain] = None
        if return_markov_chain:
            markov_chain_data = [{} for _ in range(len(states))]
        
        try:
            _rp_compute_sequential(
                states, Vh_values, Vr_values, robot_policy_values, transitions,
                human_agent_indices, robot_agent_indices, possible_goal_generator,
                num_agents, num_actions, action_powers,
                human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
                progress_callback, memory_monitor,
                sliced_cache,
                level_fct, return_values, archive_dir, disk_dag, quiet, memory_profile,
                markov_chain=markov_chain_data,
                rho_h=rho_h, rho_r=rho_r, world_model=world_model,
            )
        except KeyboardInterrupt:
            if not quiet:
                print("\n[Phase2] Computation interrupted (KeyboardInterrupt).")
                print("         Partial results have been computed.")
            # Re-raise to allow caller to handle
            raise

    robot_policy = TabularRobotPolicy(
        world_model=world_model, 
        robot_agent_indices=robot_agent_indices, 
        values=robot_policy_values
    )
    
    # Print actual memory after Phase 2 backward induction
    if not quiet:
        print(f"\nActual memory usage (after Phase 2 backward induction):")
        print(f"  Process RSS: {get_process_memory_mb():.1f} MB")
        vh_actual = deep_sizeof(Vh_values) / (1024**2)
        vr_actual = deep_sizeof(Vr_values) / (1024**2)
        policy_actual = deep_sizeof(robot_policy_values) / (1024**2)
        cache_actual = deep_sizeof(sliced_cache) / (1024**2) if sliced_cache is not None else 0.0
        print(f"  Vh_values: {vh_actual:.1f} MB")
        print(f"  Vr_values: {vr_actual:.1f} MB")
        print(f"  robot_policy_values: {policy_actual:.1f} MB")
        if sliced_cache is not None:
            print(f"  sliced_cache: {cache_actual:.1f} MB")
        print(f"  Total measured Phase 2 structures: {vh_actual + vr_actual + policy_actual + cache_actual:.1f} MB")
    
    if return_values:
        # Convert Vr_values from array to dict
        Vr_dict = {states[idx]: float(Vr_values[idx]) for idx in range(len(states))}
        
        # Convert Vh_values from list-indexed to state-indexed dict
        Vh_dict = {}
        for state_idx, state in enumerate(states):
            if any(Vh_values[state_idx][agent_idx] for agent_idx in human_agent_indices):
                Vh_dict[state] = {agent_idx: Vh_values[state_idx][agent_idx] 
                                 for agent_idx in human_agent_indices
                                 if Vh_values[state_idx][agent_idx]}
        
        if _pbar is not None:
            _pbar.close()
        if return_markov_chain:
            return robot_policy, Vr_dict, Vh_dict, markov_chain_data
        return robot_policy, Vr_dict, Vh_dict
    
    if _pbar is not None:
        _pbar.close()
    if return_markov_chain:
        return robot_policy, markov_chain_data
    return robot_policy


def compute_markov_chain_value_function(
    markov_chain: MarkovChain,
    rewards: Union[npt.NDArray[np.floating[Any]], Callable[[State], npt.NDArray[np.floating[Any]]]],
    gamma: float,
    states: Optional[List[State]] = None,
) -> npt.NDArray[np.floating[Any]]:
    """Compute the expected discounted vector-valued value function on a Markov chain DAG.
    
    Given a MarkovChain (as returned by compute_robot_policy with
    return_markov_chain=True), a vector-valued reward function, and a discount
    factor, computes V(s) = R(s) + gamma * sum_{s'} P(s'|s) * V(s') for every
    state by backward induction (processing states from last to first, which is
    reverse topological order in the DAG).
    
    Args:
        markov_chain: List of length num_states, where markov_chain[i] is a dict
            mapping successor state_index to transition probability.  Terminal
            states have empty dicts.
        rewards: Either:
            - A 2D ndarray of shape (num_states, reward_dim) where rewards[i] is
              the immediate reward vector for state i, **or**
            - A callable that takes a state (not a state index!) and returns a 1D
              ndarray reward vector.  When a callable is used, the ``states``
              parameter must also be provided.
        gamma: Discount factor in [0, 1].
        states: List of states corresponding to the state indices.  Required when
            ``rewards`` is a callable; ignored when ``rewards`` is an ndarray.
    
    Returns:
        2D ndarray of shape (num_states, reward_dim) where result[i] is the
        expected discounted value vector for state i.
    
    Example:
        >>> policy, mc = compute_robot_policy(..., return_markov_chain=True)
        >>> # Scalar reward as 1-column matrix:
        >>> r = np.array([[1.0], [0.0], [0.5]])  # 3 states, 1 reward dim
        >>> V = compute_markov_chain_value_function(mc, r, gamma=0.9)
        >>> V.shape  # (3, 1)
        >>>
        >>> # Or use a callable:
        >>> def reward_fn(state):
        ...     return np.array([float(state[0] == 'goal'), float(state[1] > 3)])
        >>> V = compute_markov_chain_value_function(mc, reward_fn, gamma=0.9, states=states_list)
    """
    num_states = len(markov_chain)
    
    # Materialise reward matrix if a callable was provided
    if callable(rewards):
        if states is None:
            raise ValueError(
                "states must be provided when rewards is a callable"
            )
        reward_vectors = [rewards(states[i]) for i in range(num_states)]
        reward_matrix = np.stack(reward_vectors, axis=0)  # (num_states, reward_dim)
    else:
        reward_matrix = np.asarray(rewards, dtype=np.float64)
        if reward_matrix.ndim != 2:
            raise ValueError(
                f"rewards array must be 2-dimensional, got {reward_matrix.ndim}D"
            )
        if reward_matrix.shape[0] != num_states:
            raise ValueError(
                f"rewards has {reward_matrix.shape[0]} rows but markov_chain has "
                f"{num_states} states"
            )
    
    reward_dim = reward_matrix.shape[1]
    V = np.zeros((num_states, reward_dim), dtype=np.float64)
    
    # Backward induction: states are in topological order (highest index = latest / terminal)
    for i in range(num_states - 1, -1, -1):
        successors = markov_chain[i]
        if not successors:
            # Terminal state: V(s) = R(s)
            V[i] = reward_matrix[i]
        else:
            # V(s) = R(s) + gamma * sum_{s'} P(s'|s) * V(s')
            future = np.zeros(reward_dim, dtype=np.float64)
            for succ_idx, prob in successors.items():
                future += prob * V[succ_idx]
            V[i] = reward_matrix[i] + gamma * future
    
    return V
