"""
Base class for Gymnasium environments with state management and transition probability computation capabilities.

This module provides a base class that extends gymnasium.Env with methods
for explicit state management (get_state, set_state) and transition probability
computation.
"""

from abc import abstractmethod
from collections import deque
from typing import List, Dict, Tuple, Any, Optional, Set, Union, overload, Literal
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import heapq

import gymnasium as gym
import numpy as np
from tqdm import tqdm

# Type aliases for complex types
State = Any  # State is typically a hashable tuple
ActionProfile = Tuple[int, ...]
TransitionResult = List[Tuple[float, State]]  # List of (probability, successor_state)
TransitionData = Tuple[ActionProfile, List[float], List[State]]  # (action_profile, probs, successor_states)

# Module-level globals for parallel DAG computation
# Each worker creates its own environment instance to avoid shared state issues
_worker_env: Optional['WorldModel'] = None  # Worker-local environment instance
_worker_num_agents: Optional[int] = None
_worker_num_actions_per_agent: Optional[List[int]] = None


def _init_dag_worker(env_pickle: bytes) -> None:
    """
    Initialize worker process for parallel DAG computation.
    Deserializes the environment to create an identical copy in each worker.
    
    This approach preserves ALL attributes of the environment, including
    immutable agent properties like can_enter_magic_walls and can_push_rocks.
    """
    global _worker_env, _worker_num_agents, _worker_num_actions_per_agent
    # Deserialize environment - this creates a complete copy with all attributes
    _worker_env = pickle.loads(env_pickle)
    _worker_num_agents = len(_worker_env.agents)  # type: ignore[union-attr, attr-defined]
    _worker_num_actions_per_agent = _worker_env.num_actions_per_agent  # type: ignore[union-attr]


def _process_state_actions(state: State) -> Tuple[State, Set[State], List[TransitionData]]:
    """
    Process all action combinations for a single state.
    Uses worker's own environment instance.
    
    Args:
        state: The state to process
        
    Returns:
        tuple: (state, successor_states_set, action_data_list)
    """
    global _worker_env, _worker_num_agents, _worker_num_actions_per_agent
    
    assert _worker_env is not None
    assert _worker_num_agents is not None
    assert _worker_num_actions_per_agent is not None
    
    env = _worker_env
    num_agents = _worker_num_agents
    num_actions_per_agent = _worker_num_actions_per_agent
    total_action_combinations = 1
    for n in num_actions_per_agent:
        total_action_combinations *= n
    
    successor_states: Set[State] = set()
    action_data: List[TransitionData] = []
    
    for combo_idx in range(total_action_combinations):
        # Convert combo_idx to action profile
        action_profile: List[int] = []
        temp = combo_idx
        for n in num_actions_per_agent:
            action_profile.append(temp % n)
            temp //= n
        action_profile_tuple = tuple(action_profile)
        
        # Get transitions for this action (this calls set_state internally)
        trans_result = env.transition_probabilities(state, action_profile)
        
        if trans_result is None:
            continue
        
        # Collect successor states and probabilities
        probs: List[float] = []
        succ_states: List[State] = []
        for prob, successor_state in trans_result:
            probs.append(prob)
            succ_states.append(successor_state)
            successor_states.add(successor_state)
        
        action_data.append((action_profile_tuple, probs, succ_states))
    
    return state, successor_states, action_data


class WorldModel(gym.Env):
    """
    Base class for Gymnasium environments with explicit state management.
    
    This class extends gymnasium.Env to provide:
    1. get_state(): Get a hashable representation of the complete environment state
    2. set_state(): Restore the environment to a specific state
    3. transition_probabilities(): Compute exact transition probabilities for actions
    
    These methods enable planning algorithms and empowerment computation that require
    explicit enumeration of state spaces and transition dynamics.
    
    Subclasses must implement the abstract methods to provide environment-specific
    state serialization and transition logic.
    """
    
    @abstractmethod
    def get_state(self) -> Any:
        """
        Get the complete state of the environment.
        
        Returns a hashable representation containing everything needed to predict
        the consequences of possible actions. The state must be sufficient to
        restore the environment to an identical configuration via set_state().
        
        Returns:
            A hashable representation of the complete environment state.
            Must be usable as a dictionary key.
        """
        raise NotImplementedError("Subclasses must implement get_state()")
    
    @abstractmethod
    def set_state(self, state: Any) -> None:
        """
        Set the environment to a specific state.
        
        Restores the environment to the exact configuration represented by the
        given state. After calling set_state(s), get_state() should return a
        state equivalent to s.
        
        Args:
            state: A state as returned by get_state()
        """
        raise NotImplementedError("Subclasses must implement set_state()")
    
    @abstractmethod
    def transition_probabilities(
        self, 
        state: Any, 
        actions: List[int]
    ) -> Optional[List[Tuple[float, Any]]]:
        """
        Given a state and vector of actions, return possible transitions with exact probabilities.
        
        This method computes all possible successor states and their probabilities
        given the current state and actions. It is essential for planning algorithms
        that require explicit enumeration of transition dynamics.
        
        Args:
            state: A state tuple as returned by get_state()
            actions: List of action indices, one per agent (for multi-agent envs)
                    or a single action (for single-agent envs)
            
        Returns:
            list: List of (probability, successor_state) tuples describing all
                  possible transitions. Probabilities should sum to 1.0.
                  Returns None if the state is terminal or if actions are invalid.
        """
        raise NotImplementedError("Subclasses must implement transition_probabilities()")
    
    def transition_durations(
        self,
        state: Any,
        actions: List[int],
        transitions: List[Tuple[float, Any]]
    ) -> List[float]:
        """
        Return the expected duration for each transition outcome.
        
        Given a state, an action profile, and the list of (probability, successor_state)
        outcomes from transition_probabilities(), return a list of durations D(s, a, s')
        of the same length.
        
        The duration D(s, a, s') represents the certainty-equivalent expected real-time
        elapsed between taking the action and regaining control in successor state s'.
        For deterministic durations, this reduces to the actual elapsed time.
        
        The default implementation returns [1.0, ...] (unit duration for every transition),
        preserving backward compatibility with the existing uniform-step assumption.
        
        Args:
            state: The current state.
            actions: The joint action profile.
            transitions: The list of (probability, successor_state) tuples as returned
                by transition_probabilities().
        
        Returns:
            List of floats, one per transition outcome. Must have len == len(transitions).
        """
        return [1.0] * len(transitions)
    
    def terminal_duration(self, state: Any) -> float:
        """
        Return the expected duration of a terminal state before the episode ends.
        
        Used for terminal-state power aggregation: D(s) for terminal s.
        Default returns 1.0.
        
        Args:
            state: A terminal state.
        
        Returns:
            Duration as a float.
        """
        return 1.0

    def V_r_estimate(self, state: Any) -> float:
        """
        Return an external estimate of the robot's value (returns-to-go) for a
        terminal state.

        The default implementation returns 0.0.  Subclasses that have access to
        extrinsic value estimates (e.g. from an LLM) should override this.

        Args:
            state: A terminal state as returned by get_state().

        Returns:
            Estimated value (returns-to-go) for the terminal state.
        """
        return 0.0
    
    def initial_state(self) -> Any:
        """
        Get the initial state of the environment without permanently resetting it.
        
        This method:
        1. Saves the current state
        2. Calls reset() to get the initial state
        3. Returns the initial state
        4. Restores the environment to its previous state
        
        This is useful for algorithms that need to know the initial state
        without losing the current environment configuration.
        
        Returns:
            The initial state of the environment (as returned by get_state() after reset())
        """
        # Save current state
        saved_state = self.get_state()
        
        # Reset to get initial state
        self.reset()
        init_state = self.get_state()
        
        # Restore previous state
        self.set_state(saved_state)
        
        return init_state

    # DAG cache: stores computed DAG to avoid redundant computation
    # Key: return_probabilities (bool), Value: the DAG tuple
    _dag_cache: Optional[Dict[bool, tuple]] = None
    
    def clear_dag_cache(self) -> None:
        """
        Clear the cached DAG.
        
        Call this method if the environment's transition dynamics have changed
        (e.g., after modifying max_steps or other parameters that affect the DAG).
        The DAG will be recomputed on the next call to get_dag().
        """
        self._dag_cache = None
    
    @property
    def human_agent_indices(self) -> List[int]:
        """
        Get the indices of human agents in the environment.
        
        Returns a list of agent indices that represent human agents.
        The indices correspond to positions in the action list passed to step().
        
        Subclasses should override this property to identify human agents based on
        environment-specific criteria (e.g., color, type, name).
        
        Returns:
            List of indices for human agents.
        
        Raises:
            NotImplementedError: If the subclass doesn't implement this property.
        """
        raise NotImplementedError(
            "Subclasses must implement human_agent_indices property to identify human agents. "
            "This is required for Phase 2 training."
        )
    
    @property
    def robot_agent_indices(self) -> List[int]:
        """
        Get the indices of robot agents in the environment.
        
        Returns a list of agent indices that represent robot agents.
        The indices correspond to positions in the action list passed to step().
        
        Subclasses should override this property to identify robot agents based on
        environment-specific criteria (e.g., color, type, name).
        
        Returns:
            List of indices for robot agents.
        
        Raises:
            NotImplementedError: If the subclass doesn't implement this property.
        """
        raise NotImplementedError(
            "Subclasses must implement robot_agent_indices property to identify robot agents. "
            "This is required for Phase 2 training."
        )
    
    @property
    def num_actions_per_agent(self) -> List[int]:
        """
        Get the number of actions available to each agent.
        
        Returns a list where element i is the number of actions for agent i.
        The default implementation returns [action_space.n] for all agents
        (uniform action counts). Subclasses can override this to support
        agents with different numbers of actions.
        
        Returns:
            List of action counts, one per agent.
        """
        n: int = self.action_space.n  # type: ignore[attr-defined]
        num_agents: int = len(self.agents)  # type: ignore[attr-defined]
        return [n] * num_agents
    
    def is_terminal(self, state: Optional[Any] = None) -> bool:
        """
        Check if a state is terminal (no valid transitions exist).
        
        A state is considered terminal if transition_probabilities() returns None
        for all possible actions, indicating that no further transitions are possible.
        
        Args:
            state: The state to check. If None, checks the current state.
        
        Returns:
            True if the state is terminal, False otherwise.
        """
        # Get the state to check
        if state is None:
            state = self.get_state()
        
        # Try a simple action (e.g., all zeros) to check if transitions are possible
        # For multi-agent environments, we need to know the number of agents
        if hasattr(self, 'agents'):
            num_agents = len(self.agents)
        else:
            num_agents = 1
        
        # Use action 0 for all agents as a test action
        test_actions = [0] * num_agents
        
        # Check if transition_probabilities returns None (terminal) or a list (non-terminal)
        transitions = self.transition_probabilities(state, test_actions)
        
        return transitions is None
    
    def _get_construction_args(self) -> tuple:
        """
        Get positional arguments needed to reconstruct this environment.
        
        Subclasses should override this to return the positional arguments
        passed to __init__(). This is used for parallel DAG computation where
        workers need to create fresh environment instances.
        
        Returns:
            tuple: Positional arguments for __init__()
        """
        return ()
    
    def _get_construction_kwargs(self) -> dict:
        """
        Get keyword arguments needed to reconstruct this environment.
        
        Subclasses should override this to return the keyword arguments
        passed to __init__(). This is used for parallel DAG computation where
        workers need to create fresh environment instances.
        
        Returns:
            dict: Keyword arguments for __init__()
        """
        return {}
    
    def step(self, actions: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment using transition probabilities.
        
        This default implementation:
        1. Gets the current state
        2. Calls transition_probabilities() to get possible transitions
        3. Samples from those transitions based on probabilities
        4. Calls set_state() with the sampled successor state
        5. Returns (observation, reward, terminated, truncated, info)
        
        Subclasses may override this for more efficient implementations or
        to add custom logic (e.g., rendering, reward computation).
        
        Args:
            actions: Action(s) to take. For multi-agent environments, this should
                    be a list of actions, one per agent.
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: The new state (as returned by get_state())
                - reward: 0.0 (subclasses should override for actual rewards)
                - terminated: True if the new state is terminal
                - truncated: False (subclasses should override for truncation logic)
                - info: Empty dict (subclasses can add additional info)
        """
        # Ensure actions is a list for multi-agent compatibility
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
        
        # Get current state
        current_state = self.get_state()
        
        # Get transition probabilities
        transitions = self.transition_probabilities(current_state, list(actions))
        
        # If terminal state (no transitions), return current state
        if transitions is None:
            return current_state, 0.0, True, False, {}
        
        # Sample from transitions based on probabilities
        probabilities = [prob for prob, _ in transitions]
        successor_states = [state for _, state in transitions]
        
        # Use numpy to sample based on probabilities
        chosen_idx = np.random.choice(len(transitions), p=probabilities)
        new_state = successor_states[chosen_idx]
        
        # Set the environment to the new state
        self.set_state(new_state)
        
        # Check if new state is terminal
        terminated = self.is_terminal(new_state)
        
        # Return observation (the new state), reward, terminated, truncated, info
        # Subclasses should override to provide actual rewards
        return new_state, 0.0, terminated, False, {}
    
    @overload
    def get_dag(
        self, return_probabilities: Literal[False] = False, quiet: bool = False
    ) -> Tuple[List[State], Dict[State, int], List[List[int]]]: ...
    
    @overload
    def get_dag(
        self, return_probabilities: Literal[True], quiet: bool = False
    ) -> Tuple[List[State], Dict[State, int], List[List[int]], List[List[Tuple[ActionProfile, List[float], List[int]]]]]: ...
    
    def get_dag(
        self, return_probabilities: bool = False, quiet: bool = False
    ) -> Union[
        Tuple[List[State], Dict[State, int], List[List[int]]],
        Tuple[List[State], Dict[State, int], List[List[int]], List[List[Tuple[ActionProfile, List[float], List[int]]]]]
    ]:
        """
        Efficiently compute the DAG structure of an acyclic finite environment.
        
        This method uses a two-phase algorithm:
        1. BFS to discover all reachable states and edges
        2. Topological sort (Kahn's algorithm) to order states correctly
        
        This ensures that successor states always come after their predecessors,
        even when a state is reachable via multiple paths of different lengths.
        
        Results are cached and reused on subsequent calls. Use clear_dag_cache()
        to invalidate the cache if the environment's dynamics have changed.
        
        Time Complexity: O(|S| + |T|) where |S| is the number of states and |T| is the
        total number of transitions. Phase 1 visits each state once and examines each
        transition once. Phase 2 (topological sort) also runs in O(|S| + |T|).
        
        Space Complexity: O(|S| + |T|) for storing states, edges, and auxiliary structures.
        
        Args:
            return_probabilities: If True, also return transition probabilities as a
                fourth item. Each element is a list of (action, probs, succ_indices)
                tuples for that state, where action is the action tuple, probs is a
                list of transition probabilities, and succ_indices is a list of
                successor state indices (parallel lists).
            quiet: If True, suppress the progress bar. Default is False (show progress).
        
        Returns:
            A tuple containing:
            1. states (List): List of all reachable states in topological order
               (successor states always come after predecessor states)
            2. state_to_idx (Dict): Dictionary mapping each state to its index in
               the states list
            3. successors (List[List[int]]): List where successors[i] contains the
               indices of all possible successor states of states[i]
            4. (optional) transitions (List[List[Tuple]]): If return_probabilities=True,
               a list where transitions[i] contains (action, probs, succ_indices) tuples
               for each action combination from states[i]
        
        Example:
            >>> states, state_to_idx, successors = env.get_dag()
            >>> # states[0] is the root state
            >>> # state_to_idx[states[i]] == i
            >>> # successors[i] are indices of states reachable from states[i]
            >>> # For any edge i -> j in the DAG: i < j (topological ordering)
            
            >>> # With probabilities:
            >>> states, state_to_idx, successors, transitions = env.get_dag(return_probabilities=True)
            >>> # transitions[i] = [(action, probs, succ_indices), ...]
        """
        # Check cache first
        if self._dag_cache is None:
            self._dag_cache = {}
        
        # If we have cached result with probabilities, we can derive the non-prob version
        if return_probabilities:
            if True in self._dag_cache:
                if not quiet:
                    print("Using cached DAG (with probabilities)")
                return self._dag_cache[True]
        else:
            # Can use either cached version for non-prob request
            if False in self._dag_cache:
                if not quiet:
                    print("Using cached DAG")
                return self._dag_cache[False]
            if True in self._dag_cache:
                # Extract non-prob version from prob version
                if not quiet:
                    print("Using cached DAG (extracting from full version)")
                states, state_to_idx, successors, _ = self._dag_cache[True]
                return states, state_to_idx, successors
        
        # Reset environment to get root state
        self.reset()
        root_state = self.get_state()

        # Ensure _dag_cache is available (reset() may have cleared it)
        if self._dag_cache is None:
            self._dag_cache = {}
        
        # PHASE 1: Discover all states and edges using BFS
        discovered_states: List[State] = []  # Temporary list during discovery
        temp_state_to_idx: Dict[State, int] = {}  # Temporary mapping
        edges: Dict[int, Set[int]] = {}  # edges[i] = set of successor indices (temporary)
        
        # Store transition probabilities if requested
        temp_transitions: Optional[Dict[int, List[Tuple[ActionProfile, List[float], List[State]]]]] = {} if return_probabilities else None
        
        queue: deque[State] = deque([root_state])
        temp_state_to_idx[root_state] = 0
        discovered_states.append(root_state)
        edges[0] = set()
        if return_probabilities and temp_transitions is not None:
            temp_transitions[0] = []
        
        # Get action space info once before the loop
        num_agents: int = len(self.agents)  # type: ignore[attr-defined]
        num_actions_per_agent: List[int] = self.num_actions_per_agent
        total_combinations = 1
        for n in num_actions_per_agent:
            total_combinations *= n
        
        # Set up progress bar
        pbar: Optional[tqdm[int]] = None
        if not quiet:
            pbar = tqdm(desc="Building DAG", unit=" states", leave=False)
        
        states_processed = 0
        while queue:
            current_state = queue.popleft()
            current_idx = temp_state_to_idx[current_state]
            
            # Extract timestep from state (state[0] for most environments like MultiGrid)
            # Handle states that might not have timestep as first element
            try:
                current_timestep = current_state[0] if isinstance(current_state, (tuple, list)) and len(current_state) > 0 else None
            except (TypeError, IndexError):
                current_timestep = None
            
            # Track unique successor states for this current state
            seen_successors: Set[State] = set()
            
            for combo_idx in range(total_combinations):
                # Convert combo_idx to action profile tuple
                action_profile: List[int] = []
                temp = combo_idx
                for n in num_actions_per_agent:
                    action_profile.append(temp % n)
                    temp //= n
                action_profile_tuple = tuple(action_profile)
                
                # Get transition probabilities for this action combination
                trans_result = self.transition_probabilities(current_state, action_profile)
                
                # If None, state is terminal or actions are invalid
                if trans_result is None:
                    continue
                
                # If storing probabilities, collect them
                if return_probabilities and temp_transitions is not None:
                    probs: List[float] = []
                    succ_states: List[State] = []
                    for prob, successor_state in trans_result:
                        probs.append(prob)
                        succ_states.append(successor_state)
                    temp_transitions[current_idx].append((action_profile_tuple, probs, succ_states))
                
                # Process all successor states from these transitions
                for prob, successor_state in trans_result:
                    # Skip if we've already seen this successor from current state
                    if successor_state in seen_successors:
                        continue
                    
                    seen_successors.add(successor_state)
                    
                    # If this is a new state, add it to discovery list
                    if successor_state not in temp_state_to_idx:
                        temp_state_to_idx[successor_state] = len(discovered_states)
                        discovered_states.append(successor_state)
                        edges[len(discovered_states) - 1] = set()
                        if return_probabilities and temp_transitions is not None:
                            temp_transitions[len(discovered_states) - 1] = []
                        queue.append(successor_state)
                    
                    # Add edge from current to successor
                    successor_idx = temp_state_to_idx[successor_state]
                    edges[current_idx].add(successor_idx)
            
            # Update progress bar
            states_processed += 1
            if pbar is not None:
                pbar.n = states_processed
                # Update description to show current timestep if available
                if current_timestep is not None:
                    pbar.set_description(f"Building DAG (t={current_timestep})")
                pbar.refresh()
        
        # Close progress bar
        if pbar is not None:
            pbar.close()
        
        # PHASE 2: Topological sort using Kahn's algorithm with deterministic ordering
        # We use a heap sorted by state tuples to ensure identical ordering regardless
        # of discovery order (which differs between sequential and parallel versions)
        num_states = len(discovered_states)
        
        # Compute in-degree for each state
        in_degree: List[int] = [0] * num_states
        for state_idx in range(num_states):
            for successor_idx in edges[state_idx]:
                in_degree[successor_idx] += 1
        
        # Initialize heap with all states that have in-degree 0, sorted by state tuple
        # We use (state_tuple, idx) as heap entries for deterministic ordering
        topo_heap: List[Tuple[State, int]] = []
        for i in range(num_states):
            if in_degree[i] == 0:
                heapq.heappush(topo_heap, (discovered_states[i], i))
        
        # Topologically sorted order (indices in discovered_states)
        topo_order: List[int] = []
        
        while topo_heap:
            _, current_idx = heapq.heappop(topo_heap)
            topo_order.append(current_idx)
            
            # Reduce in-degree of successors
            for successor_idx in edges[current_idx]:
                in_degree[successor_idx] -= 1
                if in_degree[successor_idx] == 0:
                    heapq.heappush(topo_heap, (discovered_states[successor_idx], successor_idx))
        
        # Verify we got all states (no cycles)
        if len(topo_order) != num_states:
            raise ValueError("Environment contains cycles (not a DAG)")
        
        # PHASE 3: Build final output with new indices
        states: List[State] = []
        state_to_idx: Dict[State, int] = {}
        old_to_new: Dict[int, int] = {}  # Map old indices to new indices
        
        for new_idx, old_idx in enumerate(topo_order):
            state = discovered_states[old_idx]
            states.append(state)
            state_to_idx[state] = new_idx
            old_to_new[old_idx] = new_idx
        
        # Build successors list with new indices
        successors: List[List[int]] = [[] for _ in range(num_states)]
        for old_idx in range(num_states):
            new_idx = old_to_new[old_idx]
            for old_succ_idx in edges[old_idx]:
                new_succ_idx = old_to_new[old_succ_idx]
                successors[new_idx].append(new_succ_idx)
        
        if not return_probabilities:
            # Cache and return
            result = (states, state_to_idx, successors)
            self._dag_cache[False] = result
            return result
        
        # PHASE 4: Convert transitions to use new indices
        assert temp_transitions is not None
        transitions: List[List[Tuple[ActionProfile, List[float], List[int]]]] = [[] for _ in range(num_states)]
        for old_idx in range(num_states):
            new_idx = old_to_new[old_idx]
            for action_prof, trans_probs, trans_succ_states in temp_transitions[old_idx]:
                # Convert successor states to new indices
                succ_indices = [state_to_idx[s] for s in trans_succ_states]
                transitions[new_idx].append((action_prof, trans_probs, succ_indices))
        
        # Cache and return
        result_with_prob = (states, state_to_idx, successors, transitions)
        self._dag_cache[True] = result_with_prob
        return result_with_prob
    
    @overload
    def get_dag_parallel(
        self, return_probabilities: Literal[False] = False, num_workers: Optional[int] = None
    ) -> Tuple[List[State], Dict[State, int], List[List[int]]]: ...
    
    @overload
    def get_dag_parallel(
        self, return_probabilities: Literal[True], num_workers: Optional[int] = None
    ) -> Tuple[List[State], Dict[State, int], List[List[int]], List[List[Tuple[ActionProfile, List[float], List[int]]]]]: ...
    
    def get_dag_parallel(
        self, return_probabilities: bool = False, num_workers: Optional[int] = None
    ) -> Union[
        Tuple[List[State], Dict[State, int], List[List[int]]],
        Tuple[List[State], Dict[State, int], List[List[int]], List[List[Tuple[ActionProfile, List[float], List[int]]]]]
    ]:
        """
        Parallel version of get_dag() using spawn-based multiprocessing.
        
        Each worker creates its own fresh environment instance to avoid
        shared state issues when calling transition_probabilities().
        Parallelizes BFS exploration by processing multiple states in parallel.
        
        Args:
            return_probabilities: If True, also return transition probabilities.
            num_workers: Number of worker processes. If None, uses CPU count.
        
        Returns:
            Same as get_dag(): (states, state_to_idx, successors) or 
            (states, state_to_idx, successors, transitions) if return_probabilities=True
        """
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        # Serialize the entire environment for workers
        # This preserves ALL attributes including immutable agent properties
        # like can_enter_magic_walls and can_push_rocks
        self.reset()  # Ensure consistent initial state
        env_pickle = pickle.dumps(self)
        
        # Use spawn context - each worker gets its own deserialized copy
        ctx = mp.get_context('spawn')
        
        # BFS to discover all states
        initial_state = self.get_state()
        visited: Set[State] = {initial_state}
        edges: Dict[State, List[State]] = {}  # state -> list of successor states
        temp_transitions: Dict[State, List[TransitionData]] = {}  # state -> list of (action_profile, probs, succ_states)
        
        # Process states in waves for parallelization
        current_wave: List[State] = [initial_state]
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx,
                                 initializer=_init_dag_worker,
                                 initargs=(env_pickle,)) as executor:
            while current_wave:
                # Process all states in current wave in parallel
                futures = {executor.submit(_process_state_actions, state): state 
                           for state in current_wave}
                
                next_wave: List[State] = []
                
                for future in futures:
                    state = futures[future]
                    try:
                        _, successor_states, action_data = future.result()
                    except Exception as e:
                        import sys, traceback
                        print(f"Worker failed processing state: {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        continue
                    
                    edges[state] = list(successor_states)
                    temp_transitions[state] = action_data
                    
                    # Add new states to next wave
                    for succ_state in successor_states:
                        if succ_state not in visited:
                            visited.add(succ_state)
                            next_wave.append(succ_state)
                
                current_wave = next_wave
        
        # PHASE 2: Topological sort using Kahn's algorithm with deterministic ordering
        # We use a heap sorted by state tuples to ensure identical ordering regardless
        # of discovery order (which differs between sequential and parallel versions)
        
        # First, build edges using indices
        discovered_states: List[State] = list(visited)
        num_states = len(discovered_states)
        temp_state_to_idx: Dict[State, int] = {state: idx for idx, state in enumerate(discovered_states)}
        
        # Build edge list with indices
        indexed_edges: List[Set[int]] = [set() for _ in range(num_states)]
        for state in edges:
            state_idx = temp_state_to_idx[state]
            for succ_state in edges[state]:
                succ_idx = temp_state_to_idx[succ_state]
                indexed_edges[state_idx].add(succ_idx)
        
        # Compute in-degree for each state
        in_degree: List[int] = [0] * num_states
        for state_idx in range(num_states):
            for successor_idx in indexed_edges[state_idx]:
                in_degree[successor_idx] += 1
        
        # Initialize heap with all states that have in-degree 0, sorted by state tuple
        # We use (state_tuple, idx) as heap entries for deterministic ordering
        topo_heap: List[Tuple[State, int]] = []
        for i in range(num_states):
            if in_degree[i] == 0:
                heapq.heappush(topo_heap, (discovered_states[i], i))
        
        # Topologically sorted order
        topo_order: List[int] = []
        
        while topo_heap:
            _, current_idx = heapq.heappop(topo_heap)
            topo_order.append(current_idx)
            
            # Reduce in-degree of successors
            for successor_idx in indexed_edges[current_idx]:
                in_degree[successor_idx] -= 1
                if in_degree[successor_idx] == 0:
                    heapq.heappush(topo_heap, (discovered_states[successor_idx], successor_idx))
        
        # Verify we got all states (no cycles)
        if len(topo_order) != num_states:
            raise ValueError("Environment contains cycles (not a DAG)")
        
        # Build final output with new indices
        states: List[State] = []
        state_to_idx: Dict[State, int] = {}
        old_to_new: Dict[int, int] = {}  # Map old indices to new indices
        
        for new_idx, old_idx in enumerate(topo_order):
            state = discovered_states[old_idx]
            states.append(state)
            state_to_idx[state] = new_idx
            old_to_new[old_idx] = new_idx
        
        # Build successors list with new indices
        successors: List[List[int]] = [[] for _ in range(num_states)]
        for old_idx in range(num_states):
            new_idx = old_to_new[old_idx]
            for old_succ_idx in indexed_edges[old_idx]:
                new_succ_idx = old_to_new[old_succ_idx]
                successors[new_idx].append(new_succ_idx)
        
        if not return_probabilities:
            return states, state_to_idx, successors
        
        # PHASE 3: Convert transitions to use new indices
        transitions: List[List[Tuple[ActionProfile, List[float], List[int]]]] = [[] for _ in range(num_states)]
        for state in temp_transitions:
            new_idx = state_to_idx[state]
            for action_profile, probs, succ_states in temp_transitions[state]:
                succ_indices = [state_to_idx[s] for s in succ_states]
                transitions[new_idx].append((action_profile, probs, succ_indices))
        
        return states, state_to_idx, successors, transitions
    
    def plot_dag(
        self,
        states: Optional[List[Any]] = None,
        state_to_idx: Optional[Dict[Any, int]] = None,
        successors: Optional[List[List[int]]] = None,
        output_file: Optional[str] = "dag",
        format: str = "png",
        state_labels: Optional[Dict[Any, str]] = None,
        rankdir: str = "TB"
    ) -> str:
        """
        Plot the DAG structure using Graphviz.
        
        If states, state_to_idx, and successors are not provided, they will be
        computed by calling get_dag().
        
        Args:
            states: List of states in topological order (from get_dag)
            state_to_idx: Dictionary mapping states to indices (from get_dag)
            successors: List of successor indices for each state (from get_dag)
            output_file: Output filename without extension (default: "dag")
            format: Output format - 'png', 'pdf', 'svg', etc. (default: "png")
            state_labels: Optional dict mapping states to custom labels for display
            rankdir: Graph direction - 'TB' (top-bottom), 'LR' (left-right), etc.
        
        Returns:
            Path to the generated image file
        
        Raises:
            ImportError: If graphviz is not installed
        
        Example:
            >>> env.plot_dag(output_file="my_dag", format="pdf")
            'my_dag.pdf'
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz is required for plotting. Install with: pip install graphviz"
            )
        
        # Compute DAG if not provided
        if states is None or state_to_idx is None or successors is None:
            states, state_to_idx, successors = self.get_dag()
        
        # Create directed graph
        dot = graphviz.Digraph(comment='State Space DAG')
        dot.attr(rankdir=rankdir)
        dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')
        
        # Add nodes
        for idx, state in enumerate(states):
            # Determine node label
            if state_labels and state in state_labels:
                label = state_labels[state]
            else:
                label = str(state)
            
            # Highlight root node
            if idx == 0:
                dot.node(str(idx), label, fillcolor='lightgreen', style='filled')
            # Highlight terminal nodes (no successors)
            elif len(successors[idx]) == 0:
                dot.node(str(idx), label, fillcolor='lightcoral', style='filled')
            else:
                dot.node(str(idx), label)
        
        # Add edges
        for idx, succ_list in enumerate(successors):
            for succ_idx in succ_list:
                dot.edge(str(idx), str(succ_idx))
        
        # Render to file
        output_path = dot.render(output_file, format=format, cleanup=True)
        
        return output_path
