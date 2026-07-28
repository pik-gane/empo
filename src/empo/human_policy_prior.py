"""
Human Policy Prior for Modeling Human Behavior.

This module provides abstract and concrete implementations for modeling
human behavior as goal-directed policies. The policy prior represents
our belief about what action a human agent would take in a given state,
optionally conditioned on a specific goal.

Classes:
    HumanPolicyPrior: Abstract base class for human policy priors.
    TabularHumanPolicyPrior: Concrete implementation using precomputed lookup tables.

The policy prior is central to the EMPO framework where robots reason about
human behavior to compute empowerment and select helpful actions.

Key concepts:
    - A policy prior maps (state, agent, goal) -> action distribution
    - When called without a goal, returns marginal over all possible goals
    - The sample() method enables Monte Carlo simulation of human behavior

Example usage:
    >>> # Using a precomputed policy prior
    >>> from empo.backward_induction import compute_human_policy_prior
    >>> policy_prior = compute_human_policy_prior(env, [0], goal_generator)
    >>> 
    >>> # Get action distribution for agent 0 with specific goal
    >>> action_dist = policy_prior(state, 0, my_goal)  # numpy array
    >>> 
    >>> # Sample an action
    >>> action = policy_prior.sample(state, 0, my_goal)  # int
    >>> 
    >>> # Get marginal action distribution (averaging over goals)
    >>> marginal_dist = policy_prior(state, 0)  # numpy array
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from empo.world_model import WorldModel
    from empo.possible_goal import PossibleGoal, PossibleGoalGenerator


class HumanPolicyPrior(ABC):
    """
    Abstract base class for human policy priors.
    
    A human policy prior is a callable that returns a probability distribution
    over actions for a human agent, optionally conditioned on a possible goal.
    
    When called with a goal, returns P(action | state, agent, goal).
    When called without a goal, returns the marginal P(action | state, agent)
    obtained by averaging over possible goals weighted by their prior probabilities.
    
    Attributes:
        world_model: The world model (environment) this prior applies to.
        human_agent_indices: List of agent indices considered as "human" agents.
    
    Note:
        This class assumes independence between human agents when sampling
        joint actions without conditioning on specific goals.
    """

    world_model: 'WorldModel'
    human_agent_indices: List[int]
    
    def __init__(self, world_model: 'WorldModel', human_agent_indices: List[int]):
        """
        Initialize the human policy prior.
        
        Args:
            world_model: The world model (environment) this prior applies to.
            human_agent_indices: List of indices of agents to model as humans.
        """
        self.world_model = world_model
        self.human_agent_indices = human_agent_indices
    
    def set_world_model(self, world_model: 'WorldModel') -> None:
        """
        Set or update the world model reference.
        
        This is used for async training where the world_model cannot be pickled
        and must be recreated in child processes.
        
        Args:
            world_model: The world model (environment) to use.
        """
        self.world_model = world_model
    
    def __getstate__(self):
        """Exclude world_model from pickling (it contains thread locks)."""
        state = self.__dict__.copy()
        state['world_model'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling (world_model will be set later)."""
        self.__dict__.update(state)

    @abstractmethod
    def __call__(
        self, 
        state, 
        human_agent_index: int, 
        possible_goal: Optional['PossibleGoal'] = None
    ) -> np.ndarray:
        """
        Get the action distribution for a human agent.
        
        Args:
            state: Current world state (hashable tuple from get_state()).
            human_agent_index: Index of the human agent.
            possible_goal: If provided, condition the distribution on this goal.
                          If None, return marginal distribution over goals.
        
        Returns:
            np.ndarray: Probability distribution over actions (sums to 1.0).
                       Shape is (num_actions,) where num_actions = action_space.n.
        """

    @staticmethod
    def _to_probability_array(action_distribution) -> np.ndarray:
        """
        Convert action distribution to numpy array.
        
        Handles both dict (from neural policy priors) and array returns.
        For dict input, assumes keys are consecutive integers starting from 0.
        Ensures the probabilities sum to 1.0.
        
        Args:
            action_distribution: Either a dict mapping action index to probability,
                               or a numpy array of probabilities.
        
        Returns:
            np.ndarray: Probability array indexed by action, normalized to sum to 1.0.
        """
        if isinstance(action_distribution, dict):
            num_actions = len(action_distribution)
            probs = np.array([action_distribution[i] for i in range(num_actions)])
        else:
            probs = action_distribution
#            probs = np.asarray(action_distribution)
        
        # Ensure probabilities sum to 1.0 (handle floating-point errors and edge cases)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            # If all probabilities are zero, use uniform distribution
            probs = np.ones_like(probs) / len(probs)
        
        return probs

    def sample(
        self, 
        state, 
        human_agent_index: Optional[int] = None, 
        possible_goal: Optional['PossibleGoal'] = None
    ) -> Union[int, List[int]]:
        """
        Sample action(s) from the policy prior.
        
        If human_agent_index is provided, samples a single action for that agent.
        If human_agent_index is None, samples actions for ALL human agents (assuming
        independence and marginalizing over goals for each).
        
        Args:
            state: Current world state.
            human_agent_index: If provided, sample for this specific agent.
                              If None, sample for all human agents.
            possible_goal: If provided, condition on this goal.
                          Only valid when human_agent_index is also provided.
        
        Returns:
            int: If human_agent_index is provided, the sampled action index.
            List[int]: If human_agent_index is None, list of sampled actions
                      for all human agents (in order of human_agent_indices).
        
        Raises:
            AssertionError: If possible_goal is provided without human_agent_index.
        """
        if human_agent_index is not None:
            action_distribution = self(state, human_agent_index, possible_goal)
            probs = self._to_probability_array(action_distribution)
            return int(np.random.choice(len(probs), p=probs))
        else:
            assert possible_goal is None, \
                "When sampling actions for all human agents, no possible goal can be given."
            actions = []
            for agent_index in self.human_agent_indices:
                action_distribution = self(state, agent_index)
                probs = self._to_probability_array(action_distribution)
                action = int(np.random.choice(len(probs), p=probs))
                actions.append(action)
            return actions

    def profile_distribution(
        self, 
        state,
        device: Optional[str] = None
    ) -> List[tuple]:
        """
        Get the joint action profile distribution for all human agents. 
        Assumes independence between agents.
        Args:
            state: Current world state.
            device: Optional computation backend:
                   - None or 'cpu': NumPy (default, good for small/medium problems)
                   - 'cuda' or 'cuda:N': PyTorch GPU (best for large problems)
        Returns:
            List of tuples (probability, action_profile) where action_profile is
            a list of actions for each human agent in order of human_agent_indices.
        """
        # Pre-compute marginal distributions for all agents
        marginals = [
            self._to_probability_array(self(state, agent_index))
            for agent_index in self.human_agent_indices
        ]
        
        if not marginals:
            return [(1.0, [])]
        
        if device is not None and device.startswith('cuda'):
            # Use PyTorch for GPU computation
            return self._profile_distribution_torch(marginals, device)
        else:
            # Use NumPy - fastest for CPU
            return self._profile_distribution_numpy(marginals)

    def profile_distribution_with_fixed_goal(
        self, 
        state,
        fixed_agent_index: int,
        fixed_goal: 'PossibleGoal',
        device: Optional[str] = None
    ) -> List[tuple]:
        """
        Get the joint action profile distribution where one agent has a fixed goal.
        
        For the specified agent, uses the goal-conditioned policy for the fixed goal.
        For all other human agents, uses their marginal policy (averaged over their goals).
        Assumes independence between agents.
        
        Args:
            state: Current world state.
            fixed_agent_index: The agent index whose goal is fixed.
            fixed_goal: The specific goal for the fixed agent.
            device: Optional computation backend (same as profile_distribution).
        
        Returns:
            List of tuples (probability, action_profile) where action_profile is
            a list of actions for each human agent in order of human_agent_indices.
        """
        # Pre-compute distributions for all agents
        # For the fixed agent, use goal-conditioned policy; for others, use marginal
        marginals = []
        for agent_index in self.human_agent_indices:
            if agent_index == fixed_agent_index:
                # Use goal-specific policy for this agent
                marginals.append(self._to_probability_array(self(state, agent_index, fixed_goal)))
            else:
                # Use marginal policy (averaged over goals) for other agents
                marginals.append(self._to_probability_array(self(state, agent_index)))
        
        if not marginals:
            return [(1.0, [])]
        
        if device is not None and device.startswith('cuda'):
            return self._profile_distribution_torch(marginals, device)
        else:
            return self._profile_distribution_numpy(marginals)

    def _profile_distribution_numpy(self, marginals: List[np.ndarray]) -> List[tuple]:
        """NumPy implementation of profile distribution computation.
        
        Optimized with fast paths for common cases (1-2 agents).
        """
        n_agents = len(marginals)
        
        # Fast path for single agent (most common case)
        if n_agents == 1:
            m = marginals[0]
            # Direct iteration is faster than meshgrid for single agent
            return [(float(m[a]), [int(a)]) for a in np.nonzero(m > 0)[0]]
        
        # Fast path for two agents (very common case)
        if n_agents == 2:
            m0, m1 = marginals[0], marginals[1]
            # Use outer product instead of meshgrid - much faster
            joint = np.outer(m0, m1)
            i_indices, j_indices = np.nonzero(joint > 0)
            # Vectorized extraction of probabilities
            probs = joint[i_indices, j_indices]
            # Build result - convert indices to Python ints for list creation
            return [(float(probs[k]), [int(i_indices[k]), int(j_indices[k])]) 
                    for k in range(len(probs))]
        
        # General case for 3+ agents (rare)
        # Compute joint probabilities directly using broadcasting
        # Start with first marginal reshaped to broadcast
        joint = marginals[0].copy()
        for i in range(1, n_agents):
            # Reshape current joint to add new dimension, multiply with next marginal
            joint = np.outer(joint.ravel(), marginals[i]).reshape(joint.shape + (len(marginals[i]),))
        
        # Find non-zero entries
        nonzero_indices = np.nonzero(joint > 0)
        probs = joint[nonzero_indices]
        
        # Convert multi-indices to action profiles
        result = [
            (float(probs[k]), [int(nonzero_indices[agent][k]) for agent in range(n_agents)])
            for k in range(len(probs))
        ]
        
        return result
    
    def _profile_distribution_torch(self, marginals: List[np.ndarray], device: str) -> List[tuple]:
        """
        PyTorch implementation for GPU-accelerated profile distribution.
        
        Uses outer product formulation which is highly parallelizable:
        P(a1, a2, ..., an) = P(a1) * P(a2) * ... * P(an)
        
        For GPU: The joint probability tensor is computed as a series of
        outer products, which are batched matrix multiplications on GPU.
        """
        import torch
        
        # Convert marginals to torch tensors on specified device
        tensors = [torch.tensor(m, dtype=torch.float32, device=device) for m in marginals]
        
        # Compute joint probability tensor via iterative outer products
        # This creates a tensor of shape (A1, A2, ..., An) where Ai = num_actions for agent i
        joint = tensors[0]
        for t in tensors[1:]:
            # outer product: (shape...) x (k,) -> (shape..., k)
            joint = joint.unsqueeze(-1) * t
        
        # Flatten and find non-zero entries
        flat_probs = joint.flatten()
        nonzero_mask = flat_probs > 0.0
        nonzero_indices = torch.where(nonzero_mask)[0]
        nonzero_probs = flat_probs[nonzero_indices]
        
        # Convert flat indices back to multi-dimensional indices
        # Using torch.unravel_index equivalent
        shape = joint.shape
        profiles = []
        for flat_idx in nonzero_indices:
            idx = flat_idx.item()
            profile = []
            for dim_size in reversed(shape):
                profile.append(idx % dim_size)
                idx //= dim_size
            profiles.append(list(reversed(profile)))
        
        # Build result (move probabilities back to CPU for output)
        probs_cpu = nonzero_probs.cpu().numpy()
        result = [(float(probs_cpu[i]), profiles[i]) for i in range(len(profiles))]
        
        return result


class TabularHumanPolicyPrior(HumanPolicyPrior):
    """
    Tabular (lookup-table) implementation of human policy prior.
    
    This implementation stores precomputed policy distributions in a nested
    dictionary structure, indexed by (state, agent_index, goal).
    
    Typically created by the `compute_human_policy_prior()` function which
    performs backward induction to compute optimal Boltzmann policies.
    
    Attributes:
        values: Nested dict mapping state -> agent_index -> goal -> action_distribution.
        possible_goal_generator: Generator for enumerating possible goals.
    
    Structure of `values`:
        {
            state1: {
                agent_idx1: {
                    goal1: np.array([p_action0, p_action1, ...]),
                    goal2: np.array([...]),
                    ...
                },
                agent_idx2: {...},
            },
            state2: {...},
        }
    """

    values: dict
    possible_goal_generator: 'PossibleGoalGenerator'

    def __init__(
        self, 
        world_model: 'WorldModel', 
        human_agent_indices: List[int], 
        possible_goal_generator: 'PossibleGoalGenerator', 
        values: dict
    ):
        """
        Initialize the tabular policy prior.
        
        Args:
            world_model: The world model (environment) this prior applies to.
            human_agent_indices: List of indices of human agents.
            possible_goal_generator: Generator for enumerating possible goals.
            values: Precomputed policy lookup table (state -> agent -> goal -> distribution).
        """
        super().__init__(world_model, human_agent_indices)
        self.values = values
        self.possible_goal_generator = possible_goal_generator

    def __call__(
        self, 
        state, 
        human_agent_index: int, 
        possible_goal: Optional['PossibleGoal'] = None
    ) -> np.ndarray:
        """
        Look up or compute the action distribution.
        
        Args:
            state: Current world state (must be a key in self.values).
            human_agent_index: Index of the human agent.
            possible_goal: If provided, return distribution conditioned on this goal.
                          If None, compute marginal by averaging over goals.
        
        Returns:
            np.ndarray: Probability distribution over actions.
        
        Raises:
            KeyError: If state or agent_index not found in lookup table.
        """
        if possible_goal is not None:
#            key = possible_goal.index if hasattr(possible_goal, 'index') else possible_goal
            key = possible_goal
            return self.values[state][human_agent_index][key]
        else:
            # Compute marginal by averaging over goals weighted by their prior
            vs = self.values[state][human_agent_index]
            # Support override for parallel mode where world_model is None after unpickling
            if hasattr(self, '_num_actions_override') and self._num_actions_override is not None:
                num_actions: int = self._num_actions_override
            else:
                num_actions = self.world_model.action_space.n  # type: ignore[attr-defined]
            total = np.zeros(num_actions)
            for goal, weight in self.possible_goal_generator.generate(state, human_agent_index):
#                key = goal.index if hasattr(goal, 'index') else goal
                key = goal
                total += vs[key] * weight
            return total
        

class HeuristicPotentialPolicy(HumanPolicyPrior):
    """
    Heuristic goal-directed policy based on potential function gradients.
    
    This policy uses the precomputed shortest paths and potential function from
    PathDistanceCalculator to guide agents toward goals without learning. At each
    state, it evaluates the potential at the current position and at all
    neighboring empty cells, then produces a soft probability distribution over
    actions that favors moves toward higher potential (closer to goal).
    
    Door Handling:
    If the agent is adjacent to an actionable door, the policy overrides the
    potential-based action and instead turns toward the door and opens/unlocks it.
    An actionable door is either:
    - A locked door where the agent carries a key of matching color
    - An unlocked but closed door
    
    When facing an actionable door:
    - If num_actions > 6 (full Actions set), uses the toggle action
    - Otherwise, uses the forward action (which won't actually open the door
      in SmallActions environments, but represents the agent's intent)
    
    Key Pickup:
    If the agent is adjacent to a key that can open at least one locked door,
    and the agent is not already carrying a key, the policy overrides the
    potential-based action and instead turns toward the key and picks it up.
    - If num_actions > 4 (full Actions set), uses the pickup action
    - Otherwise, uses the forward action
    
    Key Drop:
    If the agent is carrying a key that can no longer open any locked door
    (all matching doors have been unlocked), the policy overrides the potential-based
    action and instead drops the key on the neighboring cell with the worst potential
    (furthest from goal), freeing the agent to carry other objects if needed.
    - If num_actions > 5 (full Actions set), uses the drop action
    - Otherwise, uses the forward action
    
    The policy is parameterized by:
    - beta: Controls how deterministic the policy is.
      - beta -> 0: Uniform random over all actions
      - beta -> inf: Deterministic (always pick best action)
      - Typical values: 1-10 for exploration, 10-100 for exploitation
    
    Action selection logic (in priority order):
    1. Check for adjacent actionable doors (locked with matching key, or closed)
       - If found and not facing: return turn action to face the door
       - If found and facing: return toggle (or forward) action
    2. Check for adjacent useful keys (can open at least one locked door)
       - If found and not carrying a key: turn toward key and pick it up
    3. Check for useless key being carried (can't open any locked door)
       - If carrying useless key: turn toward worst-potential cell and drop it
    4. Compute potential Φ(current_pos) for current position
    5. For each of 4 cardinal directions, compute potential Φ(neighbor_pos)
       for the neighboring cell (if walkable)
    6. Compute advantage for each direction: A_dir = Φ(neighbor) - Φ(current)
    7. Convert to action probabilities using softmax:
       - If facing direction d: forward action gets advantage A_d
       - Turn actions get advantage of the direction they would face
       - Still action gets advantage 0 (no improvement)
    8. Apply softmax: P(action) ∝ exp(beta * advantage)
    
    Attributes:
        path_calculator: PathDistanceCalculator for potential computation.
        beta: Temperature parameter for softmax (higher = more deterministic).
        num_actions: Number of available actions (typically 4 for SmallActions).
    
    Note:
        This policy requires a MultiGridEnv-like world model with:
        - grid attribute for checking cell contents
        - agents attribute with pos and dir for each agent
        - get_state() method returning (step_count, agent_states, mobile_objects, mutable_objects)
    """
    
    # Direction vectors: 0=east (+x), 1=south (+y), 2=west (-x), 3=north (-y)
    DIR_TO_VEC = [
        (1, 0),   # 0 = east
        (0, 1),   # 1 = south
        (-1, 0),  # 2 = west
        (0, -1),  # 3 = north
    ]
    
    # Action indices for SmallActions: 0=still, 1=left, 2=right, 3=forward
    ACTION_STILL = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_FORWARD = 3
    
    # Action indices for full Actions set
    ACTION_PICKUP = 4
    ACTION_DROP = 5
    ACTION_TOGGLE = 6
    
    def __init__(
        self,
        world_model: 'WorldModel',
        human_agent_indices: List[int],
        path_calculator,  # PathDistanceCalculator from empo.learning_based.multigrid
        beta: float = 10.0,
        num_actions: int = 4
    ):
        """
        Initialize the heuristic potential policy.
        
        Args:
            world_model: The multigrid environment.
            human_agent_indices: Indices of agents to control with this policy.
            path_calculator: PathDistanceCalculator with precomputed shortest paths.
            beta: Softmax temperature. Higher = more deterministic.
                  Default 10.0 provides moderate exploitation.
            num_actions: Number of actions (default 4 for SmallActions).
        """
        super().__init__(world_model, human_agent_indices)
        self.path_calculator = path_calculator
        self.beta = beta
        self.num_actions = num_actions
    
    def set_world_model(self, world_model: 'WorldModel') -> None:
        """
        Set or update the world model reference.
        
        For HeuristicPotentialPolicy, this also recreates the path_calculator
        to reflect the new environment's wall layout.
        
        Args:
            world_model: The world model (environment) to use.
        """
        super().set_world_model(world_model)
        
        # Recreate path calculator for new environment
        # Import here to avoid circular dependency
        from empo.learning_based.multigrid import PathDistanceCalculator
        
        self.path_calculator = PathDistanceCalculator(
            grid_height=world_model.height,
            grid_width=world_model.width,
            world_model=world_model
        )
    
    def _get_goal_tuple(self, possible_goal: 'PossibleGoal') -> tuple:
        """
        Extract goal coordinates from a PossibleGoal object.
        
        Handles both point goals (x, y) and rectangle goals (x1, y1, x2, y2).
        
        Args:
            possible_goal: Goal object with target_pos or target_rect attribute.
        
        Returns:
            Tuple of (x, y) for point goals or (x1, y1, x2, y2) for rectangles.
        """
        if hasattr(possible_goal, 'target_rect'):
            return possible_goal.target_rect
        elif hasattr(possible_goal, 'target_pos'):
            return possible_goal.target_pos
        elif hasattr(possible_goal, 'target'):
            target = possible_goal.target
            if isinstance(target, tuple):
                return target
        # Fallback: try to extract from string representation or raise error
        raise ValueError(f"Cannot extract goal coordinates from {possible_goal}")
    
    def _is_cell_walkable(self, x: int, y: int, agent_positions: set) -> bool:
        """
        Check if a cell is walkable (can be moved into).
        
        A cell is walkable if:
        - It's within grid bounds
        - It's not a wall, lava, or other impassable obstacle
        - It's not occupied by another agent
        - It's not a closed/locked door (unless the agent can open it)
        
        Note: Blocks and rocks are considered walkable because they can be pushed.
        The potential function already accounts for the difficulty of pushing them
        via higher passing costs in PathDistanceCalculator.
        
        Args:
            x, y: Cell coordinates to check.
            agent_positions: Set of (x, y) positions occupied by agents.
        
        Returns:
            True if the cell can be entered, False otherwise.
        """
        # Check bounds
        if not (0 <= x < self.world_model.width and 0 <= y < self.world_model.height):
            return False
        
        # Check if occupied by another agent
        if (x, y) in agent_positions:
            return False
        
        # Check cell contents
        cell = self.world_model.grid.get(x, y)
        if cell is None:
            return True  # Empty cell
        
        cell_type = getattr(cell, 'type', None)
        
        # Impassable obstacles (cannot be moved through even with pushing)
        if cell_type in ('wall', 'magicwall', 'lava', 'bush', 'killbutton', 'pauseswitch', 'disablingswitch', 'controlbutton'):
            return False
        
        # Doors: closed/locked doors are obstacles (for simplicity)
        if cell_type == 'door':
            is_open = getattr(cell, 'is_open', False)
            if not is_open:
                return False
        
        # Blocks and rocks ARE walkable - they can be pushed
        # The potential function already accounts for the pushing cost
        # (block cost = 2, rock cost = 50 in DEFAULT_PASSING_COSTS)
        
        # Everything else (floor, goal, key, ball, box, switch, block, rock, etc.) is walkable
        return True
    
    def _find_actionable_door(
        self,
        agent_x: int,
        agent_y: int,
        carrying_type: Optional[str],
        carrying_color: Optional[str]
    ) -> Optional[int]:
        """
        Find an adjacent door that can be opened/unlocked by the agent.
        
        An actionable door is either:
        - A locked door where the agent carries a key of matching color
        - An unlocked but closed door
        
        Args:
            agent_x, agent_y: Agent's current position.
            carrying_type: Type of object the agent is carrying (e.g., 'key'), or None.
            carrying_color: Color of the carried object, or None.
        
        Returns:
            Direction (0-3) of the actionable door, or None if no actionable door found.
            Direction 0=east, 1=south, 2=west, 3=north.
        """
        for direction in range(4):
            dx, dy = self.DIR_TO_VEC[direction]
            neighbor_x, neighbor_y = agent_x + dx, agent_y + dy
            
            # Check bounds
            if not (0 <= neighbor_x < self.world_model.width and 
                    0 <= neighbor_y < self.world_model.height):
                continue
            
            cell = self.world_model.grid.get(neighbor_x, neighbor_y)
            if cell is None:
                continue
            
            cell_type = getattr(cell, 'type', None)
            if cell_type != 'door':
                continue
            
            is_open = getattr(cell, 'is_open', False)
            is_locked = getattr(cell, 'is_locked', False)
            door_color = getattr(cell, 'color', None)
            
            # Skip already-open doors
            if is_open:
                continue
            
            # Check if the door can be opened
            if is_locked:
                # Locked door: need matching key
                if carrying_type == 'key' and carrying_color == door_color:
                    return direction
            else:
                # Unlocked but closed door: can be opened by anyone
                return direction
        
        return None
    
    def _get_locked_doors_by_color(self) -> dict:
        """
        Get a dictionary mapping door colors to lists of locked door positions.
        
        Returns:
            Dict mapping color -> list of (x, y) positions of locked doors.
        """
        locked_doors = {}
        
        for y in range(self.world_model.height):
            for x in range(self.world_model.width):
                cell = self.world_model.grid.get(x, y)
                if cell is None:
                    continue
                
                cell_type = getattr(cell, 'type', None)
                if cell_type != 'door':
                    continue
                
                is_locked = getattr(cell, 'is_locked', False)
                if not is_locked:
                    continue
                
                door_color = getattr(cell, 'color', None)
                if door_color is not None:
                    if door_color not in locked_doors:
                        locked_doors[door_color] = []
                    locked_doors[door_color].append((x, y))
        
        return locked_doors
    
    def _find_useful_key(
        self,
        agent_x: int,
        agent_y: int,
        carrying_type: Optional[str],
        locked_doors_by_color: dict
    ) -> Optional[int]:
        """
        Find an adjacent key that can open at least one locked door.
        
        Only considers picking up a key if:
        - Agent is not already carrying a key
        - The key's color matches at least one locked door
        
        Args:
            agent_x, agent_y: Agent's current position.
            carrying_type: Type of object the agent is carrying, or None.
            locked_doors_by_color: Dict mapping color -> list of locked door positions.
        
        Returns:
            Direction (0-3) of the useful key, or None if no useful key found.
        """
        # Don't pick up a key if already carrying one
        if carrying_type == 'key':
            return None
        
        for direction in range(4):
            dx, dy = self.DIR_TO_VEC[direction]
            neighbor_x, neighbor_y = agent_x + dx, agent_y + dy
            
            # Check bounds
            if not (0 <= neighbor_x < self.world_model.width and 
                    0 <= neighbor_y < self.world_model.height):
                continue
            
            cell = self.world_model.grid.get(neighbor_x, neighbor_y)
            if cell is None:
                continue
            
            cell_type = getattr(cell, 'type', None)
            if cell_type != 'key':
                continue
            
            key_color = getattr(cell, 'color', None)
            
            # Check if this key can open at least one locked door
            if key_color in locked_doors_by_color and len(locked_doors_by_color[key_color]) > 0:
                return direction
        
        return None
    
    def _find_drop_cell_for_useless_key(
        self,
        agent_x: int,
        agent_y: int,
        carrying_type: Optional[str],
        carrying_color: Optional[str],
        locked_doors_by_color: dict,
        goal_tuple: tuple,
        blocked_positions: set
    ) -> Optional[int]:
        """
        Find an adjacent cell to drop a useless key (one that can't open any locked door).
        
        The key is dropped on the neighboring cell with the worst potential (furthest from goal).
        
        Args:
            agent_x, agent_y: Agent's current position.
            carrying_type: Type of object the agent is carrying.
            carrying_color: Color of the carried object.
            locked_doors_by_color: Dict mapping color -> list of locked door positions.
            goal_tuple: Goal coordinates for potential calculation.
            blocked_positions: Set of positions blocked by agents/rocks.
        
        Returns:
            Direction (0-3) of the best cell to drop the key, or None if can't/shouldn't drop.
        """
        # Only consider dropping if carrying a key
        if carrying_type != 'key':
            return None
        
        # Check if the key can still open at least one locked door
        if carrying_color in locked_doors_by_color and len(locked_doors_by_color[carrying_color]) > 0:
            return None  # Key is still useful, don't drop it
        
        # Find the neighboring cell with the worst potential (most negative = furthest from goal)
        worst_potential = float('inf')
        worst_direction = None
        
        for direction in range(4):
            dx, dy = self.DIR_TO_VEC[direction]
            neighbor_x, neighbor_y = agent_x + dx, agent_y + dy
            
            # Check bounds
            if not (0 <= neighbor_x < self.world_model.width and 
                    0 <= neighbor_y < self.world_model.height):
                continue
            
            # Check if the cell is empty and not blocked
            if (neighbor_x, neighbor_y) in blocked_positions:
                continue
            
            cell = self.world_model.grid.get(neighbor_x, neighbor_y)
            
            # Can only drop on empty cells or cells that can be overlapped
            if cell is not None:
                cell_type = getattr(cell, 'type', None)
                # Can't drop on walls, doors, keys, balls, boxes, etc.
                if cell_type in ('wall', 'door', 'key', 'ball', 'box', 'block', 'rock', 'lava', 'magicwall', 'bush'):
                    continue
            
            # Calculate potential for this cell
            neighbor_potential = self.path_calculator.compute_potential(
                (neighbor_x, neighbor_y), goal_tuple, self.world_model
            )
            
            if neighbor_potential < worst_potential:
                worst_potential = neighbor_potential
                worst_direction = direction
        
        return worst_direction
    
    def _get_turn_action_to_face(self, current_dir: int, target_dir: int) -> int:
        """
        Get the turn action to face from current_dir toward target_dir.
        
        Args:
            current_dir: Current facing direction (0-3).
            target_dir: Target direction to face (0-3).
        
        Returns:
            Action index (ACTION_LEFT, ACTION_RIGHT, or ACTION_FORWARD if already facing).
        """
        if current_dir == target_dir:
            # Already facing the target direction
            return self.ACTION_FORWARD
        
        # Calculate turn direction
        diff = (target_dir - current_dir) % 4
        if diff == 1:
            return self.ACTION_RIGHT
        elif diff == 3:
            return self.ACTION_LEFT
        else:
            # diff == 2: Either direction works, prefer right
            return self.ACTION_RIGHT
    
    def __call__(
        self,
        state,
        human_agent_index: int,
        possible_goal: Optional['PossibleGoal'] = None
    ) -> np.ndarray:
        """
        Compute action distribution based on potential gradients.
        
        Args:
            state: Current world state tuple from get_state().
            human_agent_index: Index of the agent to get actions for.
            possible_goal: The goal to move toward. Required for this policy.
        
        Returns:
            np.ndarray: Probability distribution over actions.
        
        Raises:
            ValueError: If possible_goal is None (goal is required).
        """
        if possible_goal is None:
            # Without a goal, return uniform distribution
            return np.ones(self.num_actions) / self.num_actions
        
        # Extract state components
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Get agent position and direction
        if human_agent_index >= len(agent_states):
            return np.ones(self.num_actions) / self.num_actions
        
        agent_state = agent_states[human_agent_index]
        agent_x, agent_y = int(agent_state[0]), int(agent_state[1])
        agent_dir = int(agent_state[2]) if len(agent_state) > 2 else 0
        agent_pos = (agent_x, agent_y)
        
        # Extract carrying information from agent state
        # agent_state format: (x, y, dir, terminated, started, paused, carrying_type, carrying_color, ...)
        carrying_type = agent_state[6] if len(agent_state) > 6 else None
        carrying_color = agent_state[7] if len(agent_state) > 7 else None
        
        # Check for actionable doors first (override potential-based action)
        # This handles: locked doors with matching key, or unlocked closed doors
        door_direction = self._find_actionable_door(
            agent_x, agent_y, carrying_type, carrying_color
        )
        
        if door_direction is not None:
            # Found an actionable door - override potential-based action
            probs = np.zeros(self.num_actions)
            
            if agent_dir == door_direction:
                # Already facing the door - use toggle action if available, otherwise forward
                # ACTION_TOGGLE is at index 6, so we need num_actions > 6 (i.e., at least 7 actions)
                if self.num_actions > self.ACTION_TOGGLE:
                    # Full Actions set with toggle available
                    probs[self.ACTION_TOGGLE] = 1.0
                else:
                    # SmallActions - use forward to bump into door (won't open it, but
                    # this policy does its best with available actions)
                    probs[self.ACTION_FORWARD] = 1.0
            else:
                # Turn to face the door
                turn_action = self._get_turn_action_to_face(agent_dir, door_direction)
                probs[turn_action] = 1.0
            
            return probs
        
        # Get goal coordinates
        goal_tuple = self._get_goal_tuple(possible_goal)
        
        # Build set of blocked positions (excluding current agent)
        # These are positions that are truly blocked (can't be pushed through)
        blocked_positions = set()
        for i, a_state in enumerate(agent_states):
            if i != human_agent_index:
                blocked_positions.add((int(a_state[0]), int(a_state[1])))
        
        # Add rocks to blocked positions - humans typically cannot push rocks
        # Blocks are NOT added - they are pushable by humans
        for obj_type, obj_x, obj_y in mobile_objects:
            if obj_type == 'rock':
                blocked_positions.add((obj_x, obj_y))
        
        # Get locked doors by color for key handling logic
        locked_doors_by_color = self._get_locked_doors_by_color()
        
        # Check for useful keys to pick up (override potential-based action)
        # This handles: keys that can open at least one locked door, when not carrying a key
        key_direction = self._find_useful_key(
            agent_x, agent_y, carrying_type, locked_doors_by_color
        )
        
        if key_direction is not None:
            # Found a useful key - turn toward it and pick it up
            probs = np.zeros(self.num_actions)
            
            if agent_dir == key_direction:
                # Already facing the key - use pickup action if available, otherwise forward
                # ACTION_PICKUP is at index 4, so we need num_actions > 4 (i.e., at least 5 actions)
                if self.num_actions > self.ACTION_PICKUP:
                    # Full Actions set with pickup available
                    probs[self.ACTION_PICKUP] = 1.0
                else:
                    # SmallActions - use forward to bump into key (won't pick it up, but
                    # this policy does its best with available actions)
                    probs[self.ACTION_FORWARD] = 1.0
            else:
                # Turn to face the key
                turn_action = self._get_turn_action_to_face(agent_dir, key_direction)
                probs[turn_action] = 1.0
            
            return probs
        
        # Check for useless keys to drop (override potential-based action)
        # This handles: keys that cannot open any locked door anymore
        drop_direction = self._find_drop_cell_for_useless_key(
            agent_x, agent_y, carrying_type, carrying_color,
            locked_doors_by_color, goal_tuple, blocked_positions
        )
        
        if drop_direction is not None:
            # Found a cell to drop the useless key - turn toward it and drop
            probs = np.zeros(self.num_actions)
            
            if agent_dir == drop_direction:
                # Already facing the drop cell - use drop action if available, otherwise forward
                # ACTION_DROP is at index 5, so we need num_actions > 5 (i.e., at least 6 actions)
                if self.num_actions > self.ACTION_DROP:
                    # Full Actions set with drop available
                    probs[self.ACTION_DROP] = 1.0
                else:
                    # SmallActions - use forward (won't drop, but this policy does its best)
                    probs[self.ACTION_FORWARD] = 1.0
            else:
                # Turn to face the drop cell
                turn_action = self._get_turn_action_to_face(agent_dir, drop_direction)
                probs[turn_action] = 1.0
            
            return probs
        
        # Compute potential at current position
        current_potential = self.path_calculator.compute_potential(
            agent_pos, goal_tuple, self.world_model
        )
        
        # Check if already at goal
        if self.path_calculator.is_in_goal(agent_pos, goal_tuple):
            # At goal: prefer staying still
            probs = np.zeros(self.num_actions)
            probs[self.ACTION_STILL] = 1.0
            return probs
        
        # Compute potential for each neighboring direction
        dir_advantages = {}  # direction -> advantage (potential improvement)
        
        for direction in range(4):
            dx, dy = self.DIR_TO_VEC[direction]
            neighbor_x, neighbor_y = agent_x + dx, agent_y + dy
            neighbor_pos = (neighbor_x, neighbor_y)
            
            if self._is_cell_walkable(neighbor_x, neighbor_y, blocked_positions):
                neighbor_potential = self.path_calculator.compute_potential(
                    neighbor_pos, goal_tuple, self.world_model
                )
                # Advantage is improvement in potential (higher is better)
                dir_advantages[direction] = neighbor_potential - current_potential
            else:
                # Can't move there: large negative advantage
                dir_advantages[direction] = -1.0
        
        # Convert direction advantages to action advantages
        # Actions: 0=still, 1=left, 2=right, 3=forward
        action_advantages = np.zeros(self.num_actions)
        
        # Still: no change in position, advantage = 0
        action_advantages[self.ACTION_STILL] = 0.0
        
        # Forward: move in current facing direction
        action_advantages[self.ACTION_FORWARD] = dir_advantages.get(agent_dir, -1.0)
        
        # Left: turn left (counter-clockwise), then would face (agent_dir - 1) % 4
        # This doesn't move, but positions for future forward
        left_dir = (agent_dir - 1) % 4
        action_advantages[self.ACTION_LEFT] = dir_advantages.get(left_dir, -1.0) * 0.5
        
        # Right: turn right (clockwise), then would face (agent_dir + 1) % 4
        right_dir = (agent_dir + 1) % 4
        action_advantages[self.ACTION_RIGHT] = dir_advantages.get(right_dir, -1.0) * 0.5
        
        # Special case: if facing the best direction and it's walkable, strongly prefer forward
        best_dir = max(dir_advantages, key=dir_advantages.get)
        if agent_dir == best_dir and dir_advantages[best_dir] > 0:
            action_advantages[self.ACTION_FORWARD] = dir_advantages[best_dir]
        elif dir_advantages[best_dir] > 0:
            # Need to turn toward best direction
            # Compute which turn is shorter
            diff = (best_dir - agent_dir) % 4
            if diff == 1:
                # Turn right once
                action_advantages[self.ACTION_RIGHT] = dir_advantages[best_dir] * 0.8
            elif diff == 3:
                # Turn left once
                action_advantages[self.ACTION_LEFT] = dir_advantages[best_dir] * 0.8
            elif diff == 2:
                # Either turn works, slight preference for right
                action_advantages[self.ACTION_RIGHT] = dir_advantages[best_dir] * 0.6
                action_advantages[self.ACTION_LEFT] = dir_advantages[best_dir] * 0.6
        
        # Apply softmax with temperature
        scaled = action_advantages * self.beta
        # Subtract max for numerical stability
        scaled = scaled - np.max(scaled)
        exp_scaled = np.exp(scaled)
        probs = exp_scaled / np.sum(exp_scaled)
        
        return probs


# Import PathDistanceCalculator for type hints (optional, avoids circular import)
try:
    from empo.learning_based.multigrid.path_distance import PathDistanceCalculator
except ImportError:
    PathDistanceCalculator = None  # type: ignore


# MultiGridHumanExplorationPolicy moved to exploration_policies.py
# Lazy import here for backwards compatibility (avoids circular import)
def __getattr__(name):
    if name == 'MultiGridHumanExplorationPolicy':
        from empo.learning_based.multigrid.phase2.exploration_policies import (
            MultiGridHumanExplorationPolicy,
        )
        return MultiGridHumanExplorationPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
