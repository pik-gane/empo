"""
empo - Empowerment-based AI agents for multi-agent environments

A framework for multi-agent reinforcement learning that provides:

1. **World Model Abstraction** (`WorldModel`):
   - Abstract base class for gymnasium environments with explicit state management
   - Provides `get_state()`, `set_state()`, and `transition_probabilities()` methods
   - Supports DAG computation for finite state spaces
   - Used by the vendored MultiGrid environment

2. **Human Policy Prior Modeling** (`HumanPolicyPrior`, `TabularHumanPolicyPrior`):
   - Abstract and tabular implementations for modeling human behavior
   - Computes goal-conditioned policy distributions
   - Supports marginalization over possible goals

3. **Possible Goal Specification** (`PossibleGoal`, `PossibleGoalGenerator`, `PossibleGoalSampler`):
   - Abstract classes for defining and enumerating possible goals
   - Supports exact enumeration and stochastic sampling

4. **Backward Induction** (`compute_human_policy_prior`):
   - Computes optimal human policies via backward induction on the state DAG
   - Supports parallel computation for large state spaces
   - Uses Boltzmann (softmax) policies with configurable temperature

This package works with the vendored MultiGrid environment in `vendor/multigrid/`
which has been extended with:
- Explicit state management (get_state, set_state)
- Transition probability computation
- New object types: Block, Rock, Bush, UnsteadyGround, MagicWall
- Map-based environment specification

Example usage:
    >>> from empo import WorldModel
    >>> from multigrid_worlds.one_or_three_chambers import SmallOneOrTwoChambersMapEnv
    >>> 
    >>> env = SmallOneOrTwoChambersMapEnv()
    >>> isinstance(env, WorldModel)
    True
    >>> state = env.get_state()
    >>> transitions = env.transition_probabilities(state, [0, 0])
"""

from empo.world_model import WorldModel
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler
from empo.human_policy_prior import HumanPolicyPrior, TabularHumanPolicyPrior
from empo.robot_policy import RobotPolicy
from empo.backward_induction import compute_human_policy_prior, TabularRobotPolicy
from empo.hierarchical import HierarchicalWorldModel, LevelMapper
from empo.util.memory_monitor import MemoryMonitor, check_memory

# Transport environment wrapper (optional import - requires ai_transport)
try:
    _HAS_TRANSPORT = True
except ImportError:
    _HAS_TRANSPORT = False

__version__ = "0.1.0"

__all__ = [
    # World Model
    "WorldModel",
    # Possible Goals
    "PossibleGoal",
    "PossibleGoalGenerator", 
    "PossibleGoalSampler",
    # Human Policy Prior
    "HumanPolicyPrior",
    "TabularHumanPolicyPrior",
    # Robot Policy
    "RobotPolicy",
    "TabularRobotPolicy",
    # Backward Induction
    "compute_human_policy_prior",
    # Hierarchical
    "HierarchicalWorldModel",
    "LevelMapper",
    # Memory Monitoring
    "MemoryMonitor",
    "check_memory",
]

# Add transport exports if available
if _HAS_TRANSPORT:
    __all__.extend([
        "TransportEnvWrapper",
        "TransportActions",
        "StepType",
        "create_transport_env",
        "TransportGoal",
        "TransportGoalGenerator",
        "TransportGoalSampler",
    ])
