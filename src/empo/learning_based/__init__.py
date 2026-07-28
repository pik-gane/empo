"""
Neural network-based policy priors.

This package provides neural network function approximators for computing
policy priors when the state space is too large for tabular methods.

For multigrid environments, use the `multigrid` subpackage:

    from empo.learning_based.multigrid import (
        MultiGridNeuralHumanPolicyPrior,
        train_multigrid_neural_policy_prior,
        MultiGridQNetwork,
        MultiGridStateEncoder,
    )

Base classes for custom implementations:

    from empo.learning_based import (
        BaseStateEncoder,
        BaseGoalEncoder,
        BaseQNetwork,
        BasePolicyPriorNetwork,
        BaseNeuralHumanPolicyPrior,
        Trainer,
    )
"""

from .phase1.state_encoder import BaseStateEncoder
from .phase1.goal_encoder import BaseGoalEncoder
from .phase1.q_network import BaseQNetwork
from .util.soft_clamp import SoftClamp
from .phase1.policy_prior_network import BasePolicyPriorNetwork
from .phase1.neural_policy_prior import BaseNeuralHumanPolicyPrior
from .phase1.replay_buffer import ReplayBuffer
from .phase1.trainer import Trainer

from . import multigrid

__all__ = [
    # Base classes
    'BaseStateEncoder',
    'BaseGoalEncoder',
    'BaseQNetwork',
    'SoftClamp',
    'BasePolicyPriorNetwork',
    'BaseNeuralHumanPolicyPrior',
    # Utilities
    'ReplayBuffer',
    'Trainer',
    # Subpackage
    'multigrid',
]
