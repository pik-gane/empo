"""
Transport-specific neural network encoders and policy components.

This subpackage contains all transport-specific implementations for encoding
environment state into neural network inputs using Graph Neural Networks (GNNs)
instead of CNNs (as used in multigrid).

The transport environment is graph-based (road network), so we use GNNs to
process node and edge features. The encoders support both node-based and
cluster-based routing modes.

Main components:
    - constants: Feature dimensions, step type encoding, action encoding
    - feature_extraction: Extract features from transport env state
    - state_encoder: GNN-based state encoder for network topology
    - goal_encoder: Encode node/cluster goals
    - q_network: Q-network combining state and goal encoders
    - policy_prior_network: Compute marginal action probabilities
    - neural_policy_prior: Human policy prior using GNN-based Q-network

Example usage:
    >>> from empo.learning_based.transport import (
    ...     TransportQNetwork,
    ...     TransportNeuralHumanPolicyPrior,
    ...     observation_to_graph_data,
    ... )
    >>> 
    >>> # Create Q-network
    >>> q_network = TransportQNetwork(
    ...     max_nodes=100, num_clusters=10, num_actions=42
    ... )
    >>> 
    >>> # Create policy prior
    >>> prior = TransportNeuralHumanPolicyPrior.create(
    ...     world_model=env,
    ...     human_agent_indices=[0, 1, 2, 3],
    ...     num_clusters=10,
    ... )
    >>> 
    >>> # Get action probabilities
    >>> probs = prior(state=None, agent_idx=0, goal=goal)
"""

from .phase1.constants import (
    # Step type encoding
    STEP_TYPE_TO_IDX,
    IDX_TO_STEP_TYPE,
    NUM_STEP_TYPES,
    # Feature dimensions
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    AGENT_FEATURE_DIM,
    # Action encoding
    NUM_TRANSPORT_ACTIONS,
    MAX_VEHICLES_AT_NODE,
    MAX_OUTGOING_EDGES,
    MAX_CLUSTERS,
    MAX_PARKING_SLOTS,
)

from .phase1.feature_extraction import (
    extract_node_features,
    extract_edge_features,
    extract_global_features,
    extract_agent_features,
    observation_to_graph_data,
)

from .phase1.state_encoder import TransportStateEncoder
from .phase1.goal_encoder import TransportGoalEncoder
from .phase1.q_network import TransportQNetwork
from .phase1.policy_prior_network import TransportPolicyPriorNetwork
from .phase1.neural_policy_prior import (
    TransportNeuralHumanPolicyPrior,
    DEFAULT_TRANSPORT_ACTION_ENCODING,
    train_transport_neural_policy_prior,
)

__all__ = [
    # Constants
    'STEP_TYPE_TO_IDX',
    'IDX_TO_STEP_TYPE',
    'NUM_STEP_TYPES',
    'NODE_FEATURE_DIM',
    'EDGE_FEATURE_DIM',
    'GLOBAL_FEATURE_DIM',
    'AGENT_FEATURE_DIM',
    'NUM_TRANSPORT_ACTIONS',
    'MAX_VEHICLES_AT_NODE',
    'MAX_OUTGOING_EDGES',
    'MAX_CLUSTERS',
    'MAX_PARKING_SLOTS',
    'DEFAULT_TRANSPORT_ACTION_ENCODING',
    # Feature extraction
    'extract_node_features',
    'extract_edge_features',
    'extract_global_features',
    'extract_agent_features',
    'observation_to_graph_data',
    # Encoders
    'TransportStateEncoder',
    'TransportGoalEncoder',
    # Networks
    'TransportQNetwork',
    'TransportPolicyPriorNetwork',
    'TransportNeuralHumanPolicyPrior',
    # Training
    'train_transport_neural_policy_prior',
]
