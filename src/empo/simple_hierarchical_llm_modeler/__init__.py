"""
simple_hierarchical_llm_modeler — LLM-prompting-based world-model construction.

This subpackage implements a simplistic alternative to the full PDDL-based
LLM world-model pipeline.  It recursively queries an LLM to build a
state–robot-action–humans-reaction–observation trajectory tree, then converts
that tree into a :class:`~empo.world_model.WorldModel` (or a hierarchical
pair of world models).

Main entry points
-----------------
* :func:`build_tree` — build a trajectory tree from an NL state description.
* :class:`NLWorldModel` — a concrete WorldModel backed by the trajectory tree.
* :func:`build_two_level_model` — build a two-level hierarchical world model.
* :func:`check_hierarchical_status` — ask whether a sub-activity succeeded.
* :func:`match_consequence` — map an outcome to a known higher-level branch.

LLM connector
-------------
Any object satisfying the :class:`LLMConnector` protocol can be used.  The
:class:`L2PConnector` adapter wraps a vendored L2P ``BaseLLM`` instance.
"""

from empo.simple_hierarchical_llm_modeler.llm_connector import (
    AnthropicConnector,
    CachedLLMConnector,
    L2PConnector,
    LLMConnector,
    StatsTrackingLLM,
)
from empo.simple_hierarchical_llm_modeler.tree_builder import (
    LiveTreeRenderer,
    TreeNode,
    build_tree,
    collect_leaves,
    count_nodes,
)
from empo.simple_hierarchical_llm_modeler.nl_world_model import (
    NLWorldModel,
    AllSingleStatesGoalGenerator,
    SingleStateGoal,
)
from empo.simple_hierarchical_llm_modeler.hierarchical_modeler import (
    LazyTwoLevelModel,
    NLLevelMapper,
    build_two_level_model,
    check_hierarchical_status,
    match_consequence,
)

__all__ = [
    # LLM connector
    "LLMConnector",
    "L2PConnector",
    "AnthropicConnector",
    "CachedLLMConnector",
    "StatsTrackingLLM",
    # Tree builder
    "LiveTreeRenderer",
    "TreeNode",
    "build_tree",
    "count_nodes",
    "collect_leaves",
    # NL world model
    "NLWorldModel",
    "AllSingleStatesGoalGenerator",
    "SingleStateGoal",
    # Hierarchical
    "LazyTwoLevelModel",
    "NLLevelMapper",
    "build_two_level_model",
    "check_hierarchical_status",
    "match_consequence",
]
