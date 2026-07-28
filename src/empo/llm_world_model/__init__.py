"""
LLM-based WorldModel formation module.

This module provides a pipeline for constructing WorldModel instances from
natural language scene descriptions using LLMs. The pipeline uses L2P
(Language to Planning) as a foundation and extends it for multi-agent
PDDL (MA-PDDL) extraction without goals.

Key classes:
    - AgentBuilder: Extract agent specifications from natural language
    - WorldModelDomainBuilder: Extract MA-PDDL domain (extends L2P DomainBuilder)
    - WorldModelTaskBuilder: Extract objects + initial state (extends L2P TaskBuilder)
    - MAPddlWriter: Write MA-PDDL domain/problem files
    - PddlWorldModel: Convert MA-PDDL to a WorldModel instance
    - WorldModelBuilder: End-to-end pipeline façade
"""

from empo.llm_world_model.types import (
    AgentSpec,
    ConcurrentEffect,
    MADomainSpec,
    MATaskSpec,
    PrescribedAction,
    StateObservation,
)
from empo.llm_world_model.agent_builder import AgentBuilder
from empo.llm_world_model.world_model_domain_builder import WorldModelDomainBuilder
from empo.llm_world_model.world_model_task_builder import WorldModelTaskBuilder
from empo.llm_world_model.ma_pddl_writer import MAPddlWriter
from empo.llm_world_model.pddl_world_model import PddlWorldModel
from empo.llm_world_model.world_model_builder import WorldModelBuilder

__all__ = [
    # Types
    "AgentSpec",
    "ConcurrentEffect",
    "MADomainSpec",
    "MATaskSpec",
    "PrescribedAction",
    "StateObservation",
    # Builders
    "AgentBuilder",
    "WorldModelDomainBuilder",
    "WorldModelTaskBuilder",
    "MAPddlWriter",
    "PddlWorldModel",
    "WorldModelBuilder",
]
