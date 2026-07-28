"""
End-to-end pipeline: natural language → WorldModel.

WorldModelBuilder is the single-entry-point class that orchestrates the full
pipeline from a natural language scene description to a WorldModel instance.
"""

import logging
from typing import List, Optional, Tuple

from l2p.llm.base import BaseLLM
from l2p.utils.pddl_validator import SyntaxValidator

from empo.llm_world_model.agent_builder import AgentBuilder
from empo.llm_world_model.ma_pddl_writer import MAPddlWriter
from empo.llm_world_model.pddl_world_model import PddlWorldModel
from empo.llm_world_model.types import MADomainSpec, MATaskSpec
from empo.llm_world_model.world_model_domain_builder import WorldModelDomainBuilder
from empo.llm_world_model.world_model_task_builder import WorldModelTaskBuilder

LOG = logging.getLogger(__name__)


class WorldModelBuilder:
    """
    End-to-end pipeline: natural language → WorldModel.

    Usage:
        builder = WorldModelBuilder(llm=my_llm)
        world_model = builder.build(
            scene_desc="A household with a robot and a human...",
            enable_probabilistic=True,
        )

        # world_model is a WorldModel (gymnasium.Env) ready for EMPO
        state = world_model.get_state()
        transitions = world_model.transition_probabilities(state, [0, 1])
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_dir: str = None,
        syntax_validator: SyntaxValidator = None,
        max_retries: int = 3,
        max_steps: int = 50,
    ):
        self._llm = llm
        self._prompt_dir = prompt_dir
        self._syntax_validator = syntax_validator
        self._max_retries = max_retries
        self._max_steps = max_steps

        self._agent_builder = AgentBuilder()
        self._domain_builder = WorldModelDomainBuilder()
        self._task_builder = WorldModelTaskBuilder()
        self._writer = MAPddlWriter()

        self._last_domain: Optional[MADomainSpec] = None
        self._last_task: Optional[MATaskSpec] = None

    def build(
        self,
        scene_desc: str,
        enable_probabilistic: bool = False,
        domain_name: str = "world",
        problem_name: str = "scenario",
    ) -> PddlWorldModel:
        """
        Build a WorldModel from a natural language scene description.

        Steps:
        1. Identify agents (AgentBuilder)
        2. Extract domain (WorldModelDomainBuilder)
        3. Extract task (WorldModelTaskBuilder)
        4. Convert to WorldModel (PddlWorldModel)

        Returns:
            A PddlWorldModel instance ready for EMPO computation.
        """
        LOG.info("Step 1: Identifying agents...")
        agents, _ = self._agent_builder.identify_agents(
            model=self._llm,
            scene_desc=scene_desc,
            max_retries=self._max_retries,
        )
        LOG.info("Identified %d agents: %s", len(agents), [a.name for a in agents])

        LOG.info("Step 2: Extracting domain...")
        domain = self._domain_builder.build_ma_domain(
            scene_desc=scene_desc,
            agents=agents,
            model=self._llm,
            enable_probabilistic=enable_probabilistic,
            syntax_validator=self._syntax_validator,
            max_retries=self._max_retries,
        )
        domain.name = domain_name

        LOG.info("Step 3: Extracting task (objects + initial state)...")
        task, _, _ = self._task_builder.formalize_objects_and_initial(
            model=self._llm,
            scene_desc=scene_desc,
            domain=domain,
            syntax_validator=self._syntax_validator,
            max_retries=self._max_retries,
        )
        task.name = problem_name
        task.domain_name = domain_name

        LOG.info("Step 4: Building PddlWorldModel...")
        world_model = PddlWorldModel(
            domain=domain,
            task=task,
            max_steps=self._max_steps,
        )

        LOG.info(
            "WorldModel built: %d agents, %d ground atoms, %s actions per agent",
            len(domain.agents),
            world_model._num_atoms,
            {n: c for n, c in world_model._agent_action_counts.items()},
        )

        # Store for later access
        self._last_domain = domain
        self._last_task = task

        return world_model

    def build_and_export(
        self,
        scene_desc: str,
        output_dir: str,
        factored: bool = False,
        enable_probabilistic: bool = False,
        domain_name: str = "world",
        problem_name: str = "scenario",
    ) -> Tuple[PddlWorldModel, List[str]]:
        """
        Build WorldModel and also export MA-PDDL files.

        Returns:
            (world_model, list_of_pddl_file_paths)
        """
        world_model = self.build(
            scene_desc=scene_desc,
            enable_probabilistic=enable_probabilistic,
            domain_name=domain_name,
            problem_name=problem_name,
        )

        paths = self._writer.write_files(
            domain=world_model._domain,
            task=world_model._task,
            output_dir=output_dir,
            factored=factored,
        )

        return world_model, paths

    @property
    def domain(self) -> Optional[MADomainSpec]:
        """The last built domain spec (if any)."""
        return self._last_domain

    @property
    def task(self) -> Optional[MATaskSpec]:
        """The last built task spec (if any)."""
        return self._last_task
