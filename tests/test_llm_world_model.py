"""
Tests for the LLM-based WorldModel formation pipeline.

Uses a MockLLM with canned responses to test the full pipeline without
API calls. Covers Steps 1-7 of the llm2model plan.
"""

import json
import os
import tempfile
from collections import OrderedDict
from typing import List

import pytest

from l2p.llm.base import BaseLLM

from empo.llm_world_model.types import (
    AgentSpec,
    ConcurrentEffect,
    MADomainSpec,
    MATaskSpec,
)
from empo.llm_world_model.agent_builder import AgentBuilder, _parse_agents_json
from empo.llm_world_model.world_model_domain_builder import (
    WorldModelDomainBuilder,
    _parse_concurrent_effects_json,
)
from empo.llm_world_model.world_model_task_builder import (
    WorldModelTaskBuilder,
    _parse_objects_block,
    _parse_initial_block,
)
from empo.llm_world_model.ma_pddl_writer import MAPddlWriter
from empo.llm_world_model.pddl_world_model import (
    PddlWorldModel,
    _parse_atom,
    _parse_pddl_expression,
    GroundAction,
)
from empo.llm_world_model.world_model_builder import WorldModelBuilder


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """Mock LLM that returns pre-configured responses."""

    def __init__(self):
        # Skip BaseLLM.__init__ which validates model names
        self.output = ""
        self._responses = []
        self._call_idx = 0

    def query(self, prompt: str) -> str:
        if self._responses:
            if self._call_idx < len(self._responses):
                resp = self._responses[self._call_idx]
                self._call_idx += 1
                return resp
            return self._responses[-1]
        return self.output

    def valid_models(self):
        return ["mock"]

    def reset_tokens(self):
        pass

    def set_responses(self, responses: List[str]):
        """Set a sequence of responses for multiple LLM calls."""
        self._responses = responses
        self._call_idx = 0


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

HOUSEHOLD_SCENE = (
    "A household with a robot and a human. The robot can move between rooms "
    "and pick up objects. The human can also move between rooms and interact "
    "with objects. There is a kitchen, a living room, and a bedroom. "
    "Objects include a cup and a book."
)

AGENTS_JSON = json.dumps([
    {
        "name": "robot",
        "agent_type": "robot",
        "capabilities": ["move between rooms", "pick up objects", "place objects"],
        "action_hints": ["move", "pick_up", "place"],
    },
    {
        "name": "human",
        "agent_type": "person",
        "capabilities": ["move between rooms", "pick up objects", "read"],
        "action_hints": ["move", "pick_up"],
    },
])

TYPES_RESPONSE = """### Types
- room: a location in the household
- robot: a mobile robot agent
- person: a human agent
- object: a manipulable object
"""

PREDICATES_RESPONSE = """### New Predicates
```
- (at ?a - robot ?r - room): 'robot is in room'
- (at_person ?p - person ?r - room): 'person is in room'
- (in_room ?o - object ?r - room): 'object is in room'
- (holding ?a - robot ?o - object): 'robot is holding object'
- (holding_person ?p - person ?o - object): 'person is holding object'
- (connected ?r1 - room ?r2 - room): 'rooms are connected'
```
"""

MOVE_ACTION_RESPONSE = """### Action Parameters
```
- ?agent - robot: 'the robot'
- ?from - room: 'source room'
- ?to - room: 'destination room'
```

### Action Preconditions
```
(and
    (at ?agent ?from)
    (connected ?from ?to)
)
```

### Action Effects
```
(and
    (at ?agent ?to)
    (not (at ?agent ?from))
)
```

### New Predicates
```
```
"""

PICK_UP_ACTION_RESPONSE = """### Action Parameters
```
- ?agent - robot: 'the robot'
- ?obj - object: 'the object to pick up'
- ?room - room: 'the room'
```

### Action Preconditions
```
(and
    (at ?agent ?room)
    (in_room ?obj ?room)
    (not (holding ?agent ?obj))
)
```

### Action Effects
```
(and
    (holding ?agent ?obj)
    (not (in_room ?obj ?room))
)
```

### New Predicates
```
```
"""

PLACE_ACTION_RESPONSE = """### Action Parameters
```
- ?agent - robot: 'the robot'
- ?obj - object: 'the object to place'
- ?room - room: 'the room'
```

### Action Preconditions
```
(and
    (at ?agent ?room)
    (holding ?agent ?obj)
)
```

### Action Effects
```
(and
    (in_room ?obj ?room)
    (not (holding ?agent ?obj))
)
```

### New Predicates
```
```
"""

CONCURRENT_EFFECTS_RESPONSE = """### CONCURRENT EFFECTS
```json
[
  {
    "agent_a": "robot",
    "action_a": "pick_up",
    "agent_b": "human",
    "action_b": "pick_up",
    "effect_type": "conflicting",
    "resolution": "If both try to pick up the same object, neither succeeds.",
    "pddl_condition": null,
    "pddl_effect": null
  }
]
```
"""

OBJECTS_INITIAL_RESPONSE = """### OBJECTS
```
robot1 - robot ; the household robot
human1 - person ; the household human
kitchen - room ; the kitchen
living_room - room ; the living room
bedroom - room ; the bedroom
cup - object ; a cup
book - object ; a book
```

### INITIAL
```
(at robot1 kitchen)
(at_person human1 living_room)
(in_room cup kitchen)
(in_room book bedroom)
(connected kitchen living_room)
(connected living_room kitchen)
(connected living_room bedroom)
(connected bedroom living_room)
```
"""


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def household_agents():
    return [
        AgentSpec(
            name="robot",
            agent_type="robot",
            capabilities=["move between rooms", "pick up objects", "place objects"],
            action_hints=["move", "pick_up", "place"],
        ),
        AgentSpec(
            name="human",
            agent_type="person",
            capabilities=["move between rooms", "pick up objects", "read"],
            action_hints=["move", "pick_up"],
        ),
    ]


@pytest.fixture
def simple_domain(household_agents):
    """A simple pre-built domain for testing without LLM calls."""
    return MADomainSpec(
        name="household",
        types={
            "room": "a location",
            "robot": "a robot agent",
            "person": "a human agent",
            "object": "a manipulable object",
        },
        predicates=[
            {"name": "at", "desc": "robot is in room", "raw": "(at ?a - robot ?r - room)",
             "params": OrderedDict([("a", "robot"), ("r", "room")]), "clean": "(at ?a ?r)"},
            {"name": "at_person", "desc": "person is in room",
             "raw": "(at_person ?p - person ?r - room)",
             "params": OrderedDict([("p", "person"), ("r", "room")]), "clean": "(at_person ?p ?r)"},
            {"name": "in_room", "desc": "object is in room",
             "raw": "(in_room ?o - object ?r - room)",
             "params": OrderedDict([("o", "object"), ("r", "room")]), "clean": "(in_room ?o ?r)"},
            {"name": "holding", "desc": "robot holding object",
             "raw": "(holding ?a - robot ?o - object)",
             "params": OrderedDict([("a", "robot"), ("o", "object")]), "clean": "(holding ?a ?o)"},
            {"name": "connected", "desc": "rooms connected",
             "raw": "(connected ?r1 - room ?r2 - room)",
             "params": OrderedDict([("r1", "room"), ("r2", "room")]), "clean": "(connected ?r1 ?r2)"},
        ],
        agents=household_agents,
        agent_actions={
            "robot": [
                {
                    "name": "move",
                    "desc": "move robot between rooms",
                    "raw": "",
                    "params": OrderedDict([("agent", "robot"), ("from", "room"), ("to", "room")]),
                    "preconditions": "(and (at ?agent ?from) (connected ?from ?to))",
                    "effects": "(and (at ?agent ?to) (not (at ?agent ?from)))",
                },
                {
                    "name": "pick_up",
                    "desc": "pick up an object",
                    "raw": "",
                    "params": OrderedDict([("agent", "robot"), ("obj", "object"), ("room", "room")]),
                    "preconditions": "(and (at ?agent ?room) (in_room ?obj ?room) (not (holding ?agent ?obj)))",
                    "effects": "(and (holding ?agent ?obj) (not (in_room ?obj ?room)))",
                },
            ],
            "human": [],
        },
        concurrent_effects=[
            ConcurrentEffect(
                agent_a="robot",
                action_a="pick_up",
                agent_b="human",
                action_b="pick_up",
                effect_type="conflicting",
                resolution="Neither succeeds.",
            ),
        ],
        requirements=[":typing", ":multi-agent"],
    )


@pytest.fixture
def simple_task():
    """A simple pre-built task for testing."""
    return MATaskSpec(
        name="scenario",
        domain_name="household",
        objects={
            "robot1": "robot",
            "human1": "person",
            "kitchen": "room",
            "living_room": "room",
            "bedroom": "room",
            "cup": "object",
            "book": "object",
        },
        initial_state=[
            "(at robot1 kitchen)",
            "(at_person human1 living_room)",
            "(in_room cup kitchen)",
            "(in_room book bedroom)",
            "(connected kitchen living_room)",
            "(connected living_room kitchen)",
            "(connected living_room bedroom)",
            "(connected bedroom living_room)",
        ],
        agent_objects={
            "robot": ["robot1"],
            "human": ["human1"],
        },
    )


# ===========================================================================
# Step 1: AgentBuilder Tests
# ===========================================================================


class TestAgentBuilder:
    """Tests for agent identification (Step 1)."""

    def test_identify_agents_two_agents(self, mock_llm):
        """AgentBuilder identifies two agents from a household scene."""
        mock_llm.output = f"### AGENTS\n```json\n{AGENTS_JSON}\n```"
        builder = AgentBuilder()
        agents, raw = builder.identify_agents(mock_llm, HOUSEHOLD_SCENE)

        assert len(agents) == 2
        assert agents[0].name == "robot"
        assert agents[1].name == "human"
        assert "move" in agents[0].action_hints
        assert len(agents[0].capabilities) == 3

    def test_identify_agents_json_without_header(self, mock_llm):
        """Parser handles JSON without ### AGENTS header."""
        mock_llm.output = AGENTS_JSON
        builder = AgentBuilder()
        agents, _ = builder.identify_agents(mock_llm, HOUSEHOLD_SCENE)
        assert len(agents) == 2

    def test_identify_agents_retry_on_error(self, mock_llm):
        """AgentBuilder retries on malformed output."""
        mock_llm.set_responses([
            "this is not json",  # First attempt fails
            f"### AGENTS\n```json\n{AGENTS_JSON}\n```",  # Second succeeds
        ])
        builder = AgentBuilder()
        agents, _ = builder.identify_agents(mock_llm, HOUSEHOLD_SCENE, max_retries=3)
        assert len(agents) == 2

    def test_identify_agents_fails_after_max_retries(self, mock_llm):
        """AgentBuilder raises after exhausting retries."""
        mock_llm.output = "not json at all"
        builder = AgentBuilder()
        with pytest.raises(ValueError, match="Failed to identify agents"):
            builder.identify_agents(mock_llm, HOUSEHOLD_SCENE, max_retries=2)

    def test_parse_agents_json_empty_list(self):
        """Empty agent list raises ValueError."""
        with pytest.raises(ValueError, match="LLM returned empty agent list"):
            mock = MockLLM()
            mock.output = "### AGENTS\n```json\n[]\n```"
            AgentBuilder().identify_agents(mock, "some scene")

    def test_parse_agents_json_valid(self):
        """_parse_agents_json correctly parses a JSON array."""
        agents = _parse_agents_json(f"### AGENTS\n```json\n{AGENTS_JSON}\n```")
        assert len(agents) == 2
        assert agents[0].name == "robot"

    def test_parse_agents_json_no_array(self):
        """_parse_agents_json raises on missing JSON array."""
        with pytest.raises(ValueError, match="Could not find JSON array"):
            _parse_agents_json("no json here")


# ===========================================================================
# Step 2: WorldModelDomainBuilder Tests
# ===========================================================================


class TestWorldModelDomainBuilder:
    """Tests for multi-agent domain extraction (Step 2)."""

    def test_identify_concurrent_effects(self, mock_llm, household_agents):
        """Concurrent effects are parsed from LLM output."""
        mock_llm.output = CONCURRENT_EFFECTS_RESPONSE
        builder = WorldModelDomainBuilder()
        effects, _ = builder.identify_concurrent_effects(
            model=mock_llm,
            scene_desc=HOUSEHOLD_SCENE,
            agents=household_agents,
            agent_actions={"robot": [], "human": []},
            predicates=[],
        )
        assert len(effects) == 1
        assert effects[0].effect_type == "conflicting"
        assert effects[0].agent_a == "robot"
        assert effects[0].agent_b == "human"

    def test_identify_concurrent_effects_empty(self, mock_llm, household_agents):
        """Empty concurrent effects list is valid (all commutative)."""
        mock_llm.output = "### CONCURRENT EFFECTS\n```json\n[]\n```"
        builder = WorldModelDomainBuilder()
        effects, _ = builder.identify_concurrent_effects(
            model=mock_llm,
            scene_desc=HOUSEHOLD_SCENE,
            agents=household_agents,
            agent_actions={"robot": [], "human": []},
            predicates=[],
        )
        assert effects == []

    def test_parse_concurrent_effects_no_header(self):
        """No concurrent effects section returns empty list."""
        effects = _parse_concurrent_effects_json("Some text with no effects")
        assert effects == []


# ===========================================================================
# Step 3: WorldModelTaskBuilder Tests
# ===========================================================================


class TestWorldModelTaskBuilder:
    """Tests for goal-free task extraction (Step 3)."""

    def test_parse_objects_block(self):
        """Objects block is parsed correctly."""
        objs = _parse_objects_block(OBJECTS_INITIAL_RESPONSE)
        assert "robot1" in objs
        assert objs["robot1"] == "robot"
        assert "kitchen" in objs
        assert objs["kitchen"] == "room"
        assert len(objs) == 7

    def test_parse_initial_block(self):
        """Initial state block is parsed correctly."""
        initial = _parse_initial_block(OBJECTS_INITIAL_RESPONSE)
        assert "(at robot1 kitchen)" in initial
        assert "(connected kitchen living_room)" in initial
        assert len(initial) == 8

    def test_formalize_objects_and_initial(self, mock_llm, simple_domain):
        """Task extraction produces valid MATaskSpec."""
        mock_llm.output = OBJECTS_INITIAL_RESPONSE
        builder = WorldModelTaskBuilder()
        task, _, (valid, msg) = builder.formalize_objects_and_initial(
            model=mock_llm,
            scene_desc=HOUSEHOLD_SCENE,
            domain=simple_domain,
        )
        assert task.name == "scenario"
        assert "robot1" in task.objects
        assert len(task.initial_state) == 8

    def test_generate_task_pddl_no_goal(self, simple_task):
        """Generated PDDL problem has no :goal section."""
        builder = WorldModelTaskBuilder()
        pddl = builder.generate_task_pddl("household", "test", simple_task)
        assert ":goal" not in pddl
        assert "(define (problem test)" in pddl
        assert "(:domain household)" in pddl
        assert "robot1 - robot" in pddl
        assert "(at robot1 kitchen)" in pddl


# ===========================================================================
# Step 4: MAPddlWriter Tests
# ===========================================================================


class TestMAPddlWriter:
    """Tests for MA-PDDL file writing (Step 4)."""

    def test_write_unfactored_domain(self, simple_domain):
        """Unfactored domain contains all agents' actions."""
        writer = MAPddlWriter()
        pddl = writer.write_domain(simple_domain, factored=False)
        assert "(define (domain household)" in pddl
        assert ":multi-agent" in pddl
        assert ":typing" in pddl
        assert "(:action move" in pddl
        assert "(:action pick_up" in pddl
        # Should have agent comment
        assert "; Agent: robot" in pddl

    def test_write_factored_domain(self, simple_domain):
        """Factored domain produces per-agent files."""
        writer = MAPddlWriter()
        domains = writer.write_domain(simple_domain, factored=True)
        assert isinstance(domains, dict)
        assert "robot" in domains
        assert "human" in domains
        assert "household_robot" in domains["robot"]
        assert "(:action move" in domains["robot"]

    def test_write_unfactored_problem_no_goal(self, simple_task, simple_domain):
        """Problem file has no :goal section."""
        writer = MAPddlWriter()
        pddl = writer.write_problem(simple_task, simple_domain, factored=False)
        assert ":goal" not in pddl
        assert "(define (problem scenario)" in pddl
        assert "(at robot1 kitchen)" in pddl

    def test_write_files_to_disk(self, simple_domain, simple_task):
        """Files are written to disk correctly."""
        writer = MAPddlWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = writer.write_files(simple_domain, simple_task, tmpdir)
            assert len(paths) >= 2  # domain + problem
            # Check domain file exists
            domain_path = os.path.join(tmpdir, "domain.pddl")
            assert os.path.exists(domain_path)
            with open(domain_path) as f:
                content = f.read()
                assert "(define (domain household)" in content

    def test_write_concurrent_effects_sidecar(self, simple_domain, simple_task):
        """Concurrent effects JSON sidecar is written."""
        writer = MAPddlWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = writer.write_files(simple_domain, simple_task, tmpdir)
            effects_path = os.path.join(tmpdir, "concurrent_effects.json")
            assert os.path.exists(effects_path)
            with open(effects_path) as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["effect_type"] == "conflicting"


# ===========================================================================
# Step 5: PddlWorldModel Tests
# ===========================================================================


class TestPddlWorldModelParsing:
    """Tests for PDDL expression parsing utilities."""

    def test_parse_atom(self):
        assert _parse_atom("(at robot kitchen)") == ("at", "robot", "kitchen")
        assert _parse_atom("(holding robot cup)") == ("holding", "robot", "cup")
        assert _parse_atom("not an atom") is None
        assert _parse_atom("()") is None

    def test_parse_conjunction(self):
        expr = "(and (at ?agent ?from) (connected ?from ?to))"
        atoms = _parse_pddl_expression(expr)
        assert len(atoms) == 2
        assert atoms[0] == (True, ("at", "?agent", "?from"))
        assert atoms[1] == (True, ("connected", "?from", "?to"))

    def test_parse_negation(self):
        expr = "(and (at ?agent ?to) (not (at ?agent ?from)))"
        atoms = _parse_pddl_expression(expr)
        assert len(atoms) == 2
        assert atoms[0] == (True, ("at", "?agent", "?to"))
        assert atoms[1] == (False, ("at", "?agent", "?from"))

    def test_parse_simple_atom(self):
        atoms = _parse_pddl_expression("(at robot kitchen)")
        assert len(atoms) == 1
        assert atoms[0] == (True, ("at", "robot", "kitchen"))


class TestPddlWorldModel:
    """Tests for the PDDL-to-WorldModel converter (Step 5)."""

    def test_instantiation(self, simple_domain, simple_task):
        """PddlWorldModel can be instantiated from specs."""
        wm = PddlWorldModel(simple_domain, simple_task)
        assert wm is not None
        assert len(wm._agent_names) == 2

    def test_get_state_is_hashable(self, simple_domain, simple_task):
        """get_state() returns a hashable value."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        state = wm.get_state()
        # Should be usable as dict key
        d = {state: "test"}
        assert d[state] == "test"

    def test_set_state_roundtrip(self, simple_domain, simple_task):
        """set_state(get_state()) preserves the state."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        state1 = wm.get_state()
        wm.set_state(state1)
        state2 = wm.get_state()
        assert state1 == state2

    def test_initial_state_contains_atoms(self, simple_domain, simple_task):
        """Initial state contains the declared initial atoms."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        _, atoms = wm.get_state()
        assert ("at", "robot1", "kitchen") in atoms
        assert ("in_room", "cup", "kitchen") in atoms
        assert ("connected", "kitchen", "living_room") in atoms

    def test_transition_probabilities_sum_to_one(self, simple_domain, simple_task):
        """Transitions sum to 1.0."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        state = wm.get_state()
        # No-op for both agents
        result = wm.transition_probabilities(state, [0, 0])
        assert result is not None
        total_prob = sum(p for p, _ in result)
        assert abs(total_prob - 1.0) < 1e-10

    def test_terminal_returns_none(self, simple_domain, simple_task):
        """Terminal state returns None from transition_probabilities."""
        wm = PddlWorldModel(simple_domain, simple_task, max_steps=0)
        wm.reset()
        state = wm.get_state()
        assert wm.transition_probabilities(state, [0, 0]) is None

    def test_noop_preserves_state(self, simple_domain, simple_task):
        """No-op action (index 0) preserves the state atoms."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        state = wm.get_state()
        result = wm.transition_probabilities(state, [0, 0])
        assert result is not None
        _, succ_state = result[0]
        # Atoms should be same (step count advances)
        step_count, atoms = state
        succ_step, succ_atoms = succ_state
        assert succ_atoms == atoms
        assert succ_step == step_count + 1

    def test_action_changes_state(self, simple_domain, simple_task):
        """An action with satisfied preconditions changes the state."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        state = wm.get_state()
        _, initial_atoms = state

        # Find the "move" action index for robot that moves from kitchen to living_room
        robot_actions = wm._agent_ground_actions["robot"]
        move_idx = None
        for i, ga in enumerate(robot_actions):
            if (
                ga.name == "move"
                and ga.bindings.get("from") == "kitchen"
                and ga.bindings.get("to") == "living_room"
            ):
                move_idx = i + 1  # +1 because 0 is no-op
                break

        if move_idx is not None:
            result = wm.transition_probabilities(state, [move_idx, 0])
            assert result is not None
            _, succ_state = result[0]
            _, succ_atoms = succ_state
            # Robot should have moved
            assert ("at", "robot1", "living_room") in succ_atoms
            assert ("at", "robot1", "kitchen") not in succ_atoms

    def test_precondition_failure_noop(self, simple_domain, simple_task):
        """Action with unsatisfied preconditions is a no-op."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        state = wm.get_state()

        # Try to move from bedroom to kitchen (robot is not in bedroom)
        robot_actions = wm._agent_ground_actions["robot"]
        bad_move_idx = None
        for i, ga in enumerate(robot_actions):
            if (
                ga.name == "move"
                and ga.bindings.get("from") == "bedroom"
                and ga.bindings.get("to") == "living_room"
            ):
                bad_move_idx = i + 1
                break

        if bad_move_idx is not None:
            result = wm.transition_probabilities(state, [bad_move_idx, 0])
            assert result is not None
            _, succ_state = result[0]
            _, succ_atoms = succ_state
            _, initial_atoms = state
            # State should be unchanged (action was no-op)
            assert succ_atoms == initial_atoms

    def test_gymnasium_reset(self, simple_domain, simple_task):
        """reset() returns a valid observation."""
        wm = PddlWorldModel(simple_domain, simple_task)
        obs, info = wm.reset()
        assert obs is not None
        assert obs.shape == (wm._num_atoms,)
        assert info == {}

    def test_gymnasium_step(self, simple_domain, simple_task):
        """step() advances the state."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        obs, reward, terminated, truncated, info = wm.step(0)
        assert obs is not None
        assert reward == 0.0
        assert not terminated

    def test_human_agent_indices(self, simple_domain, simple_task):
        """human_agent_indices returns the correct index."""
        wm = PddlWorldModel(simple_domain, simple_task)
        # "human" has type "person" which should match
        indices = wm.human_agent_indices
        assert 1 in indices  # human is the second agent

    def test_action_grounding_count(self, simple_domain, simple_task):
        """Action grounding produces the correct number of actions."""
        wm = PddlWorldModel(simple_domain, simple_task)
        robot_actions = wm._agent_ground_actions["robot"]
        # move: 1 robot × 3 rooms(from) × 3 rooms(to) = 9 groundings
        # pick_up: 1 robot × 2 objects × 3 rooms = 6 groundings
        # Total: 15 grounded actions for robot
        assert len(robot_actions) == 15

    def test_add_state(self, simple_domain, simple_task):
        """add_state registers a new known state."""
        wm = PddlWorldModel(simple_domain, simple_task)
        new_atoms = frozenset({("at", "robot1", "bedroom")})
        wm.add_state(new_atoms)
        assert new_atoms in wm._known_states

    def test_add_transition(self, simple_domain, simple_task):
        """add_transition registers explicit transitions."""
        wm = PddlWorldModel(simple_domain, simple_task)
        wm.reset()
        state = wm.get_state()
        _, atoms = state

        new_atoms = frozenset({("at", "robot1", "bedroom")})
        wm.add_transition(atoms, (1, 0), [(1.0, new_atoms)])

        # The explicit transition should be used
        result = wm.transition_probabilities(state, [1, 0])
        assert result is not None
        assert result[0][1] == (1, new_atoms)

    def test_add_object(self, simple_domain, simple_task):
        """add_object extends the object set and re-grounds actions."""
        wm = PddlWorldModel(simple_domain, simple_task)
        orig_atom_count = wm._num_atoms
        wm.add_object("plate", "object")
        # Should have more ground atoms now
        assert wm._num_atoms >= orig_atom_count

    def test_step_time_limit_is_truncation(self, simple_domain, simple_task):
        """step() returns truncated=True (not terminated) on time limit."""
        wm = PddlWorldModel(simple_domain, simple_task, max_steps=1)
        wm.reset()
        # First step succeeds
        obs, reward, terminated, truncated, info = wm.step(0)
        assert not terminated
        # Second step hits the time limit
        obs, reward, terminated, truncated, info = wm.step(0)
        assert truncated
        assert not terminated

    def test_agent_grounding_respects_ownership(self, simple_domain, simple_task):
        """Agent-typed parameters only ground to agent-owned objects."""
        wm = PddlWorldModel(simple_domain, simple_task)
        robot_actions = wm._agent_ground_actions["robot"]
        # All robot move actions should bind the agent param to robot1
        for ga in robot_actions:
            if ga.name == "move":
                assert ga.bindings.get("agent") == "robot1"


# ===========================================================================
# Step 7: WorldModelBuilder Tests
# ===========================================================================


class TestWorldModelBuilder:
    """Tests for the end-to-end pipeline (Step 7)."""

    def test_build_produces_world_model(self, mock_llm):
        """Full build pipeline produces a PddlWorldModel."""
        # Set up responses in order: identify_agents, formalize_types,
        # formalize_predicates, formalize_agent_actions (x5 hints),
        # identify_concurrent_effects, formalize_objects_and_initial
        mock_llm.set_responses([
            # 1. identify_agents
            f"### AGENTS\n```json\n{AGENTS_JSON}\n```",
            # 2. formalize_types
            TYPES_RESPONSE,
            # 3. formalize_shared_predicates (wraps formalize_predicates)
            PREDICATES_RESPONSE,
            # 4. robot: move action
            MOVE_ACTION_RESPONSE,
            # 5. robot: pick_up action
            PICK_UP_ACTION_RESPONSE,
            # 6. robot: place action
            PLACE_ACTION_RESPONSE,
            # 7. human: move action
            MOVE_ACTION_RESPONSE,
            # 8. human: pick_up action
            PICK_UP_ACTION_RESPONSE,
            # 9. identify_concurrent_effects
            CONCURRENT_EFFECTS_RESPONSE,
            # 10. formalize_objects_and_initial
            OBJECTS_INITIAL_RESPONSE,
        ])

        builder = WorldModelBuilder(llm=mock_llm, max_steps=10)
        wm = builder.build(HOUSEHOLD_SCENE)

        assert isinstance(wm, PddlWorldModel)
        wm.reset()
        state = wm.get_state()
        assert state is not None

        # State should be hashable
        d = {state: True}
        assert d[state] is True

        # transition_probabilities should work
        result = wm.transition_probabilities(state, [0, 0])
        assert result is not None
        assert abs(sum(p for p, _ in result) - 1.0) < 1e-10
