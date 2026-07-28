#!/usr/bin/env python3
"""
Tests for the simple_hierarchical_llm_modeler subpackage.

Uses a MockLLM that returns deterministic JSON responses, so no real LLM calls
are made.
"""

import json
import math
import sys

import gymnasium as gym
import pytest

sys.modules.setdefault("gym", gym)

from empo.world_model import WorldModel
from empo.simple_hierarchical_llm_modeler.llm_connector import (
    LLMConnector,
    L2PConnector,
)
from empo.simple_hierarchical_llm_modeler.tree_builder import (
    build_tree,
    collect_leaves,
    count_nodes,
    _parse_json_list,
    _parse_json_object,
)
from empo.simple_hierarchical_llm_modeler.nl_world_model import NLWorldModel
from empo.simple_hierarchical_llm_modeler.hierarchical_modeler import (
    LazyTwoLevelModel,
    NLLevelMapper,
    build_two_level_model,
    check_hierarchical_status,
    match_consequence,
)
from empo.simple_hierarchical_llm_modeler.prompts import (
    robot_actions_prompt,
    humans_reactions_prompt,
    consequences_prompt,
    empowerment_prompt,
    hierarchical_status_prompt,
    match_consequence_prompt,
)

# ============================================================================
# Mock LLM
# ============================================================================


class MockLLM:
    """Deterministic LLM that returns canned JSON based on prompt keywords."""

    def __init__(self):
        self.call_count = 0
        self.prompts: list[str] = []

    def query(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)

        # --- Robot actions ---
        if "action options the robot" in prompt:
            n = _extract_n(prompt, "name")
            actions = [
                {"action": f"robot_action_{i}", "rationale": f"rationale_{i}"}
                for i in range(n)
            ]
            return json.dumps(actions)

        # --- Humans reactions ---
        if "things that the humans" in prompt:
            n = _extract_n(prompt, "name")
            reactions = [
                {"reaction": f"human_reaction_{i}", "rationale": f"rationale_{i}"}
                for i in range(n)
            ]
            return json.dumps(reactions)

        # --- Consequences ---
        if "consequences and their probabilities" in prompt:
            n = _extract_n(prompt, "name")
            prob = 1.0 / n if n > 0 else 1.0
            consequences = [
                {
                    "consequence": f"consequence_{i}",
                    "probability": round(prob, 4),
                    "rationale": f"rationale_{i}",
                }
                for i in range(n)
            ]
            return json.dumps(consequences)

        # --- Empowerment estimate ---
        if "meaningfully different futures" in prompt:
            return json.dumps({"estimate": 8, "rationale": "moderate empowerment"})

        # --- Hierarchical status ---
        if "success of the higher-level" in prompt:
            return json.dumps({"status": "still in progress"})

        # --- Consequence matching ---
        if "correspond to one of these" in prompt:
            return json.dumps({"match": 1, "new_consequence": None})

        # Fallback
        return json.dumps({"error": "unknown prompt type"})


def _extract_n(prompt: str, keyword: str) -> int:
    """Extract the number N from 'name N distinct ...' in the prompt."""
    import re

    m = re.search(rf"{keyword}\s+(\d+)", prompt)
    return int(m.group(1)) if m else 2


# ============================================================================
# Tests: LLM connector protocol
# ============================================================================


class TestLLMConnector:
    def test_mock_satisfies_protocol(self):
        mock = MockLLM()
        assert isinstance(mock, LLMConnector)

    def test_l2p_connector_delegates(self):
        class FakeBaseLLM:
            def query(self, prompt):
                return "hello from l2p"

        conn = L2PConnector(FakeBaseLLM())
        assert conn.query("test") == "hello from l2p"


# ============================================================================
# Tests: JSON parsing helpers
# ============================================================================


class TestJSONParsing:
    def test_parse_json_list_simple(self):
        result = _parse_json_list('[{"a": 1}, {"a": 2}]')
        assert result == [{"a": 1}, {"a": 2}]

    def test_parse_json_list_with_preamble(self):
        result = _parse_json_list('Here is the list: [{"a": 1}] done.')
        assert result == [{"a": 1}]

    def test_parse_json_list_error(self):
        with pytest.raises(ValueError, match="No JSON list"):
            _parse_json_list("no json here")

    def test_parse_json_object_simple(self):
        result = _parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_object_with_preamble(self):
        result = _parse_json_object('Result: {"key": 42} end')
        assert result == {"key": 42}

    def test_parse_json_object_error(self):
        with pytest.raises(ValueError, match="No JSON object"):
            _parse_json_object("nothing here")


# ============================================================================
# Tests: Prompt generation
# ============================================================================


class TestPrompts:
    def test_robot_actions_prompt_basic(self):
        p = robot_actions_prompt("At the airport", ["event1"], 3)
        assert "action options the robot" in p
        assert "3 distinct" in p
        assert "event1" in p

    def test_robot_actions_prompt_with_context(self):
        p = robot_actions_prompt("At the airport", [], 2, "We are in a city")
        assert "Higher-level context" in p

    def test_humans_reactions_prompt(self):
        p = humans_reactions_prompt("At the park", ["Robot: offer help"], 2)
        assert "things that the humans" in p
        assert "2 distinct" in p

    def test_consequences_prompt(self):
        p = consequences_prompt("At the park", ["Robot: act", "Humans: react"], 3)
        assert "consequences and their probabilities" in p
        assert "3 distinct" in p

    def test_empowerment_prompt(self):
        p = empowerment_prompt("At the park", ["some event"])
        assert "meaningfully different futures" in p

    def test_hierarchical_status_prompt(self):
        p = hierarchical_status_prompt(
            "Building a house", "build the roof", "At the site", ["Got materials"]
        )
        assert "success of the higher-level" in p
        assert "build the roof" in p

    def test_match_consequence_prompt(self):
        p = match_consequence_prompt(
            "Building a house",
            "build the roof",
            ["roof complete", "roof collapsed"],
            "success",
            "At the site",
            ["Built the roof"],
        )
        assert "correspond to one of these" in p
        assert "roof complete" in p


# ============================================================================
# Tests: Tree builder
# ============================================================================


class TestTreeBuilder:
    def test_build_tree_depth_0(self):
        """Depth 0 = just the root state with empowerment estimate."""
        llm = MockLLM()
        root = build_tree(llm, "Start", n_steps=0)
        assert root.node_type == "state"
        assert root.depth == 0
        assert root.empowerment_estimate is not None
        assert root.empowerment_estimate == 8.0
        assert len(root.children) == 0

    def test_build_tree_depth_1(self):
        """Depth 1: root -> robotactions -> humansreactions -> consequences -> terminal states."""
        llm = MockLLM()
        root = build_tree(
            llm,
            "Airport scenario",
            n_steps=1,
            n_robotactions=2,
            n_humansreactions=2,
            n_consequences=2,
        )
        assert root.node_type == "state"
        assert len(root.children) == 2  # 2 robot actions

        for _, _, ra_node in root.children:
            assert ra_node.node_type == "robotaction"
            assert len(ra_node.children) == 2  # 2 humans reactions

            for _, _, hr_node in ra_node.children:
                assert hr_node.node_type == "humansreaction"
                assert len(hr_node.children) == 2  # 2 consequences

                for _, prob, cons_node in hr_node.children:
                    assert cons_node.node_type == "state"
                    assert cons_node.depth == 1
                    assert cons_node.empowerment_estimate == 8.0
                    assert 0 < prob <= 1.0

    def test_count_nodes(self):
        llm = MockLLM()
        root = build_tree(
            llm,
            "Test",
            n_steps=1,
            n_robotactions=2,
            n_humansreactions=2,
            n_consequences=2,
        )
        # 1 root + 2 ra + 2*2 hr + 2*2*2 cons = 1 + 2 + 4 + 8 = 15
        assert count_nodes(root) == 15

    def test_collect_leaves(self):
        llm = MockLLM()
        root = build_tree(
            llm,
            "Test",
            n_steps=1,
            n_robotactions=2,
            n_humansreactions=2,
            n_consequences=2,
        )
        leaves = collect_leaves(root)
        assert len(leaves) == 8  # 2*2*2 terminal states
        for leaf in leaves:
            assert leaf.empowerment_estimate is not None

    def test_history_grows(self):
        llm = MockLLM()
        root = build_tree(
            llm,
            "Test",
            n_steps=1,
            n_robotactions=1,
            n_humansreactions=1,
            n_consequences=1,
        )
        leaf = collect_leaves(root)[0]
        assert len(leaf.history) == 3  # Robot:..., Humans:..., Observation:...
        assert leaf.history[0].startswith("Robot:")
        assert leaf.history[1].startswith("Humans:")
        assert leaf.history[2].startswith("Observation:")

    def test_fresh_context_per_call(self):
        """Each LLM call should be independent (fresh context)."""
        llm = MockLLM()
        build_tree(
            llm,
            "Test",
            n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        # We just verify no errors and that multiple calls were made
        assert llm.call_count > 1


# ============================================================================
# Tests: NLWorldModel
# ============================================================================


class TestNLWorldModel:
    def _make_model(self, n_steps=1, n_ra=2, n_hr=2, n_cons=2):
        llm = MockLLM()
        root = build_tree(
            llm,
            "Airport",
            n_steps=n_steps,
            n_robotactions=n_ra,
            n_humansreactions=n_hr,
            n_consequences=n_cons,
        )
        return NLWorldModel.from_tree(root, "Airport")

    def test_is_world_model(self):
        model = self._make_model()
        assert isinstance(model, WorldModel)

    def test_get_set_state(self):
        model = self._make_model()
        s0 = model.get_state()
        assert s0 == ()  # root = empty history
        model.set_state(("Robot: robot_action_0",))
        assert model.get_state() == ("Robot: robot_action_0",)
        model.set_state(s0)
        assert model.get_state() == s0

    def test_reset(self):
        model = self._make_model()
        model.set_state(("some", "state"))
        obs, info = model.reset()
        assert obs == ()
        assert model.get_state() == ()

    def test_transition_probabilities_root(self):
        model = self._make_model(n_ra=2, n_hr=2, n_cons=2)
        s0 = model.get_state()
        # action profile = (robot_action_index, humans_reaction_index)
        trans = model.transition_probabilities(s0, [0, 0])
        assert trans is not None
        assert len(trans) == 2  # 2 consequences
        total_prob = sum(p for p, _ in trans)
        assert abs(total_prob - 1.0) < 1e-6

    def test_transition_probabilities_terminal(self):
        model = self._make_model(n_steps=1, n_ra=1, n_hr=1, n_cons=1)
        # Navigate to terminal state
        s0 = model.get_state()
        trans = model.transition_probabilities(s0, [0, 0])
        assert trans is not None
        _, next_state = trans[0]
        # Terminal state should have no transitions
        assert model.transition_probabilities(next_state, [0, 0]) is None

    def test_V_r_estimate_terminal(self):
        model = self._make_model(n_steps=1, n_ra=1, n_hr=1, n_cons=1)
        terminal = model.terminal_states
        assert len(terminal) > 0
        for ts in terminal:
            v = model.V_r_estimate(ts)
            assert v == math.log2(8)  # MockLLM returns estimate=8

    def test_V_r_estimate_non_terminal(self):
        model = self._make_model()
        v = model.V_r_estimate(())  # root is not terminal
        assert v == 0.0

    def test_state_description(self):
        model = self._make_model()
        desc = model.state_description()
        assert "Airport" in desc

    def test_robot_action_labels(self):
        model = self._make_model(n_ra=3)
        labels = model.robot_action_labels()
        assert len(labels) == 3
        assert all("robot_action" in label for label in labels)

    def test_humans_reaction_labels(self):
        model = self._make_model(n_hr=2)
        labels = model.humans_reaction_labels(robot_action_index=0)
        assert len(labels) == 2

    def test_states_list(self):
        model = self._make_model(n_ra=1, n_hr=1, n_cons=1)
        states = model.states
        assert len(states) >= 2  # at least root + 1 terminal

    def test_step_follows_transitions(self):
        model = self._make_model(n_ra=1, n_hr=1, n_cons=1)
        model.reset()
        obs, reward, terminated, truncated, info = model.step([0, 0])
        assert obs != ()  # should have moved to a new state

    def test_human_robot_agent_indices(self):
        model = self._make_model()
        assert model.human_agent_indices == [1]
        assert model.robot_agent_indices == [0]


# ============================================================================
# Tests: WorldModel base V_r_estimate default
# ============================================================================


class TestWorldModelVrEstimate:
    def test_base_returns_zero(self):
        """The default V_r_estimate on WorldModel returns 0.0."""

        class MinimalModel(WorldModel):
            def __init__(self):
                super().__init__()
                self._s = 0
                self.agents = [0]
                self.action_space = type("Space", (), {"n": 1})()

            def get_state(self):
                return self._s

            def set_state(self, state):
                self._s = state

            def transition_probabilities(self, state, actions):
                return None

        m = MinimalModel()
        assert m.V_r_estimate(0) == 0.0
        assert m.V_r_estimate(42) == 0.0


# ============================================================================
# Tests: Hierarchical helpers
# ============================================================================


class TestHierarchicalHelpers:
    def test_check_status_still_in_progress(self):
        llm = MockLLM()
        status = check_hierarchical_status(
            llm, "Building house", "lay foundation", "At site", ["Got tools"]
        )
        assert status == "still in progress"

    def test_check_status_null_value_defaults(self):
        """If the LLM returns {"status": null}, default to 'still in progress'."""

        class NullStatusLLM:
            def query(self, prompt: str) -> str:
                return json.dumps({"status": None})

        status = check_hierarchical_status(NullStatusLLM(), "ctx", "act", "state", [])
        assert status == "still in progress"

    def test_check_status_non_string_defaults(self):
        """If the LLM returns {"status": 42}, default to 'still in progress'."""

        class IntStatusLLM:
            def query(self, prompt: str) -> str:
                return json.dumps({"status": 42})

        status = check_hierarchical_status(IntStatusLLM(), "ctx", "act", "state", [])
        assert status == "still in progress"

    def test_match_consequence_returns_match(self):
        llm = MockLLM()
        idx, new = match_consequence(
            llm,
            "Building house",
            "lay foundation",
            ["foundation done", "foundation cracked"],
            "success",
            "At site",
            ["Laid the foundation"],
        )
        # MockLLM returns match=1, which is 1-based -> 0-based = 0
        assert idx == 0
        assert new is None

    def test_match_consequence_out_of_range_treated_as_unmatched(self):
        """An out-of-range 1-based index from the LLM is treated as no match."""

        class OutOfRangeLLM:
            def query(self, prompt: str) -> str:
                # Return match=99, way beyond the list length
                return json.dumps({"match": 99, "new_consequence": None})

        llm = OutOfRangeLLM()
        idx, new = match_consequence(
            llm,
            "ctx",
            "action",
            ["only one consequence"],
            "success",
            "state",
            [],
        )
        assert idx is None
        assert new is not None  # Falls through to novel outcome

    def test_match_consequence_zero_index_treated_as_unmatched(self):
        """A 0 (invalid 1-based) index from the LLM is treated as no match."""

        class ZeroIndexLLM:
            def query(self, prompt: str) -> str:
                return json.dumps({"match": 0, "new_consequence": None})

        llm = ZeroIndexLLM()
        idx, new = match_consequence(
            llm, "ctx", "action", ["a", "b"], "failure", "state", []
        )
        assert idx is None
        assert new is not None


# ============================================================================
# Tests: NLLevelMapper
# ============================================================================


class TestNLLevelMapper:
    def _make_models(self):
        llm = MockLLM()
        root_c = build_tree(
            llm,
            "Coarse",
            n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        root_f = build_tree(
            llm,
            "Fine",
            n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        return (
            NLWorldModel.from_tree(root_c, "Coarse"),
            NLWorldModel.from_tree(root_f, "Fine"),
        )

    def test_is_level_mapper(self):
        from empo.hierarchical.level_mapper import LevelMapper

        coarse, fine = self._make_models()
        mapper = NLLevelMapper(coarse, fine)
        assert isinstance(mapper, LevelMapper)

    def test_super_state_root_maps_to_root(self):
        coarse, fine = self._make_models()
        mapper = NLLevelMapper(coarse, fine)
        assert mapper.super_state(()) == ()

    def test_super_state_in_progress_without_llm(self):
        """Without an LLM, super_state returns 'still in progress' -> root."""
        coarse, fine = self._make_models()
        mapper = NLLevelMapper(coarse, fine)  # no llm
        assert mapper.super_state(("any", "state")) == ()

    def test_super_state_with_llm_still_in_progress(self):
        """With LLM returning 'still in progress', stays at coarse root."""
        coarse, fine = self._make_models()
        llm = MockLLM()  # MockLLM returns "still in progress" for status
        mapper = NLLevelMapper(
            coarse,
            fine,
            llm=llm,
            higher_level_context="ctx",
            higher_level_action="act",
            initial_state_description="desc",
        )
        assert mapper.super_state(("Robot: do something",)) == ()

    def test_super_state_switches_on_success(self):
        """On 'success', super_state returns a distinct terminal marker."""

        class SuccessLLM:
            def query(self, prompt: str) -> str:
                return json.dumps({"status": "success"})

        coarse, fine = self._make_models()
        mapper = NLLevelMapper(
            coarse,
            fine,
            llm=SuccessLLM(),
            higher_level_context="ctx",
            higher_level_action="act",
            initial_state_description="desc",
        )
        result = mapper.super_state(("Robot: done",))
        assert result != ()
        assert result == ("_completed", "success")

    def test_return_control_false_while_in_progress(self):
        coarse, fine = self._make_models()
        llm = MockLLM()  # returns "still in progress"
        mapper = NLLevelMapper(
            coarse,
            fine,
            llm=llm,
            higher_level_context="ctx",
            higher_level_action="act",
            initial_state_description="desc",
        )
        assert (
            mapper.return_control((0,), ("s",), (1,), ("Robot: still going",)) is False
        )

    def test_return_control_true_on_success(self):
        class SuccessLLM:
            def query(self, prompt: str) -> str:
                return json.dumps({"status": "success"})

        coarse, fine = self._make_models()
        mapper = NLLevelMapper(
            coarse,
            fine,
            llm=SuccessLLM(),
            higher_level_context="ctx",
            higher_level_action="act",
            initial_state_description="desc",
        )
        assert mapper.return_control((0,), ("s",), (1,), ("Robot: done",)) is True

    def test_return_control_true_on_failure(self):
        class FailureLLM:
            def query(self, prompt: str) -> str:
                return json.dumps({"status": "failure"})

        coarse, fine = self._make_models()
        mapper = NLLevelMapper(
            coarse,
            fine,
            llm=FailureLLM(),
            higher_level_context="ctx",
            higher_level_action="act",
            initial_state_description="desc",
        )
        assert mapper.return_control((0,), ("s",), (1,), ("Robot: failed",)) is True

    def test_is_feasible_always_true(self):
        coarse, fine = self._make_models()
        mapper = NLLevelMapper(coarse, fine)
        assert mapper.is_feasible((0,), ("s",), (1,)) is True

    def test_is_abort_always_false(self):
        coarse, fine = self._make_models()
        mapper = NLLevelMapper(coarse, fine)
        assert mapper.is_abort((0,), ("s",), (1,)) is False


# ============================================================================
# Tests: Two-level model builder
# ============================================================================


class TestTwoLevelModel:
    def test_build_produces_lazy_model(self):
        llm = MockLLM()
        hmodel = build_two_level_model(
            llm,
            "Airport taxi scenario",
            coarse_n_steps=1,
            fine_n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        assert isinstance(hmodel, LazyTwoLevelModel)
        assert hmodel.num_levels == 2
        assert isinstance(hmodel.coarsest(), NLWorldModel)
        # No fine model built yet
        assert hmodel.finest() is None

    def test_coarse_has_transitions(self):
        llm = MockLLM()
        hmodel = build_two_level_model(
            llm,
            "Airport taxi scenario",
            coarse_n_steps=1,
            fine_n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        coarse = hmodel.coarsest()
        assert isinstance(coarse, NLWorldModel)
        trans = coarse.transition_probabilities(coarse.get_state(), [0, 0])
        assert trans is not None

    def test_fine_built_lazily_on_get(self):
        llm = MockLLM()
        hmodel = build_two_level_model(
            llm,
            "Airport taxi scenario",
            coarse_n_steps=1,
            fine_n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        # Get action labels from coarse model
        coarse = hmodel.coarsest()
        labels = coarse.robot_action_labels()
        assert len(labels) > 0

        # Lazily build fine model for the first action
        fine, mapper = hmodel.get_fine_model(labels[0])
        assert isinstance(fine, NLWorldModel)
        assert isinstance(mapper, NLLevelMapper)
        assert hmodel.finest() is fine

        # Fine model has transitions
        trans = fine.transition_probabilities(fine.get_state(), [0, 0])
        assert trans is not None

    def test_fine_cached_across_calls(self):
        llm = MockLLM()
        hmodel = build_two_level_model(
            llm,
            "Airport taxi scenario",
            coarse_n_steps=1,
            fine_n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        label = hmodel.coarsest().robot_action_labels()[0]
        fine1, mapper1 = hmodel.get_fine_model(label)
        fine2, mapper2 = hmodel.get_fine_model(label)
        assert fine1 is fine2
        assert mapper1 is mapper2

    def test_to_hierarchical_after_fine_built(self):
        from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel

        llm = MockLLM()
        hmodel = build_two_level_model(
            llm,
            "Airport taxi scenario",
            coarse_n_steps=1,
            fine_n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        label = hmodel.coarsest().robot_action_labels()[0]
        hmodel.get_fine_model(label)
        hwm = hmodel.to_hierarchical()
        assert isinstance(hwm, HierarchicalWorldModel)
        assert hwm.num_levels == 2

    def test_to_hierarchical_raises_without_fine(self):
        llm = MockLLM()
        hmodel = build_two_level_model(
            llm,
            "Airport taxi scenario",
            coarse_n_steps=1,
            fine_n_steps=1,
            n_robotactions=2,
            n_humansreactions=1,
            n_consequences=1,
        )
        with pytest.raises(RuntimeError, match="No fine model"):
            hmodel.to_hierarchical()


# ============================================================================
# Tests: Package exports
# ============================================================================


class TestExports:
    def test_imports_from_subpackage(self):
        import empo.simple_hierarchical_llm_modeler as pkg

        assert hasattr(pkg, "LLMConnector")
        assert hasattr(pkg, "L2PConnector")
        assert hasattr(pkg, "TreeNode")
        assert hasattr(pkg, "build_tree")
        assert hasattr(pkg, "count_nodes")
        assert hasattr(pkg, "collect_leaves")
        assert hasattr(pkg, "NLWorldModel")
        assert hasattr(pkg, "LazyTwoLevelModel")
        assert hasattr(pkg, "NLLevelMapper")
        assert hasattr(pkg, "build_two_level_model")
        assert hasattr(pkg, "check_hierarchical_status")
        assert hasattr(pkg, "match_consequence")

    def test_nlworldmodel_is_worldmodel(self):
        from empo.simple_hierarchical_llm_modeler import NLWorldModel

        assert issubclass(NLWorldModel, WorldModel)
