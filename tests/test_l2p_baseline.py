"""
Baseline test that L2P vendor is properly set up and importable.

Step 0 of the LLM WorldModel plan: verify that the vendored L2P library
works with MockLLM for offline testing.
"""

import pytest


class TestL2PBaseline:
    """Verify L2P vendor setup and basic functionality."""

    def test_l2p_core_imports(self):
        """L2P core classes are importable."""
        from l2p import DomainBuilder, TaskBuilder

        assert DomainBuilder is not None
        assert TaskBuilder is not None

    def test_l2p_type_imports(self):
        """L2P PDDL type definitions are importable."""
        from l2p.utils.pddl_types import Predicate, Action, Function

        assert Predicate is not None
        assert Action is not None
        assert Function is not None

    def test_l2p_llm_imports(self):
        """L2P LLM abstraction is importable."""
        from l2p.llm.base import BaseLLM

        assert BaseLLM is not None

    def test_l2p_validator_imports(self):
        """L2P syntax validator is importable."""
        from l2p.utils.pddl_validator import SyntaxValidator

        assert SyntaxValidator is not None

    def test_l2p_parser_imports(self):
        """L2P PDDL parser utilities are importable."""
        from l2p.utils.pddl_parser import parse_types, parse_predicates

        assert parse_types is not None
        assert parse_predicates is not None

    def test_mock_llm_instantiation(self):
        """MockLLM can be instantiated and returns canned responses."""
        # Use MockLLM from L2P's test suite
        from l2p.llm.base import BaseLLM

        class MockLLM(BaseLLM):
            """Minimal mock LLM for testing without API calls."""

            def __init__(self):
                # Skip BaseLLM.__init__ which validates model names
                self.output = ""

            def query(self, prompt: str) -> str:
                return self.output

            def valid_models(self):
                return ["mock"]

        mock = MockLLM()
        mock.output = "test response"
        assert mock.query("any prompt") == "test response"

    def test_domain_builder_instantiation(self):
        """DomainBuilder can be instantiated with default parameters."""
        from l2p import DomainBuilder

        builder = DomainBuilder()
        assert builder.types == {}
        assert builder.predicates == []
        assert builder.pddl_actions == []

    def test_task_builder_instantiation(self):
        """TaskBuilder can be instantiated with default parameters."""
        from l2p import TaskBuilder

        builder = TaskBuilder()
        assert builder.objects == {}
        assert builder.initial == []

    def test_syntax_validator_instantiation(self):
        """SyntaxValidator can be instantiated."""
        from l2p.utils.pddl_validator import SyntaxValidator

        validator = SyntaxValidator()
        assert validator is not None
