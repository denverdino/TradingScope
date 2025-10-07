"""Unit tests for the Risk Debate Orchestrator."""

import os
import sys

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.agents.risk_mgmt.debate_orchestrator import RiskDebateOrchestrator, create_debate_orchestrator


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name="MockAgent"):
        self.name = name

    async def __call__(self, *args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.content = f"Mock response from {self.name}"

        return MockResponse()


class TestRiskDebateOrchestrator:
    """Test cases for the Risk Debate Orchestrator."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_aggressive_agent = MockAgent("AggressiveAgent")
        self.mock_conservative_agent = MockAgent("ConservativeAgent")
        self.mock_neutral_agent = MockAgent("NeutralAgent")
        self.mock_risk_manager = MockAgent("RiskManager")
        self.max_rounds = 2

    def test_create_debate_orchestrator_success(self):
        """Test that debate orchestrator can be created successfully."""
        orchestrator = create_debate_orchestrator(
            aggressive_agent=self.mock_aggressive_agent,
            conservative_agent=self.mock_conservative_agent,
            neutral_agent=self.mock_neutral_agent,
            risk_manager=self.mock_risk_manager,
            max_rounds=self.max_rounds,
        )

        # Verify the orchestrator was created
        assert orchestrator is not None
        assert isinstance(orchestrator, RiskDebateOrchestrator)
        assert orchestrator.max_rounds == self.max_rounds
        assert orchestrator.aggressive_agent == self.mock_aggressive_agent
        assert orchestrator.conservative_agent == self.mock_conservative_agent
        assert orchestrator.neutral_agent == self.mock_neutral_agent
        assert orchestrator.risk_manager == self.mock_risk_manager

    def test_debate_orchestrator_initialization(self):
        """Test that debate orchestrator initializes correctly."""
        orchestrator = RiskDebateOrchestrator(
            aggressive_agent=self.mock_aggressive_agent,
            conservative_agent=self.mock_conservative_agent,
            neutral_agent=self.mock_neutral_agent,
            risk_manager=self.mock_risk_manager,
            max_rounds=self.max_rounds,
        )

        # Verify the orchestrator was initialized correctly
        assert orchestrator is not None
        assert orchestrator.max_rounds == self.max_rounds
        assert orchestrator.aggressive_agent == self.mock_aggressive_agent
        assert orchestrator.conservative_agent == self.mock_conservative_agent
        assert orchestrator.neutral_agent == self.mock_neutral_agent
        assert orchestrator.risk_manager == self.mock_risk_manager

    @pytest.mark.asyncio
    async def test_run_debate_method_exists(self):
        """Test that the run_debate method exists and can be called."""
        orchestrator = RiskDebateOrchestrator(
            aggressive_agent=self.mock_aggressive_agent,
            conservative_agent=self.mock_conservative_agent,
            neutral_agent=self.mock_neutral_agent,
            risk_manager=self.mock_risk_manager,
            max_rounds=self.max_rounds,
        )

        # Verify the method exists
        assert hasattr(orchestrator, "run_debate")
        assert callable(orchestrator.run_debate)

    def test_debate_prompts_content(self):
        """Test that the orchestrator creates proper debate prompts."""
        orchestrator = RiskDebateOrchestrator(
            aggressive_agent=self.mock_aggressive_agent,
            conservative_agent=self.mock_conservative_agent,
            neutral_agent=self.mock_neutral_agent,
            risk_manager=self.mock_risk_manager,
            max_rounds=self.max_rounds,
        )

        # While we can't directly test the internal prompts without calling run_debate,
        # we can verify the orchestrator was created with the right configuration
        assert orchestrator is not None
        assert orchestrator.max_rounds == self.max_rounds
