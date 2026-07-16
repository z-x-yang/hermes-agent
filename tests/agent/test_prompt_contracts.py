"""Tests for the fork's always-on behavioral contracts (agent/prompt_contracts.py).

These are the anti-erosion guard for the fork's prompt customizations: an
upstream port that reverts the injection lines in system_prompt.py (as the
0.18 port once did to a fork one-liner) fails here instead of shipping.
Every block is asserted present, gated, ordered, and byte-stable.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.prompt_contracts import (
    ASSESSMENT_FIRST_GUIDANCE,
    COMMUNICATION_GUIDANCE,
    CONTEXT_CONTINUITY_NOTE,
    MEMORY_READBACK_NOTE,
    OBSERVED_CONTENT_BOUNDARY,
    SIDE_EFFECT_CONFIRMATION_GUIDANCE,
    USER_PRECEDENCE_NOTE,
)
from agent.prompt_builder import TASK_COMPLETION_GUIDANCE
from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=["terminal", "read_file"],
        _task_completion_guidance=True,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _parts(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
        patch("agent.coding_context.coding_system_blocks", return_value=[]),
    ):
        return build_system_prompt_parts(agent)


def _stable(agent):
    return _parts(agent)["stable"]


ALL_CONTRACT_BLOCKS = (
    COMMUNICATION_GUIDANCE,
    ASSESSMENT_FIRST_GUIDANCE,
    SIDE_EFFECT_CONFIRMATION_GUIDANCE,
    OBSERVED_CONTENT_BOUNDARY,
    CONTEXT_CONTINUITY_NOTE,
    USER_PRECEDENCE_NOTE,
)


class TestContractPresence:
    def test_all_blocks_injected_with_tools_by_default(self):
        # _behavior_contracts intentionally absent from the agent — the
        # getattr default must be True so agents built outside agent_init
        # (tests, forks) still carry the contracts.
        stable = _stable(_make_agent())
        for block in ALL_CONTRACT_BLOCKS:
            assert block in stable

    def test_universal_blocks_survive_without_tools(self):
        stable = _stable(_make_agent(valid_tool_names=[]))
        assert COMMUNICATION_GUIDANCE in stable
        assert CONTEXT_CONTINUITY_NOTE in stable
        assert USER_PRECEDENCE_NOTE in stable

    def test_tool_gated_blocks_absent_without_tools(self):
        stable = _stable(_make_agent(valid_tool_names=[]))
        assert ASSESSMENT_FIRST_GUIDANCE not in stable
        assert SIDE_EFFECT_CONFIRMATION_GUIDANCE not in stable
        assert OBSERVED_CONTENT_BOUNDARY not in stable
        assert TASK_COMPLETION_GUIDANCE not in stable


class TestContractGating:
    def test_outward_actions_require_complete_payload_inspection(self):
        text = SIDE_EFFECT_CONFIRMATION_GUIDANCE.lower()
        assert "inspect the actual recipients" in text
        assert "complete payload, including attachments" in text
        assert "cannot inspect it completely" in text
        assert "do not send, publish, or share it" in text

    def test_umbrella_flag_removes_contract_family(self):
        stable = _stable(_make_agent(_behavior_contracts=False))
        for block in (
            COMMUNICATION_GUIDANCE,
            ASSESSMENT_FIRST_GUIDANCE,
            SIDE_EFFECT_CONFIRMATION_GUIDANCE,
            OBSERVED_CONTENT_BOUNDARY,
            CONTEXT_CONTINUITY_NOTE,
            USER_PRECEDENCE_NOTE,
        ):
            assert block not in stable

    def test_execution_contract_rides_task_completion_gate_not_umbrella(self):
        stable = _stable(_make_agent(_task_completion_guidance=False))
        assert TASK_COMPLETION_GUIDANCE not in stable
        stable = _stable(_make_agent(_behavior_contracts=False))
        assert TASK_COMPLETION_GUIDANCE in stable

    def test_memory_readback_rides_memory_tool_gate(self):
        with_memory = _stable(_make_agent(valid_tool_names=["memory"]))
        assert MEMORY_READBACK_NOTE in with_memory
        without_memory = _stable(_make_agent())
        assert MEMORY_READBACK_NOTE not in without_memory


class TestContractOrdering:
    def test_boundary_adjacent_after_steer_and_precedence_last(self):
        stable = _stable(_make_agent())
        steer_at = stable.index("Mid-turn user steering")
        boundary_at = stable.index(OBSERVED_CONTENT_BOUNDARY)
        continuity_at = stable.index(CONTEXT_CONTINUITY_NOTE)
        precedence_at = stable.index(USER_PRECEDENCE_NOTE)
        assert steer_at < boundary_at < continuity_at < precedence_at
        # Precedence speaks about everything above it — must close the tier.
        assert stable.endswith(USER_PRECEDENCE_NOTE)

    def test_execution_contract_precedes_communication_contract(self):
        stable = _stable(_make_agent())
        task_at = stable.index(TASK_COMPLETION_GUIDANCE)
        comm_at = stable.index(COMMUNICATION_GUIDANCE)
        assert task_at < comm_at


class TestExecutionAndStoppingContract:
    def test_preserves_artifact_evidence_and_no_fabrication(self):
        assert "working artifact" in TASK_COMPLETION_GUIDANCE
        assert "fresh evidence" in TASK_COMPLETION_GUIDANCE
        assert "never fabricate" in TASK_COMPLETION_GUIDANCE.lower()

    def test_consolidates_proportionality_and_stop_conditions(self):
        text = TASK_COMPLETION_GUIDANCE.lower()
        assert "lightest process" in text
        assert "observed failure or boundary" in text
        assert "materially advances" in text
        assert "domain oracle passes" in text
        assert "retry only when" in text
        assert "new evidence" in text
        assert "materially different strategy" in text
        assert "plan, review, or assessment" in text

    def test_prevents_solution_inflation_and_silent_fallbacks(self):
        text = TASK_COMPLETION_GUIDANCE.lower()
        assert "features, refactors, abstractions, defensive layers" in text
        assert "future compatibility" in text
        assert "current done contract does not require" in text
        assert "do not silently substitute a fallback" in text
        assert "disclose the substitution and its limitations" in text

    def test_gpt_assembled_prompt_has_one_persistence_decision_center(self):
        stable = _stable(
            _make_agent(model="gpt-5.6-sol", _tool_use_enforcement="auto")
        )
        assert stable.count("# Execution and stopping") == 1
        assert "# Finishing the job" not in stable
        assert "# Proportionality" not in stable
        assert "<tool_persistence>" not in stable
        for duplicated in (
            "Keep working until",
            "Do not stop early",
            "Keep calling tools until",
            "End your turn only",
        ):
            assert duplicated not in stable


class TestCacheStability:
    def test_stable_tier_is_byte_identical_across_builds(self):
        # The contracts join the cached prefix; any nondeterminism here
        # breaks upstream prompt caching on every rebuild.
        first = _parts(_make_agent())
        second = _parts(_make_agent())
        assert first["stable"] == second["stable"]
        assert first["context"] == second["context"]


class TestVolatileTimestampHint:
    def test_timestamp_names_itself_session_start(self):
        volatile = _parts(_make_agent())["volatile"]
        assert "Conversation started:" in volatile
        assert "session start date" in volatile
        assert "check with a tool" in volatile


class TestCodingCommentDiscipline:
    def test_coding_brief_carries_comment_contract(self):
        from agent.coding_context import CODING_AGENT_GUIDANCE

        assert "constraints the code itself can't show" in CODING_AGENT_GUIDANCE
        assert "comment density" in CODING_AGENT_GUIDANCE


class TestMemoryBlockHeader:
    def test_rendered_memory_block_flags_staleness(self):
        from tools.memory_tool import MemoryStore

        store = MemoryStore()
        block = store._render_block("memory", ["User prefers concise replies"])
        assert "background context" in block
        assert "may be stale" in block


class TestConfigWiring:
    """Exercise the real agent_init config path, not a hand-built agent.

    The SimpleNamespace tests above would keep passing if the
    ``agent._behavior_contracts`` assignment in agent_init.py were
    reverted (missing attribute defaults to enabled).  This class builds
    a full AIAgent with a patched config so a reverted assignment makes
    ``behavior_contracts: false`` stop working — and fails here.
    """

    def _make_real_agent(self, behavior_contracts):
        from run_agent import AIAgent

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "terminal tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with (
            patch("run_agent.get_tool_definitions", return_value=tool_defs),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"behavior_contracts": behavior_contracts}},
            ),
        ):
            agent = AIAgent(
                model="anthropic/claude-opus-4.8",
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            agent.client = MagicMock()
            return agent

    def test_config_false_disables_contracts(self):
        agent = self._make_real_agent(behavior_contracts=False)
        prompt = agent._build_system_prompt()
        assert COMMUNICATION_GUIDANCE not in prompt
        assert OBSERVED_CONTENT_BOUNDARY not in prompt

    def test_config_true_enables_contracts(self):
        agent = self._make_real_agent(behavior_contracts=True)
        prompt = agent._build_system_prompt()
        assert COMMUNICATION_GUIDANCE in prompt
        assert OBSERVED_CONTENT_BOUNDARY in prompt
