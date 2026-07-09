from types import SimpleNamespace


def test_tool_result_budget_uses_internal_compression_window():
    from agent.tool_executor import _budget_for_agent

    agent = SimpleNamespace(
        context_compressor=SimpleNamespace(
            context_length=1_000_000,
            compression_context_length=65_000,
        )
    )

    budget = _budget_for_agent(agent)

    assert budget.default_result_size < 100_000
    assert budget.turn_budget < 200_000
