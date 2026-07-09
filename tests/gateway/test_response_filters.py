from gateway.response_filters import (
    is_context_compaction_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
    is_partial_context_compaction_response,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")


def test_failed_agent_result_never_counts_as_intentional_silence():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    assert not is_intentional_silence_agent_result({"failed": True}, "NO_REPLY")


def test_context_compaction_summary_is_internal_response():
    raw = (
        "[CONTEXT COMPACTION] Earlier turns were compacted into the summary below; "
        "treat it as working context, not as a new user request.\n"
        "## Primary Request and Intent\n"
        "Do the thing.\n"
        "## Current Work\n"
        "Still debugging.\n"
        "--- END OF COMPACTED CONTEXT ---"
    )

    assert is_context_compaction_response(raw)


def test_legacy_context_compaction_reference_banner_is_internal_response():
    raw = (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted.\n"
        "## Active Task\n"
        "Internal state.\n"
        "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
    )

    assert is_context_compaction_response(raw)
    assert is_partial_context_compaction_response(raw[:60])


def test_context_compaction_body_without_banner_is_internal_response():
    # Discord incident regression: a compacted summary body was chunk-delivered
    # after a streaming/final path stripped the banner line.
    raw = (
        "## Primary Request and Intent\n"
        "Do the thing.\n"
        "## Key Technical Concepts\n"
        "Important constraints.\n"
        "## All User Messages\n"
        "1. User asked.\n"
        "## Current Work\n"
        "Still debugging.\n"
        "--- END OF COMPACTED CONTEXT ---"
    )

    assert is_context_compaction_response(raw)


def test_partial_context_compaction_body_holds_streaming_prefix():
    assert is_partial_context_compaction_response(
        "## Primary Request and Intent\nInternal state so far"
    )


def test_partial_context_compaction_does_not_hold_ordinary_markdown():
    assert not is_partial_context_compaction_response("## Different Heading\nhello")


def test_normal_markdown_report_is_not_context_compaction_response():
    raw = (
        "## Primary Request and Intent\n"
        "This is a user-facing report that mentions current work, but it is not "
        "a persisted compaction block."
    )

    assert not is_context_compaction_response(raw)
