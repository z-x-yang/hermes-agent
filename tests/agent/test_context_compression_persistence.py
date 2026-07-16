"""Persistence regression tests for in-place context compression."""

from typing import Any, cast

from agent.conversation_compression import compress_context
from run_agent import AIAgent


class _FakeCompressor:
    compression_count = 1
    last_compression_rough_tokens = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    awaiting_real_usage_after_compression = False

    def __init__(self, compressed):
        self._compressed = compressed
        self._last_compress_aborted = False
        self._last_summary_error = None
        self._last_compression_audit_record = {
            "event": "context_compression",
            "compression_id": "cid-test",
            "session_id": "session-1",
            "retained_tail_output_count": len(compressed),
        }
        self.persist_audit_calls = []

    def compress(self, messages, **_kwargs):
        return self._compressed

    def write_compression_persist_audit(self, **kwargs):
        self.persist_audit_calls.append(kwargs)


class _FakeTodoStore:
    def __init__(self, snapshot: str = ""):
        self.snapshot = snapshot

    def format_for_injection(self):
        return self.snapshot


class _FakeSessionDB:
    def __init__(self):
        self.archived = None
        self.appended = []
        self.updated_system_prompt = None
        self.lock_released = False
        self.row_ids = [101, 102]

    def try_acquire_compression_lock(self, _session_id, _holder):
        return True

    def release_compression_lock(self, _session_id, _holder):
        self.lock_released = True

    def archive_and_compact(self, _session_id, compressed):
        self.archived = list(compressed)

    def archive_and_compact_returning_ids(self, _session_id, compressed):
        self.archived = list(compressed)
        return list(self.row_ids[: len(compressed)])

    def get_messages(self, _session_id):
        rows = []
        for idx, msg in enumerate(self.archived or [], start=101):
            rows.append({"id": idx, **msg})
        return rows

    def update_system_prompt(self, _session_id, system_prompt):
        self.updated_system_prompt = system_prompt

    def append_message(self, **kwargs):
        self.appended.append(kwargs)


def _make_agent(compressed, *, todo_snapshot: str = "") -> Any:
    agent = cast(Any, object.__new__(AIAgent))
    agent.session_id = "session-1"
    agent.model = "test-model"
    agent.platform = "discord"
    agent.tools = []
    agent.compression_in_place = True
    agent.context_compressor = _FakeCompressor(compressed)
    agent._session_db = _FakeSessionDB()
    agent._session_db_created = True
    agent._session_init_model_config = {}
    agent._compression_feasibility_checked = True
    agent._memory_manager = None
    agent._todo_store = _FakeTodoStore(todo_snapshot)
    agent._cached_system_prompt = "system-old"
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = agent.session_id
    agent._gateway_session_key = None
    agent._emit_status = lambda *_args, **_kwargs: None
    agent._emit_warning = lambda *_args, **_kwargs: None
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda system_message: f"rebuilt: {system_message}"
    agent.commit_memory_session = lambda _messages: None
    agent._apply_persist_user_message_override = lambda _messages: None
    return agent


def test_in_place_compression_rebaselines_session_db_flush_ids():
    """archive_and_compact already writes the compacted transcript.

    The same turn's final session flush must not append the compressed summary
    and protected tail a second time.  This reproduces the live duplicate
    active-history shape seen after in-place compression.
    """
    original = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    compressed = [
        {"role": "user", "content": "[CONTEXT COMPACTION] compacted summary"},
        {"role": "user", "content": "current user"},
    ]
    agent = _make_agent(compressed)

    returned, _system_prompt = compress_context(agent, original, "system")

    assert returned is compressed
    assert agent._session_db.archived == compressed

    # The pre-compaction flush persisted the (never-flushed) originals exactly
    # once — that is the single chokepoint feeding the cumulative
    # message_count/tool_call_count accounting. The compressed rows are written
    # by archive_and_compact only, never by the flush path.
    assert [m["content"] for m in agent._session_db.appended] == [
        "old user",
        "old assistant",
        "current user",
    ]

    # Normal turn cleanup path after compression.
    agent._flush_messages_to_session_db(returned, original)

    # Unchanged: the same turn's final flush must not append the compressed
    # summary/protected tail a second time (flush-cursor seed honored).
    assert [m["content"] for m in agent._session_db.appended] == [
        "old user",
        "old assistant",
        "current user",
    ]


def test_in_place_compression_writes_persist_audit_with_output_row_ids():
    """The audit companion event must be emitted after archive persistence.

    The decision event is written before SQLite assigns row ids; the in-place
    persistence path must enrich it with the exact rows that now make up the
    active compacted transcript.
    """
    original = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    compressed = [
        {"role": "assistant", "content": "[CONTEXT COMPACTION] compacted summary"},
        {"role": "user", "content": "current user"},
    ]
    agent = _make_agent(compressed)

    compress_context(agent, original, "system")

    assert agent.context_compressor.persist_audit_calls == [
        {
            "output_row_ids": [101, 102],
            "retained_tail_output_count": 2,
        }
    ]


def test_in_place_persist_audit_marks_post_compression_todo_injection():
    """Todo snapshot injection happens after compressor accounting.

    The companion audit must make that extra synthetic row explicit so the
    content-free compression record is not mistaken for the exact persisted
    after/tail shape.
    """
    original = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    compressed = [
        {"role": "assistant", "content": "[CONTEXT COMPACTION] compacted summary"},
        {"role": "user", "content": "current user"},
    ]
    todo_snapshot = "[Your active task list was preserved across context compression]\n- [>] fix.tail (in_progress)"
    agent = _make_agent(
        compressed,
        todo_snapshot=todo_snapshot,
    )
    agent._session_db.row_ids = [101, 102, 103]

    compress_context(agent, original, "system")

    assert agent._session_db.archived[-1] == {
        "role": "user",
        "content": todo_snapshot,
    }
    assert agent.context_compressor.persist_audit_calls == [
        {
            "output_row_ids": [101, 102, 103],
            "retained_tail_output_count": 2,
            "post_compression_injected_count": 1,
            "post_compression_injected_row_ids": [103],
        }
    ]


def test_in_place_persist_audit_uses_archive_returned_row_ids_if_reload_fails():
    """Row-id audit should not depend on a second post-archive DB read.

    Live evidence showed decision audit records without companion persist events;
    if the enrichment read fails after a successful archive, the current code
    silently loses the output row ids. The write transaction already knows the
    inserted ids, so prefer those when the DB exposes them.
    """
    original = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    compressed = [
        {"role": "assistant", "content": "[CONTEXT COMPACTION] compacted summary"},
        {"role": "user", "content": "current user"},
    ]
    agent = _make_agent(compressed)

    def fail_reload(_session_id):
        raise RuntimeError("post-archive reload unavailable")

    agent._session_db.get_messages = fail_reload

    compress_context(agent, original, "system")

    assert agent.context_compressor.persist_audit_calls == [
        {
            "output_row_ids": [101, 102],
            "retained_tail_output_count": 2,
        }
    ]
