from argparse import Namespace
from dataclasses import fields
import contextlib
import hashlib
import io
import json
import sys
import types

import pytest

from hermes_cli.console_engine import HermesConsoleEngine
from hermes_state import SessionDB
from state_db_maintenance import (
    MaintenanceBlockedError,
    MaintenanceJournal,
    write_maintenance_journal,
)


def test_sessions_repair_fails_closed_during_active_maintenance(tmp_path, monkeypatch):
    import hermes_state

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    write_maintenance_journal(db_path, MaintenanceJournal.new("op-1", db_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

    result = HermesConsoleEngine().execute("sessions repair", confirmed=True)

    assert result.status == "error"
    assert "write blocked by active maintenance" in result.output
    assert "fts-status" in result.output


def test_main_sessions_repair_prints_recovery_instructions(tmp_path, monkeypatch, capsys):
    import hermes_state
    from hermes_cli import main as main_mod

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    write_maintenance_journal(db_path, MaintenanceJournal.new("op-1", db_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "repair"])

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert "write blocked by active maintenance" in output
    assert "fts-status" in output


def test_doctor_fix_is_read_only_and_actionable_during_maintenance(
    tmp_path, monkeypatch
):
    import hermes_state
    from hermes_cli import doctor as doctor_mod

    home = tmp_path / ".hermes"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    db_path = home / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    write_maintenance_journal(db_path, MaintenanceJournal.new("op-1", db_path))

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *args, **kwargs: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    real_connect = hermes_state.sqlite3.connect
    state_db_connects = []

    def recording_connect(database, *args, **kwargs):
        if str(db_path) in str(database):
            state_db_connects.append((database, kwargs.copy()))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(hermes_state.sqlite3, "connect", recording_connect)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        doctor_mod.run_doctor(Namespace(fix=True, ack=None))

    rendered = output.getvalue()
    assert "write diagnostics/fixes blocked by active maintenance" in rendered
    assert "fts-status" in rendered
    assert state_db_connects
    assert all(
        str(database).startswith("file:")
        and "mode=ro" in str(database)
        and kwargs.get("uri") is True
        for database, kwargs in state_db_connects
    )


def test_doctor_preserves_read_only_corruption_reason_during_maintenance(
    tmp_path, monkeypatch
):
    import hermes_state
    from hermes_cli import doctor as doctor_mod

    home = tmp_path / ".hermes"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    db_path = home / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *args, **kwargs: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    def blocked_write_probe(path, *, write_probe=True):
        assert path == db_path
        if write_probe:
            raise MaintenanceBlockedError(
                "write blocked by active maintenance; run fts-status"
            )
        return "readonly-corruption-evidence"

    monkeypatch.setattr(hermes_state, "_db_opens_cleanly", blocked_write_probe)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        doctor_mod.run_doctor(Namespace(fix=False, ack=None))

    rendered = output.getvalue()
    assert "readonly-corruption-evidence" in rendered
    assert "fails a read-only health check" in rendered
    assert "fails a write-health probe" not in rendered


_NEW_SESSION_ACTIONS = (
    "fts-plan",
    "fts-status",
    "fts-migrate",
    "fts-resume",
    "fts-abort",
    "fts-rollback",
    "retention-estimate",
)


def _prepare_cli(monkeypatch, argv):
    from hermes_cli import main as main_mod

    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *argv])
    return main_mod


def test_sessions_help_lists_safe_fts_actions_and_no_permit_flags(monkeypatch, capsys):
    main_mod = _prepare_cli(monkeypatch, ["--help"])

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for action in _NEW_SESSION_ACTIONS:
        assert action in output

    main_mod = _prepare_cli(monkeypatch, ["fts-status", "--operation-id", "forged"])
    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()
    assert exc_info.value.code == 2


def test_read_only_fts_actions_dispatch_before_sessiondb(tmp_path, monkeypatch, capsys):
    import hermes_state
    import state_db_fts_migration as migration

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(
        hermes_state,
        "SessionDB",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("read-only command constructed SessionDB")
        ),
    )
    monkeypatch.setattr(
        migration,
        "plan_fts_migration",
        lambda path: types.SimpleNamespace(
            schema_kind="v1_inline",
            schema_marker=None,
            db_bytes=10,
            wal_bytes=0,
            shm_bytes=0,
            free_bytes=100,
            required_free_bytes=20,
            message_count=2,
            session_count=1,
            archived_session_holds=0,
            session_deletion_candidates=0,
            fts_object_bytes=(),
            writer_status="not_checked",
            maintenance_status="none",
            can_apply=False,
            reasons=("explicit apply required",),
            paired_corpus_version="controlled-v1",
        ),
    )
    monkeypatch.setattr(
        migration,
        "status_fts_migration",
        lambda path: {
            "schema_kind": "v1_inline",
            "schema_marker": None,
            "counts": {"messages": 2, "sessions": 1},
            "maintenance_status": "none",
            "file_fingerprints": {},
            "read_only": True,
        },
    )

    _prepare_cli(monkeypatch, ["fts-plan"]).main()
    plan_output = capsys.readouterr().out
    assert "v1_inline" in plan_output
    assert "can apply: no" in plan_output.lower()

    _prepare_cli(monkeypatch, ["fts-status"]).main()
    status_output = capsys.readouterr().out
    assert "v1_inline" in status_output
    assert "read-only: yes" in status_output.lower()


@pytest.mark.parametrize(
    ("argv", "function_name"),
    [
        (["fts-migrate", "--apply"], "apply_fts_migration"),
        (["fts-resume"], "resume_fts_migration"),
        (["fts-abort"], "abort_fts_migration"),
        (["fts-rollback"], "rollback_fts_migration"),
    ],
)
def test_destructive_fts_actions_confirm_before_mutation(
    tmp_path, monkeypatch, capsys, argv, function_name
):
    import hermes_state
    import state_db_fts_migration as migration

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"fixture")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    calls = []
    monkeypatch.setattr(
        migration,
        function_name,
        lambda *args, **kwargs: calls.append((args, kwargs))
        or types.SimpleNamespace(phase="complete", completed=True),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    _prepare_cli(monkeypatch, argv).main()

    assert calls == []
    assert "Cancelled." in capsys.readouterr().out

    _prepare_cli(monkeypatch, [*argv, "--yes"]).main()
    assert len(calls) == 1


def test_fts_rollback_passes_explicit_backup_after_confirmation(
    tmp_path, monkeypatch
):
    import hermes_state
    import state_db_fts_migration as migration

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"fixture")
    backup = tmp_path / "state.db.pre-v2.backup"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    calls = []
    monkeypatch.setattr(
        migration,
        "rollback_fts_migration",
        lambda path, backup=None: calls.append((path, backup))
        or types.SimpleNamespace(phase="rolled_back", completed=False),
    )

    _prepare_cli(
        monkeypatch, ["fts-rollback", str(backup), "--yes"]
    ).main()

    assert calls == [(db_path, backup)]


def test_fts_migrate_requires_apply_and_failures_exit_nonzero(
    tmp_path, monkeypatch, capsys
):
    import hermes_state
    import state_db_fts_migration as migration

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"fixture")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

    with pytest.raises(SystemExit) as exc_info:
        _prepare_cli(monkeypatch, ["fts-migrate", "--yes"]).main()
    assert exc_info.value.code != 0
    assert "--apply" in capsys.readouterr().out

    def blocked(path):
        raise RuntimeError("write blocked by active maintenance")

    monkeypatch.setattr(migration, "apply_fts_migration", blocked)
    with pytest.raises(SystemExit) as exc_info:
        _prepare_cli(monkeypatch, ["fts-migrate", "--apply", "--yes"]).main()
    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert "write blocked by active maintenance" in output
    assert "fts-status" in output
    assert "fts-resume" in output or "fts-rollback" in output


def test_retention_estimate_json_is_stable_and_read_only(tmp_path, monkeypatch, capsys):
    import hermes_state
    from state_db_fts_migration import RetentionEstimate

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id="s", source="cli")
    db.append_message("s", role="assistant", content="payload")
    db.close()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    before_names = sorted(path.name for path in tmp_path.iterdir())

    _prepare_cli(monkeypatch, ["retention-estimate", "--json"]).main()

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {field.name for field in fields(RetentionEstimate)}
    assert payload["clock_status"] == "unavailable"
    assert payload["rows_by_age_basis"] == "non_actionable_upper_bound"
    assert payload["session_deletion_candidates"] == 0
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == before_names
