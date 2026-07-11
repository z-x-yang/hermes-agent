from argparse import Namespace
import contextlib
import io
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
