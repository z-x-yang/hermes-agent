import os
import stat
from dataclasses import replace

import pytest

from state_db_maintenance import (
    JournalPhase,
    MaintenanceBlockedError,
    MaintenanceJournal,
    MaintenancePermit,
    assert_state_db_maintenance_access,
    fingerprint_path,
    issue_maintenance_permit,
    load_maintenance_journal,
    maintenance_journal_path,
    state_db_file_inventory,
    write_maintenance_journal,
)


def test_write_journal_is_0600_and_fsyncs_parent(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    fsynced_modes = []
    real_fsync = os.fsync

    def record(fd):
        fsynced_modes.append(os.fstat(fd).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record)
    write_maintenance_journal(db_path, MaintenanceJournal.new("op-1", db_path))
    path = maintenance_journal_path(db_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_journal_round_trip_preserves_required_fields(tmp_path):
    db = tmp_path / "state.db"
    original = replace(
        MaintenanceJournal.new("op-1", db),
        backup_path=str(tmp_path / "backup.db"),
        work_path=str(tmp_path / "work"),
        candidate_path=str(tmp_path / "candidate.db"),
        fingerprints={"db": {"size": 12, "sha256": "abc"}},
        expected_row_counts={"messages": 7},
    )
    write_maintenance_journal(db, original)
    assert load_maintenance_journal(db) == original


def test_nonterminal_journal_blocks_write_but_allows_read_only(tmp_path):
    db = tmp_path / "state.db"
    write_maintenance_journal(db, MaintenanceJournal.new("op-1", db))
    with pytest.raises(MaintenanceBlockedError, match="fts-status"):
        assert_state_db_maintenance_access(db, write_capable=True)
    assert assert_state_db_maintenance_access(db, write_capable=False) is None


@pytest.mark.parametrize(
    "phase", [JournalPhase.COMPLETE, JournalPhase.ABORTED, JournalPhase.ROLLED_BACK]
)
def test_terminal_journal_allows_ordinary_write(tmp_path, phase):
    db = tmp_path / "state.db"
    terminal_journal = replace(MaintenanceJournal.new("op-1", db), phase=phase)
    write_maintenance_journal(db, terminal_journal)
    assert assert_state_db_maintenance_access(db, write_capable=True) is None


def test_permit_rejects_wrong_operation_id(tmp_path):
    db = tmp_path / "state.db"
    write_maintenance_journal(db, MaintenanceJournal.new("op-1", db))
    with pytest.raises(MaintenanceBlockedError, match="operation"):
        issue_maintenance_permit(db, "op-2", frozenset({JournalPhase.PLANNED}))


def test_permit_rejects_disallowed_phase(tmp_path):
    db = tmp_path / "state.db"
    write_maintenance_journal(db, MaintenanceJournal.new("op-1", db))
    permit = issue_maintenance_permit(
        db, "op-1", frozenset({JournalPhase.WRITERS_STOPPED})
    )
    with pytest.raises(MaintenanceBlockedError, match="phase"):
        assert_state_db_maintenance_access(db, write_capable=True, permit=permit)


def test_permit_rejects_replaced_journal_inode(tmp_path):
    db = tmp_path / "state.db"
    journal = MaintenanceJournal.new("op-1", db)
    write_maintenance_journal(db, journal)
    permit = issue_maintenance_permit(db, "op-1", frozenset({JournalPhase.PLANNED}))
    write_maintenance_journal(db, journal)
    with pytest.raises(MaintenanceBlockedError, match="replaced"):
        assert_state_db_maintenance_access(db, write_capable=True, permit=permit)


def test_issued_matching_permit_allows_scoped_write(tmp_path):
    db = tmp_path / "state.db"
    write_maintenance_journal(db, MaintenanceJournal.new("op-1", db))
    permit = issue_maintenance_permit(db, "op-1", frozenset({JournalPhase.PLANNED}))
    assert (
        assert_state_db_maintenance_access(db, write_capable=True, permit=permit) is None
    )
    with pytest.raises(TypeError, match="only be issued"):
        MaintenancePermit()


def test_canonical_and_symlink_db_paths_share_journal(tmp_path):
    real_db = tmp_path / "real-state.db"
    real_db.touch()
    alias = tmp_path / "state.db"
    alias.symlink_to(real_db)
    write_maintenance_journal(alias, MaintenanceJournal.new("op-1", alias))
    assert maintenance_journal_path(alias) == maintenance_journal_path(real_db)
    with pytest.raises(MaintenanceBlockedError):
        assert_state_db_maintenance_access(real_db, write_capable=True)


def test_malformed_journal_fails_closed_for_write_capable_access(tmp_path):
    db = tmp_path / "state.db"
    maintenance_journal_path(db).write_text("{not-json", encoding="utf-8")
    with pytest.raises(MaintenanceBlockedError, match="malformed"):
        assert_state_db_maintenance_access(db, write_capable=True)
    assert assert_state_db_maintenance_access(db, write_capable=False) is None


def test_fingerprint_and_inventory_cover_db_wal_and_shm(tmp_path):
    db = tmp_path / "state.db"
    db.write_bytes(b"database")
    db.with_name(f"{db.name}-wal").write_bytes(b"wal")

    fingerprint = fingerprint_path(db)
    assert fingerprint is not None
    assert fingerprint["size"] == len(b"database")
    assert len(str(fingerprint["sha256"])) == 64

    inventory = state_db_file_inventory(db)
    assert set(inventory) == {"db", "wal", "shm"}
    assert inventory["db"] == fingerprint
    assert inventory["wal"] is not None
    assert inventory["shm"] is None
