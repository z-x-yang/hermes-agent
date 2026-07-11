"""Crash-durable external maintenance journal for the Hermes state database."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class JournalPhase(str, Enum):
    PLANNED = "planned"
    WRITERS_STOPPED = "writers_stopped"
    CHECKPOINTED = "checkpointed"
    BACKUP_READY = "backup_ready"
    CANDIDATE_READY = "candidate_ready"
    SWAPPING = "swapping"
    OLD_MOVED = "old_moved"
    CANDIDATE_LIVE = "candidate_live"
    CANARY_PASSED = "canary_passed"
    COMPLETE = "complete"
    ABORTED = "aborted"
    ROLLED_BACK = "rolled_back"


TERMINAL_PHASES = frozenset(
    {JournalPhase.COMPLETE, JournalPhase.ABORTED, JournalPhase.ROLLED_BACK}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


@dataclass(frozen=True)
class MaintenanceJournal:
    version: int
    operation_id: str
    phase: JournalPhase
    db_path: str
    backup_path: str | None
    work_path: str | None
    candidate_path: str | None
    fingerprints: dict[str, dict[str, int | str] | None]
    created_at: str
    updated_at: str
    expected_row_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def new(cls, operation_id: str, db_path: Path) -> "MaintenanceJournal":
        if not operation_id:
            raise ValueError("operation_id must not be empty")
        now = _utc_now()
        return cls(
            version=1,
            operation_id=operation_id,
            phase=JournalPhase.PLANNED,
            db_path=str(_canonical(db_path)),
            backup_path=None,
            work_path=None,
            candidate_path=None,
            fingerprints={},
            created_at=now,
            updated_at=now,
            expected_row_counts={},
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        return value

    @classmethod
    def from_dict(cls, value: object) -> "MaintenanceJournal":
        if not isinstance(value, dict):
            raise ValueError("journal root must be an object")
        required = {
            "version",
            "operation_id",
            "phase",
            "db_path",
            "backup_path",
            "work_path",
            "candidate_path",
            "fingerprints",
            "created_at",
            "updated_at",
            "expected_row_counts",
        }
        if set(value) != required:
            raise ValueError("journal fields do not match version 1 schema")
        if value["version"] != 1:
            raise ValueError("unsupported journal version")
        if not isinstance(value["operation_id"], str) or not value["operation_id"]:
            raise ValueError("operation_id must be a non-empty string")
        if not isinstance(value["db_path"], str) or not value["db_path"]:
            raise ValueError("db_path must be a non-empty string")
        for name in ("backup_path", "work_path", "candidate_path"):
            if value[name] is not None and not isinstance(value[name], str):
                raise ValueError(f"{name} must be a string or null")
        if not isinstance(value["fingerprints"], dict):
            raise ValueError("fingerprints must be an object")
        if not isinstance(value["created_at"], str) or not isinstance(
            value["updated_at"], str
        ):
            raise ValueError("timestamps must be strings")
        row_counts = value["expected_row_counts"]
        if not isinstance(row_counts, dict) or not all(
            isinstance(key, str) and isinstance(count, int) and count >= 0
            for key, count in row_counts.items()
        ):
            raise ValueError("expected_row_counts must contain non-negative integers")
        return cls(
            version=1,
            operation_id=value["operation_id"],
            phase=JournalPhase(value["phase"]),
            db_path=value["db_path"],
            backup_path=value["backup_path"],
            work_path=value["work_path"],
            candidate_path=value["candidate_path"],
            fingerprints=value["fingerprints"],
            created_at=value["created_at"],
            updated_at=value["updated_at"],
            expected_row_counts=row_counts,
        )


class MaintenanceBlockedError(RuntimeError):
    """Raised when state DB maintenance makes write-capable access unsafe."""


class MaintenancePermit:
    """An opaque, immutable, process-local capability issued by this module."""

    __slots__ = (
        "db_path",
        "journal_path",
        "journal_device",
        "journal_inode",
        "operation_id",
        "allowed_phases",
        "_token",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("MaintenancePermit can only be issued by issue_maintenance_permit()")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MaintenancePermit is immutable")


_PERMIT_TOKEN = object()
_ISSUED_PERMITS: set[MaintenancePermit] = set()


def maintenance_journal_path(db_path: Path) -> Path:
    canonical = _canonical(db_path)
    return canonical.with_name(f"{canonical.name}.maintenance.json")


def write_maintenance_journal(db_path: Path, record: MaintenanceJournal) -> None:
    path = maintenance_journal_path(db_path)
    if _canonical(Path(record.db_path)) != _canonical(db_path):
        raise ValueError("journal db_path does not match target database")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record.to_dict(), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def load_maintenance_journal(db_path: Path) -> MaintenanceJournal | None:
    path = maintenance_journal_path(db_path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError:
        return None
    return MaintenanceJournal.from_dict(value)


def fingerprint_path(path: Path) -> dict[str, int | str] | None:
    try:
        with Path(path).open("rb") as stream:
            stat_result = os.fstat(stream.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return None
    return {
        "path": str(_canonical(path)),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "sha256": digest.hexdigest(),
    }


def state_db_file_inventory(db_path: Path) -> dict[str, dict | None]:
    canonical = _canonical(db_path)
    return {
        "db": fingerprint_path(canonical),
        "wal": fingerprint_path(Path(f"{canonical}-wal")),
        "shm": fingerprint_path(Path(f"{canonical}-shm")),
    }


def issue_maintenance_permit(
    db_path: Path,
    operation_id: str,
    allowed_phases: frozenset[JournalPhase],
) -> MaintenancePermit:
    try:
        journal = load_maintenance_journal(db_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise MaintenanceBlockedError(
            f"fts-status: malformed maintenance journal: {exc}"
        ) from exc
    if journal is None:
        raise MaintenanceBlockedError("fts-status: no maintenance journal for permit")
    if journal.operation_id != operation_id:
        raise MaintenanceBlockedError("fts-status: maintenance operation id mismatch")
    journal_path = maintenance_journal_path(db_path)
    journal_stat = journal_path.stat()
    permit = object.__new__(MaintenancePermit)
    object.__setattr__(permit, "db_path", _canonical(db_path))
    object.__setattr__(permit, "journal_path", journal_path)
    object.__setattr__(permit, "journal_device", journal_stat.st_dev)
    object.__setattr__(permit, "journal_inode", journal_stat.st_ino)
    object.__setattr__(permit, "operation_id", operation_id)
    object.__setattr__(permit, "allowed_phases", frozenset(allowed_phases))
    object.__setattr__(permit, "_token", _PERMIT_TOKEN)
    _ISSUED_PERMITS.add(permit)
    return permit


def assert_state_db_maintenance_access(
    db_path: Path,
    *,
    write_capable: bool,
    permit: MaintenancePermit | None = None,
) -> None:
    if not write_capable:
        return
    try:
        journal = load_maintenance_journal(db_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise MaintenanceBlockedError(
            f"fts-status: malformed maintenance journal: {exc}"
        ) from exc
    if journal is None or journal.phase in TERMINAL_PHASES:
        return
    canonical_db = _canonical(db_path)
    if _canonical(Path(journal.db_path)) != canonical_db:
        raise MaintenanceBlockedError(
            "fts-status: maintenance journal database path mismatch"
        )
    if (
        permit is None
        or permit not in _ISSUED_PERMITS
        or permit._token is not _PERMIT_TOKEN
    ):
        raise MaintenanceBlockedError(
            "fts-status: write blocked by active maintenance; issued permit required"
        )
    if (
        permit.db_path != canonical_db
        or permit.journal_path != maintenance_journal_path(db_path)
    ):
        raise MaintenanceBlockedError(
            "fts-status: maintenance permit database path mismatch"
        )
    try:
        journal_stat = permit.journal_path.stat()
    except FileNotFoundError as exc:
        raise MaintenanceBlockedError(
            "fts-status: maintenance journal was replaced"
        ) from exc
    if (journal_stat.st_dev, journal_stat.st_ino) != (
        permit.journal_device,
        permit.journal_inode,
    ):
        raise MaintenanceBlockedError("fts-status: maintenance journal was replaced")
    if permit.operation_id != journal.operation_id:
        raise MaintenanceBlockedError(
            "fts-status: maintenance permit operation mismatch"
        )
    if journal.phase not in permit.allowed_phases:
        raise MaintenanceBlockedError("fts-status: maintenance permit phase mismatch")
