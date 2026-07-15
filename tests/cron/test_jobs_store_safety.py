"""Cron registry writes must fail closed against wholesale clobbering."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest


def _job(job_id: str, *, status: str | None = None) -> dict:
    return {
        "id": job_id,
        "name": f"job-{job_id}",
        "prompt": "test",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "schedule_display": "every 60m",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "next_run_at": "2099-01-01T00:00:00+00:00",
        "last_status": status,
        "deliver": "local",
    }


def _completed_oneshot(job_id: str) -> dict:
    job = _job(job_id)
    job.update(
        {
            "schedule": {
                "kind": "once",
                "run_at": "2020-01-01T00:00:00+00:00",
                "display": "once at 2020-01-01 00:00",
            },
            "schedule_display": "once at 2020-01-01 00:00",
            "repeat": {"times": 1, "completed": 1},
            "next_run_at": "2020-01-01T00:00:00+00:00",
        }
    )
    return job


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.delenv("EVELYN_HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    import cron.jobs

    importlib.reload(hermes_constants)
    jobs = importlib.reload(cron.jobs)
    assert jobs.JOBS_FILE == home / "cron" / "jobs.json"
    return jobs


def test_suspicious_count_drop_is_rejected_without_touching_live_bytes(store):
    original = [_job(f"old{i}") for i in range(6)]
    store.save_jobs(original)
    before = store.JOBS_FILE.read_bytes()

    with pytest.raises(store.CronStoreBulkOverwriteError, match="suspicious bulk replacement"):
        store.save_jobs([_job("fixture")])

    assert store.JOBS_FILE.read_bytes() == before
    assert [j["id"] for j in store.load_jobs()] == [j["id"] for j in original]


def test_same_count_foreign_ids_are_rejected(store):
    original = [_job(f"old{i}") for i in range(6)]
    store.save_jobs(original)

    with pytest.raises(store.CronStoreBulkOverwriteError, match="overlap"):
        store.save_jobs([_job(f"new{i}") for i in range(6)])

    assert [j["id"] for j in store.load_jobs()] == [j["id"] for j in original]


def test_normal_single_job_removal_is_allowed_and_preserves_exact_preimage(store):
    original = [_job(f"old{i}") for i in range(6)]
    store.save_jobs(original)
    before = store.JOBS_FILE.read_bytes()

    store.save_jobs(original[:-1])

    assert [j["id"] for j in store.load_jobs()] == [j["id"] for j in original[:-1]]
    assert store.JOBS_PREIMAGE_FILE.read_bytes() == before
    assert (store.JOBS_PREIMAGE_FILE.stat().st_mode & 0o777) == 0o600


def test_ordinary_field_update_preserves_exact_preimage(store):
    original = [_job(f"old{i}") for i in range(6)]
    store.save_jobs(original)
    before = store.JOBS_FILE.read_bytes()
    updated = [dict(job) for job in original]
    updated[0]["last_status"] = "ok"

    store.save_jobs(updated)

    assert store.JOBS_PREIMAGE_FILE.read_bytes() == before
    assert store.load_jobs()[0]["last_status"] == "ok"


def test_explicit_verified_override_allows_bulk_recovery_and_keeps_preimage(store):
    original = [_job(f"old{i}") for i in range(6)]
    store.save_jobs(original)
    before = store.JOBS_FILE.read_bytes()

    store.save_jobs([_job("recovered")], allow_bulk_replace=True)

    assert [j["id"] for j in store.load_jobs()] == ["recovered"]
    assert store.JOBS_PREIMAGE_FILE.read_bytes() == before


def test_due_scan_can_remove_multiple_completed_oneshots(store):
    original = [_completed_oneshot(f"done{i}") for i in range(6)]
    store.save_jobs(original)
    before = store.JOBS_FILE.read_bytes()

    assert store.get_due_jobs() == []

    assert store.load_jobs() == []
    assert store.JOBS_PREIMAGE_FILE.read_bytes() == before
