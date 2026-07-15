"""Collection-time imports must never bind profile state to a live home."""
from __future__ import annotations

from pathlib import Path

# Intentionally import during pytest collection, before autouse fixtures run.
import cron.jobs as cron_jobs

COLLECTION_JOBS_FILE = cron_jobs.JOBS_FILE.resolve()


def test_collection_time_cron_store_is_not_the_default_live_profile():
    default_live = (Path.home() / ".hermes" / "cron" / "jobs.json").resolve()
    assert COLLECTION_JOBS_FILE != default_live
    assert "pytest-collection" in str(COLLECTION_JOBS_FILE)
