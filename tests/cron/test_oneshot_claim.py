"""Durable one-shot claim for the built-in scheduler tick path.

Root cause (Phase A): ``tick()`` had no durable, cross-process at-most-once for
one-shot jobs. ``advance_next_run`` is a no-op for one-shots, ``get_due_jobs``
kept returning a fired-but-incomplete one-shot every tick, and the only dedup
was the *per-process* ``_running_job_ids`` set. Two gateway tickers therefore
both started the same one-shot (observed 06:07:43 + 06:07:47).

These tests exercise the real store against the per-test isolated HERMES_HOME
(autouse ``_hermetic_environment`` fixture) — no mocks of the store, per the
E2E-over-mocks discipline for file-touching code.

Crash-recovery policy (documented by tests here): an in-flight one-shot is
gated by a durable ``state="running"`` marker + owner liveness, NOT a
time-based TTL. A run is reclaimed ONLY when its owner is provably dead (same
host, pid gone). A legitimately long-running job (owner alive) is never
re-fired no matter how long it runs. Cross-host / unknown owners are never
auto-reclaimed (conservative: never double-fire).
"""
from __future__ import annotations

import os
import socket


def _oneshot(job_id: str = "os1", *, state: str = "scheduled", **extra):
    """A one-shot job record that is due now (next_run_at in the past)."""
    job = {
        "id": job_id,
        "name": f"oneshot-{job_id}",
        "prompt": "do the thing",
        "schedule": {"kind": "once", "run_at": "2020-01-01T00:00:00+00:00"},
        "schedule_display": "once at 2020-01-01 00:00",
        "enabled": True,
        "state": state,
        "next_run_at": "2020-01-01T00:00:00+00:00",
        "last_run_at": None,
        "repeat": {"times": 1, "completed": 0},
        "deliver": "local",
    }
    job.update(extra)
    return job


def _alive_owner():
    from cron.jobs import _machine_token
    return {"machine": _machine_token(), "pid": os.getpid(), "host": socket.gethostname()}


def _dead_owner():
    """An owner on this machine whose pid is provably gone."""
    import subprocess
    import sys

    from cron.jobs import _machine_token

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()  # process is now dead; its pid is free
    return {"machine": _machine_token(), "pid": p.pid, "host": socket.gethostname()}


# ---------------------------------------------------------------------------
# claim_oneshot_for_run — the durable compare-and-set
# ---------------------------------------------------------------------------

class TestClaimOneshot:
    def test_claim_marks_running_and_records_owner(self):
        from cron.jobs import claim_oneshot_for_run, get_job, save_jobs

        save_jobs([_oneshot()])
        assert claim_oneshot_for_run("os1") is True

        job = get_job("os1")
        assert job["state"] == "running"
        assert job["started_at"]  # a timestamp was stamped
        assert job["run_owner"]["pid"] == os.getpid()

    def test_second_claim_loses_while_owner_alive(self):
        """Two tickers racing the same due one-shot: exactly one wins."""
        from cron.jobs import claim_oneshot_for_run, save_jobs

        save_jobs([_oneshot()])
        assert claim_oneshot_for_run("os1") is True
        assert claim_oneshot_for_run("os1") is False

    def test_claim_reclaims_when_owner_is_dead(self):
        """Crash recovery: a run owned by a dead pid can be reclaimed."""
        from cron.jobs import claim_oneshot_for_run, save_jobs

        save_jobs([_oneshot(state="running", started_at="2020-01-01T00:00:00+00:00",
                            run_owner=_dead_owner())])
        assert claim_oneshot_for_run("os1") is True

    def test_claim_does_not_reclaim_live_long_running(self):
        """No TTL: a long-alive owner is never reclaimed regardless of age."""
        from cron.jobs import claim_oneshot_for_run, save_jobs

        save_jobs([_oneshot(state="running", started_at="2000-01-01T00:00:00+00:00",
                            run_owner=_alive_owner())])
        assert claim_oneshot_for_run("os1") is False

    def test_claim_rejects_missing_disabled_paused(self):
        from cron.jobs import claim_oneshot_for_run, save_jobs

        assert claim_oneshot_for_run("nope") is False
        save_jobs([_oneshot("d", enabled=False)])
        assert claim_oneshot_for_run("d") is False
        save_jobs([_oneshot("p", state="paused")])
        assert claim_oneshot_for_run("p") is False


# ---------------------------------------------------------------------------
# get_due_jobs — a claimed/running one-shot is not due again
# ---------------------------------------------------------------------------

class TestGetDueExcludesRunning:
    def test_running_oneshot_with_live_owner_not_due(self):
        from cron.jobs import get_due_jobs, save_jobs

        save_jobs([_oneshot(state="running", started_at="2020-01-01T00:00:00+00:00",
                            run_owner=_alive_owner())])
        assert [j["id"] for j in get_due_jobs()] == []

    def test_scheduled_oneshot_is_due(self):
        """Sanity: an un-claimed due one-shot is still returned."""
        from cron.jobs import get_due_jobs, save_jobs

        save_jobs([_oneshot()])
        assert [j["id"] for j in get_due_jobs()] == ["os1"]

    def test_dead_owner_oneshot_is_rearmed_and_due(self):
        """Crash recovery surfaces through get_due_jobs: a dead-owner running
        one-shot is reset to scheduled and returned as due again."""
        from cron.jobs import get_due_jobs, get_job, save_jobs

        save_jobs([_oneshot(state="running", started_at="2020-01-01T00:00:00+00:00",
                            run_owner=_dead_owner())])
        assert [j["id"] for j in get_due_jobs()] == ["os1"]
        # re-arm is persisted, not just reflected in the returned copy
        assert get_job("os1")["state"] == "scheduled"
        assert get_job("os1")["started_at"] is None


class TestRecoveryFailsClosed:
    """At-most-once must hold even when the owner cannot be positively verified
    dead. These guard the two double-fire paths flagged in adversarial review:
    a pid that looks dead in OUR namespace but belongs to another machine, and
    a malformed owner record. Both must be treated as alive (not reclaimed)."""

    def test_other_machine_dead_pid_not_reclaimed(self):
        """Same hostname, DIFFERENT machine, pid that is dead in our local pid
        namespace: must NOT be reclaimed (would be a cross-host duplicate)."""
        from cron.jobs import claim_oneshot_for_run, get_due_jobs, save_jobs

        owner = {"machine": "ffffffffffff", "pid": _dead_owner()["pid"],
                 "host": socket.gethostname()}
        save_jobs([_oneshot(state="running",
                            started_at="2020-01-01T00:00:00+00:00", run_owner=owner)])
        assert [j["id"] for j in get_due_jobs()] == []
        assert claim_oneshot_for_run("os1") is False

    def test_malformed_owner_not_reclaimed(self):
        """A running marker with a non-int / missing pid must fail closed."""
        from cron.jobs import claim_oneshot_for_run, get_due_jobs, save_jobs

        host = socket.gethostname()
        machine = _alive_owner()["machine"]
        for bad in ({"machine": machine, "host": host},  # no pid
                    {"machine": machine, "pid": "x", "host": host}):
            save_jobs([_oneshot(state="running",
                                started_at="2020-01-01T00:00:00+00:00", run_owner=bad)])
            assert [j["id"] for j in get_due_jobs()] == []
            assert claim_oneshot_for_run("os1") is False


# ---------------------------------------------------------------------------
# Two tick paths racing for the same due one-shot run it exactly once
# ---------------------------------------------------------------------------

class TestTickAtMostOnce:
    def test_two_ticks_fire_oneshot_once(self, monkeypatch):
        """Headline regression. Tick once, clear the in-process running set
        (simulating a second gateway process), tick again. The durable claim
        must prevent the second fire. mark_job_run is stubbed to a no-op so the
        job is NOT popped between ticks — this isolates the cross-process claim
        from the repeat-limit removal."""
        import cron.scheduler as sched
        from cron.jobs import save_jobs

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        save_jobs([_oneshot()])

        fires: list = []
        monkeypatch.setattr(sched, "run_job",
                            lambda j: (fires.append(j["id"]), (True, "out", "resp", None))[1])
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)

        sched.tick(verbose=False)
        sched._running_job_ids.clear()  # simulate a different process
        sched.tick(verbose=False)

        assert fires == ["os1"]

        sched._shutdown_parallel_pool()


# ---------------------------------------------------------------------------
# Completion clears in-flight state; repeat/removal semantics unchanged
# ---------------------------------------------------------------------------

class TestCompletionClearsRunning:
    def test_mark_job_run_clears_running_markers(self):
        """A completing job must not keep a stale running marker."""
        from cron.jobs import get_job, mark_job_run, save_jobs

        recurring = {
            "id": "r1", "name": "recurring", "prompt": "x",
            "schedule": {"kind": "interval", "minutes": 5},
            "enabled": True, "state": "running",
            "started_at": "2020-01-01T00:00:00+00:00",
            "run_owner": _alive_owner(),
            "next_run_at": "2020-01-01T00:00:00+00:00",
            "last_run_at": None, "repeat": {"times": None, "completed": 0},
            "deliver": "local",
        }
        save_jobs([recurring])
        mark_job_run("r1", True)

        job = get_job("r1")
        assert job["state"] != "running"
        assert job.get("started_at") is None
        assert job.get("run_owner") is None

    def test_completed_oneshot_is_removed(self):
        """Removal semantics unchanged: a one-shot (repeat=1) is removed on
        completion, so it cannot remain scheduled or duplicate-deliver."""
        from cron.jobs import get_due_jobs, get_job, mark_job_run, save_jobs

        save_jobs([_oneshot(state="running", started_at="2020-01-01T00:00:00+00:00",
                            run_owner=_alive_owner())])
        mark_job_run("os1", True)

        assert get_job("os1") is None
        assert get_due_jobs() == []


# ---------------------------------------------------------------------------
# Listing / serialization surfaces the in-flight state
# ---------------------------------------------------------------------------

class TestRunningStateVisible:
    def test_cron_list_shows_running(self, capsys):
        from cron.jobs import save_jobs
        from hermes_cli.cron import cron_list

        save_jobs([_oneshot(state="running", started_at="2026-06-24T06:07:00+00:00",
                            run_owner=_alive_owner())])
        cron_list(show_all=True)
        out = capsys.readouterr().out

        assert "running" in out.lower()
        assert "2026-06-24T06:07:00" in out  # started_at surfaced

    def test_format_job_serializes_started_at(self):
        from tools.cronjob_tools import _format_job

        job = _oneshot(state="running", started_at="2026-06-24T06:07:00+00:00",
                       run_owner=_alive_owner())
        fmt = _format_job(job)
        assert fmt["state"] == "running"
        assert "started_at" in fmt
        assert fmt["started_at"] == "2026-06-24T06:07:00+00:00"


# ---------------------------------------------------------------------------
# Real liveness primitive (no mocks)
# ---------------------------------------------------------------------------

def test_pid_alive_real():
    from cron.jobs import _pid_alive

    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(_dead_owner()["pid"]) is False
