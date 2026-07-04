"""Tests for per-job workdir support in cron jobs.

Covers:
  - jobs.create_job: param plumbing, validation, default-None preserved
  - jobs._normalize_workdir: absolute / relative / missing / file-not-dir
  - jobs.update_job: set, clear, re-validate
  - tools.cronjob_tools.cronjob: create + update JSON round-trip, schema
    includes workdir, _format_job exposes it when set
  - scheduler.tick(): only serializes no_agent workdir jobs that still chdir;
    LLM workdir jobs run through the parallel pool using ContextVar cwd
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate cron job storage into a temp dir so tests don't stomp on real jobs."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# ---------------------------------------------------------------------------
# jobs._normalize_workdir
# ---------------------------------------------------------------------------

class TestNormalizeWorkdir:
    def test_none_returns_none(self):
        from cron.jobs import _normalize_workdir
        assert _normalize_workdir(None) is None

    def test_empty_string_returns_none(self):
        from cron.jobs import _normalize_workdir
        assert _normalize_workdir("") is None
        assert _normalize_workdir("   ") is None

    def test_absolute_existing_dir_returns_resolved_str(self, tmp_path):
        from cron.jobs import _normalize_workdir
        result = _normalize_workdir(str(tmp_path))
        assert result == str(tmp_path.resolve())

    def test_tilde_expands(self, tmp_path, monkeypatch):
        from cron.jobs import _normalize_workdir
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _normalize_workdir("~")
        assert result == str(tmp_path.resolve())

    def test_relative_path_rejected(self):
        from cron.jobs import _normalize_workdir
        with pytest.raises(ValueError, match="absolute path"):
            _normalize_workdir("some/relative/path")

    def test_missing_dir_rejected(self, tmp_path):
        from cron.jobs import _normalize_workdir
        missing = tmp_path / "does-not-exist"
        with pytest.raises(ValueError, match="does not exist"):
            _normalize_workdir(str(missing))

    def test_file_not_dir_rejected(self, tmp_path):
        from cron.jobs import _normalize_workdir
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="not a directory"):
            _normalize_workdir(str(f))


# ---------------------------------------------------------------------------
# jobs.create_job and update_job
# ---------------------------------------------------------------------------

class TestCreateJobWorkdir:
    def test_workdir_stored_when_set(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job
        job = create_job(
            prompt="hello",
            schedule="every 1h",
            workdir=str(tmp_cron_dir),
        )
        stored = get_job(job["id"])
        assert stored["workdir"] == str(tmp_cron_dir.resolve())

    def test_workdir_none_preserves_old_behaviour(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job
        job = create_job(prompt="hello", schedule="every 1h")
        stored = get_job(job["id"])
        # Field is present on the dict but None — downstream code checks
        # truthiness to decide whether the feature is active.
        assert stored.get("workdir") is None

    def test_create_rejects_invalid_workdir(self, tmp_cron_dir):
        from cron.jobs import create_job
        with pytest.raises(ValueError):
            create_job(
                prompt="hello",
                schedule="every 1h",
                workdir="not/absolute",
            )


class TestUpdateJobWorkdir:
    def test_set_workdir_via_update(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(prompt="x", schedule="every 1h")
        update_job(job["id"], {"workdir": str(tmp_cron_dir)})
        assert get_job(job["id"])["workdir"] == str(tmp_cron_dir.resolve())

    def test_clear_workdir_with_none(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(
            prompt="x", schedule="every 1h", workdir=str(tmp_cron_dir)
        )
        update_job(job["id"], {"workdir": None})
        assert get_job(job["id"])["workdir"] is None

    def test_clear_workdir_with_empty_string(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(
            prompt="x", schedule="every 1h", workdir=str(tmp_cron_dir)
        )
        update_job(job["id"], {"workdir": ""})
        assert get_job(job["id"])["workdir"] is None

    def test_update_rejects_invalid_workdir(self, tmp_cron_dir):
        from cron.jobs import create_job, update_job
        job = create_job(prompt="x", schedule="every 1h")
        with pytest.raises(ValueError):
            update_job(job["id"], {"workdir": "nope/relative"})


# ---------------------------------------------------------------------------
# tools.cronjob_tools: end-to-end JSON round-trip
# ---------------------------------------------------------------------------

class TestCronjobToolWorkdir:
    def test_create_with_workdir_json_roundtrip(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
                workdir=str(tmp_cron_dir),
            )
        )
        assert result["success"] is True
        assert result["job"]["workdir"] == str(tmp_cron_dir.resolve())

    def test_create_without_workdir_hides_field_in_format(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
            )
        )
        assert result["success"] is True
        # _format_job omits the field when unset — reduces noise in agent output.
        assert "workdir" not in result["job"]

    def test_update_clears_workdir_with_empty_string(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
                workdir=str(tmp_cron_dir),
            )
        )
        job_id = created["job_id"]

        updated = json.loads(
            cronjob(action="update", job_id=job_id, workdir="")
        )
        assert updated["success"] is True
        assert "workdir" not in updated["job"]

    def test_schema_advertises_workdir(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA
        assert "workdir" in CRONJOB_SCHEMA["parameters"]["properties"]
        desc = CRONJOB_SCHEMA["parameters"]["properties"]["workdir"]["description"]
        assert "absolute" in desc.lower()


# ---------------------------------------------------------------------------
# scheduler.tick(): workdir partition
# ---------------------------------------------------------------------------

class TestTickWorkdirPartition:
    """
    tick() must not serialize LLM workdir jobs. They use ContextVar cwd, so
    long jobs in the same project should not head-of-line block other cron
    jobs. Only no_agent workdir jobs remain sequential because run_job still
    uses os.chdir() for those script-only jobs.
    """

    def test_llm_workdir_jobs_use_parallel_pool_but_no_agent_workdir_is_sequential(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        # Two LLM workdir jobs + one no_agent workdir job + one plain job.
        workdir_a = {"id": "a", "name": "A", "workdir": str(tmp_path)}
        workdir_b = {"id": "b", "name": "B", "workdir": str(tmp_path)}
        no_agent_workdir = {"id": "d", "name": "D", "workdir": str(tmp_path), "no_agent": True}
        parallel_job = {"id": "c", "name": "C", "workdir": None}

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [workdir_a, workdir_b, no_agent_workdir, parallel_job])
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)

        # Record call order / thread context.
        import threading
        calls: list[tuple[str, str]] = []
        order_lock = threading.Lock()

        def fake_run_job(job):
            # Return a minimal tuple matching run_job's signature.
            with order_lock:
                calls.append((job["id"], threading.current_thread().name))
            return True, "output", "response", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda _jid, _o: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            sched, "_deliver_result", lambda *_a, **_kw: None
        )

        n = sched.tick(verbose=False)
        assert n == 4

        # LLM workdir jobs now use the parallel pool, same as non-workdir jobs.
        for jid in ("a", "b", "c"):
            thread_name = next(t for j, t in calls if j == jid)
            assert thread_name.startswith("cron-parallel"), thread_name

        # no_agent workdir jobs still run on the persistent single-thread
        # cron-seq pool because run_job uses os.chdir() for script-only jobs.
        main_thread_name = threading.current_thread().name
        no_agent_thread = next(t for j, t in calls if j == "d")
        assert no_agent_thread != main_thread_name
        assert no_agent_thread.startswith("cron-seq"), no_agent_thread


# ---------------------------------------------------------------------------
# scheduler.run_job: TERMINAL_CWD + skip_context_files wiring
# ---------------------------------------------------------------------------

class TestRunJobTerminalCwd:
    """
    run_job pins workdir in ContextVar cwd + flips skip_context_files=False,
    without mutating process-global TERMINAL_CWD.
    We stub AIAgent so no real API call happens.
    """

    @staticmethod
    def _install_stubs(monkeypatch, observed: dict):
        """Patch enough of run_job's deps that it executes without real creds."""
        import os
        import sys
        import cron.scheduler as sched

        class FakeAgent:
            def __init__(self, **kwargs):
                from agent.runtime_cwd import resolve_context_cwd

                observed["skip_context_files"] = kwargs.get("skip_context_files")
                observed["load_soul_identity"] = kwargs.get("load_soul_identity")
                observed["terminal_cwd_during_init"] = os.environ.get(
                    "TERMINAL_CWD", "_UNSET_"
                )
                observed["context_cwd_during_init"] = str(resolve_context_cwd())

            def run_conversation(self, *_a, **_kw):
                from agent.runtime_cwd import resolve_context_cwd
                from tools.terminal_tool import (
                    _resolve_container_task_id,
                    resolve_task_overrides,
                )

                observed["run_task_id"] = _kw.get("task_id")
                observed["terminal_cwd_during_run"] = os.environ.get(
                    "TERMINAL_CWD", "_UNSET_"
                )
                observed["context_cwd_during_run"] = str(resolve_context_cwd())
                observed["task_overrides_during_run"] = resolve_task_overrides(
                    _kw.get("task_id")
                )
                observed["container_task_id_during_run"] = _resolve_container_task_id(
                    _kw.get("task_id")
                )
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        # Bypass the real provider resolver — it reads ~/.hermes and credentials.
        from hermes_cli import runtime_provider as _rtp
        monkeypatch.setattr(
            _rtp,
            "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "test",
                "api_key": "k",
                "base_url": "http://test.local",
                "api_mode": "chat_completions",
            },
        )

        # Stub scheduler helpers that would otherwise hit the filesystem / config.
        monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None: "hi")
        monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
        # Unlimited inactivity so the poll loop returns immediately.
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        # run_job calls load_dotenv(~/.hermes/.env, override=True), which will
        # happily clobber TERMINAL_CWD out from under us if the real user .env
        # has TERMINAL_CWD set (common on dev boxes).  Stub it out.
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

    def test_workdir_uses_context_cwd_and_terminal_task_override(
        self, tmp_path, monkeypatch
    ):
        import os
        import cron.scheduler as sched

        # Make sure the test's TERMINAL_CWD starts at a known non-workdir value.
        # Use monkeypatch.setenv so it's restored on teardown regardless of
        # whatever other tests in this xdist worker have left behind.
        monkeypatch.setenv("TERMINAL_CWD", "/original/cwd")

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        job = {
            "id": "abc",
            "name": "wd-job",
            "workdir": str(tmp_path),
            "schedule_display": "manual",
        }

        success, _output, response, error = sched.run_job(job)
        assert success is True, f"run_job failed: error={error!r} response={response!r}"

        # AIAgent was built with skip_context_files=False (feature ON).
        assert observed["skip_context_files"] is False
        assert observed["load_soul_identity"] is True
        # TERMINAL_CWD stays untouched; the job workdir is carried in ContextVar cwd.
        assert observed["terminal_cwd_during_init"] == "/original/cwd"
        assert observed["terminal_cwd_during_run"] == "/original/cwd"
        assert observed["context_cwd_during_init"] == str(tmp_path.resolve())
        assert observed["context_cwd_during_run"] == str(tmp_path.resolve())
        assert observed["run_task_id"].startswith("cron_abc_")
        assert observed["container_task_id_during_run"] == observed["run_task_id"]
        assert observed["task_overrides_during_run"] == {
            "cwd": str(tmp_path.resolve()),
            "_force_task_isolation": True,
        }

        # And it remains the original value in finally; terminal overrides are cleared too.
        assert os.environ["TERMINAL_CWD"] == "/original/cwd"
        from tools.terminal_tool import resolve_task_overrides

        assert resolve_task_overrides(observed["run_task_id"]) == {}

    def test_workdir_terminal_override_failure_fails_job(self, tmp_path, monkeypatch):
        """Cron must fail closed if it cannot isolate a workdir job's terminal env."""
        import cron.scheduler as sched
        import tools.terminal_tool as terminal_tool

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        def boom(*_a, **_kw):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(terminal_tool, "register_task_env_overrides", boom)

        job = {
            "id": "abc",
            "name": "wd-job",
            "workdir": str(tmp_path),
            "schedule_display": "manual",
        }

        success, _output, _response, error = sched.run_job(job)
        assert success is False
        assert "failed to isolate cron workdir terminal environment" in (error or "")
        assert "run_task_id" not in observed

    def test_tool_cwd_resolution_prefers_contextvar(self, tmp_path, monkeypatch):
        """Tool cwd helpers must honor session ContextVar cwd before env cwd."""
        env_cwd = tmp_path / "env"
        ctx_cwd = tmp_path / "ctx"
        staging = tmp_path / "stage"
        env_cwd.mkdir()
        ctx_cwd.mkdir()
        staging.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(env_cwd))

        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.terminal_tool import _get_env_config
        from tools.file_tools import _configured_terminal_cwd
        from tools.code_execution_tool import _resolve_child_cwd

        tokens = set_session_vars(cwd=str(ctx_cwd))
        try:
            assert _get_env_config()["cwd"] == str(ctx_cwd)
            assert _configured_terminal_cwd() == str(ctx_cwd)
            assert _resolve_child_cwd("project", str(staging)) == str(ctx_cwd)
            monkeypatch.setenv("TERMINAL_ENV", "docker")
            monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
            docker_cfg = _get_env_config()
            assert docker_cfg["cwd"] == "/workspace"
            assert docker_cfg["host_cwd"] == str(ctx_cwd)
        finally:
            clear_session_vars(tokens)

    def test_no_workdir_leaves_terminal_cwd_untouched(self, monkeypatch):
        """When workdir is absent, run_job must not touch TERMINAL_CWD at all —
        whatever value was present before the call should be present after.

        We don't assert on the *content* of TERMINAL_CWD (other tests in the
        same xdist worker may leave it set to something like '.'); we just
        check it's unchanged by run_job.
        """
        import os
        import cron.scheduler as sched

        # Pin TERMINAL_CWD to a sentinel via monkeypatch so we control both
        # the before-value and the after-value regardless of cross-test state.
        monkeypatch.setenv("TERMINAL_CWD", "/cron-test-sentinel")
        before = os.environ["TERMINAL_CWD"]

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        job = {
            "id": "xyz",
            "name": "no-wd-job",
            "workdir": None,
            "schedule_display": "manual",
        }

        success, *_ = sched.run_job(job)
        assert success is True

        # Feature is OFF — skip_context_files stays True.
        assert observed["skip_context_files"] is True
        # Cron still forces SOUL.md identity even when cwd context files stay off.
        assert observed["load_soul_identity"] is True
        # TERMINAL_CWD saw the same value during init as it had before.
        assert observed["terminal_cwd_during_init"] == before
        # And after run_job completes, it's still the sentinel (nothing
        # overwrote or cleared it).
        assert os.environ["TERMINAL_CWD"] == before
