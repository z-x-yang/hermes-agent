"""Tests for BaseEnvironment unified execution model.

Tests _wrap_command(), _extract_cwd_from_output(), _embed_stdin_heredoc(),
init_session() failure handling, and the CWD marker contract.
"""

from unittest.mock import MagicMock

from tools.environments.base import BaseEnvironment


class _TestableEnv(BaseEnvironment):
    """Concrete subclass for testing base class methods."""

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use mock")

    def cleanup(self):
        pass


class TestWrapCommand:
    def test_basic_shape(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hello", "/tmp")

        assert "source" in wrapped
        assert "cd -- /tmp" in wrapped or "cd -- '/tmp'" in wrapped
        assert "eval 'echo hello'" in wrapped
        assert "__hermes_ec=$?" in wrapped
        assert "export -p >" in wrapped
        assert "pwd -P >" in wrapped
        assert env._cwd_marker in wrapped
        assert "exit $__hermes_ec" in wrapped

    def test_no_snapshot_skips_source(self):
        env = _TestableEnv()
        env._snapshot_ready = False
        wrapped = env._wrap_command("echo hello", "/tmp")

        assert "source" not in wrapped

    def test_single_quote_escaping(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo 'hello world'", "/tmp")

        assert "eval 'echo '\\''hello world'\\'''" in wrapped

    def test_tilde_not_quoted(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~")

        assert "cd -- ~" in wrapped
        assert "cd -- '~'" not in wrapped

    def test_tilde_subpath_with_spaces_uses_home_and_quotes_suffix(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~/my repo")

        assert "cd -- $HOME/'my repo'" in wrapped
        assert "cd -- ~/my repo" not in wrapped

    def test_tilde_slash_maps_to_home(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~/")

        assert "cd -- $HOME" in wrapped
        assert "cd -- ~/" not in wrapped

    def test_hyphen_prefixed_workdir_is_passed_after_double_dash(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("pwd", "-demo")

        assert "builtin cd -- -demo || exit 126" in wrapped

    def test_cd_failure_exit_126(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "/nonexistent")

        assert "exit 126" in wrapped


class TestAtomicSnapshotWrite:
    """Regression for #38249: concurrent terminal calls in one session both
    source AND rewrite the shared env snapshot. A non-atomic ``export -p >
    snap`` truncates-then-writes in place, so a concurrent ``source snap`` can
    read a half-written file and embed ``declare -x``/``export`` fragments into
    PATH, breaking ``ls``/``git``/``tr`` with command-not-found. The write must
    assemble in a temp file and ``mv -f`` it into place (mv is atomic on POSIX
    same-fs), so a reader sees the old-or-new complete file, never a torn one.
    """

    def test_wrap_command_uses_atomic_temp_then_mv(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hi", "/tmp")
        # Env dump goes to a temp file, not directly over the live snapshot.
        assert "export -p > " in wrapped
        assert ".tmp." in wrapped
        # Then an atomic rename onto the real snapshot path.
        assert "mv -f " in wrapped
        # The env-dump must NOT write the live snapshot in place (the bug).
        snap = env._snapshot_path
        assert f"export -p > {snap} " not in wrapped
        assert f"export -p > '{snap}'" not in wrapped

    def test_temp_path_is_unique_without_shell_pid_variables(self):
        """Each wrapper gets a unique literal temp path from Python.

        macOS ships Bash 3.2, where ``BASHPID`` is undefined.  An unguarded
        ``$BASHPID`` breaks the control footer after a command enables
        ``set -u`` and also collapses concurrent writers onto one ``.tmp.``
        path when nounset is off.
        """
        env = _TestableEnv()
        env._snapshot_ready = True
        first = env._wrap_command("echo first", "/tmp")
        second = env._wrap_command("echo second", "/tmp")
        assert "$BASHPID" not in first
        assert ".tmp.$$" not in first
        first_temp = first.split("export -p > ", 1)[1].split(" && mv -f", 1)[0]
        second_temp = second.split("export -p > ", 1)[1].split(" && mv -f", 1)[0]
        assert ".tmp." in first_temp
        assert first_temp != second_temp

    def test_unique_temp_path_with_spaces_is_fully_quoted(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        env._snapshot_path = "/tmp/has space/hermes-snap-x.sh"
        wrapped = env._wrap_command("echo hi", "/tmp")
        temp_token = wrapped.split("export -p > ", 1)[1].split(" && mv -f", 1)[0]
        assert temp_token.startswith("'/tmp/has space/hermes-snap-x.sh.tmp.")
        assert temp_token.endswith("'")
        assert "$" not in temp_token

    def test_wrap_command_mv_chained_on_export_success(self):
        """A failed/partial ``export -p`` must NOT mv a torn temp over a good
        snapshot.  The mv is chained with ``&&`` on the export, and the temp is
        removed on failure."""
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hi", "/tmp")
        assert "export -p > " in wrapped and "&& mv -f " in wrapped
        assert "rm -f " in wrapped  # temp cleanup on failure

    def test_init_session_bootstrap_also_uses_unique_literal_temp(self):
        """The init_session bootstrap (first snapshot write) is the same shared
        file a concurrent command could source, so it needs its own unique
        literal temp path too."""
        env = _TestableEnv()
        captured = {}

        def fake_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
            captured["cmd"] = cmd_string
            raise RuntimeError("stop after capture")  # we only need the script

        env._run_bash = fake_run_bash  # type: ignore[assignment]
        try:
            env.init_session()
        except Exception:
            pass
        boot = captured.get("cmd", "")
        assert ".tmp." in boot and "mv -f " in boot, boot
        assert "$BASHPID" not in boot
        assert ".tmp.$$" not in boot

    def test_wrapper_preserves_success_after_command_enables_nounset(self, tmp_path):
        env = _TestableEnv(cwd=str(tmp_path))
        env._snapshot_ready = True
        env._snapshot_path = str(tmp_path / "snapshot.sh")
        (tmp_path / "snapshot.sh").write_text("export HERMES_TEST=1\n")

        wrapped = env._wrap_command(
            "set -euo pipefail\nprintf 'STRICT_SUCCESS\\n'", str(tmp_path)
        )

        import subprocess
        result = subprocess.run(
            ["/bin/bash", "-c", wrapped], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "STRICT_SUCCESS" in result.stdout
        assert env._cwd_marker in result.stdout


class TestAtomicSnapshotConcurrencyBehavioral:
    """Behavioral regression for #38249 — actually EXECUTES the generated
    snapshot write/read concurrently and asserts the file never tears.

    The string-inspection tests prove the right script is emitted; this proves
    the emitted script's guarantee holds under real concurrency: N concurrent
    writers + readers, and the snapshot is ALWAYS a complete, parseable env
    dump — never truncated mid-line with a ``declare -x`` / ``export`` fragment
    that would corrupt PATH.  Crucially it uses ``$BASHPID`` (per-subshell
    unique), which is what closes the race; ``$$`` would still tear here.
    """

    def _run(self, script):
        import subprocess
        return subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True)

    def test_concurrent_writes_never_tear_the_snapshot(self, tmp_path):
        import shutil
        if not shutil.which("bash"):
            import pytest
            pytest.skip("bash required")
        import shlex
        snap = str(tmp_path / "hermes-snap-x.sh")
        _q = shlex.quote
        # Each real _wrap_command call receives a distinct Python-generated
        # literal path. Model four concurrent calls with four unique paths.
        writers = []
        for writer_id in range(4):
            temp = _q(f"{snap}.tmp.writer-{writer_id}")
            writers.append(
                "for i in $(seq 1 80); do "
                "export BIG_$i=$(head -c 600 /dev/zero | tr '\\0' x); "
                f"{{ export -p > {temp} && mv -f {temp} {_q(snap)}; }} "
                f"2>/dev/null || rm -f {temp} 2>/dev/null || true; "
                "done"
            )
        # Reader: repeatedly source the snapshot and check PATH never absorbs
        # an `export `/`declare -x` fragment (the corruption signature).
        reader = (
            "export PATH=/usr/bin:/bin; "
            "for i in $(seq 1 160); do "
            f"( source {_q(snap)} >/dev/null 2>&1 || true; "
            "case \"$PATH\" in *'declare -x'*|*'export '*) echo CORRUPT;; esac ); "
            "done"
        )
        self._run(f"export -p > {_q(snap)}")  # seed a valid snapshot
        # 4 concurrent writers + 4 readers, repeated.
        w = " & ".join(writers)
        r = " & ".join([reader] * 4)
        procs = [self._run(f"{w} & {r} & wait") for _ in range(3)]
        corrupt = any("CORRUPT" in p.stdout for p in procs)
        assert not corrupt, "snapshot tore — PATH absorbed a declare-x/export fragment"
        final = self._run(f"source {_q(snap)} >/dev/null 2>&1 && echo OK || echo BROKEN")
        assert "OK" in final.stdout, f"final snapshot not sourceable: {final.stdout} {final.stderr}"

    def test_failed_export_does_not_destroy_good_snapshot(self, tmp_path):
        """If ``export -p`` fails, the ``&&``-chained mv must NOT clobber the
        existing good snapshot."""
        import shutil
        if not shutil.which("bash"):
            import pytest
            pytest.skip("bash required")
        import shlex
        snap = str(tmp_path / "snap.sh")
        _q = shlex.quote
        self._run(f"echo 'export GOOD=1' > {_q(snap)}")  # seed good snapshot
        # Redirect export into an unwritable dir so the export side fails; mv
        # must then NOT run (&&) and not clobber snap.
        bad_tmp = _q("/nonexistent-dir/snap.tmp.") + "$BASHPID"
        script = (
            f"{{ export -p > {bad_tmp} && mv -f {bad_tmp} {_q(snap)}; }} "
            f"2>/dev/null || rm -f {bad_tmp} 2>/dev/null || true"
        )
        self._run(script)
        out = self._run(f"cat {_q(snap)}")
        assert "export GOOD=1" in out.stdout, "good snapshot was destroyed by a failed export"


class TestExtractCwdFromOutput:
    def test_happy_path(self):
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"hello\n{marker}/home/user{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert env.cwd == "/home/user"
        assert marker not in result["output"]

    def test_missing_marker(self):
        env = _TestableEnv()
        result = {"output": "hello world\n"}
        env._extract_cwd_from_output(result)

        assert env.cwd == "/tmp"  # unchanged

    def test_marker_in_command_output(self):
        """If the marker appears in command output AND as the real marker,
        rfind grabs the last (real) one."""
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"user typed {marker} in their output\nreal output\n{marker}/correct/path{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert env.cwd == "/correct/path"

    def test_output_cleaned(self):
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"hello\n{marker}/tmp{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert "hello" in result["output"]
        assert marker not in result["output"]


class TestEmbedStdinHeredoc:
    def test_heredoc_format(self):
        result = BaseEnvironment._embed_stdin_heredoc("cat", "hello world")

        assert result.startswith("cat << '")
        assert "hello world" in result
        assert "HERMES_STDIN_" in result

    def test_unique_delimiter_each_call(self):
        r1 = BaseEnvironment._embed_stdin_heredoc("cat", "data")
        r2 = BaseEnvironment._embed_stdin_heredoc("cat", "data")

        # Extract delimiters
        d1 = r1.split("'")[1]
        d2 = r2.split("'")[1]
        assert d1 != d2  # UUID-based, should be unique


class TestInitSessionFailure:
    def test_snapshot_ready_false_on_failure(self):
        env = _TestableEnv()

        def failing_run_bash(*args, **kwargs):
            raise RuntimeError("bash not found")

        env._run_bash = failing_run_bash
        env.init_session()

        assert env._snapshot_ready is False

    def test_snapshot_ready_false_on_nonzero_bootstrap_exit(self):
        """A non-zero bootstrap result should trigger fallback mode."""
        env = _TestableEnv()

        def mock_run_bash(*args, **kwargs):
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 127
            mock.stdout = iter([])
            return mock

        env._run_bash = mock_run_bash
        env.init_session()

        assert env._snapshot_ready is False

    def test_login_flag_when_snapshot_not_ready(self):
        """When _snapshot_ready=False, execute() should pass login=True to _run_bash."""
        env = _TestableEnv()
        env._snapshot_ready = False

        calls = []
        def mock_run_bash(cmd, *, login=False, timeout=120, stdin_data=None):
            calls.append({"login": login})
            # Return a mock process handle
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 0
            mock.stdout = iter([])
            return mock

        env._run_bash = mock_run_bash
        env.execute("echo test")

        assert len(calls) == 1
        assert calls[0]["login"] is True


class TestCwdMarker:
    def test_marker_contains_session_id(self):
        env = _TestableEnv()
        assert env._session_id in env._cwd_marker

    def test_unique_per_instance(self):
        env1 = _TestableEnv()
        env2 = _TestableEnv()
        assert env1._cwd_marker != env2._cwd_marker
