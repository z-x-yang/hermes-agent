"""Regression tests for sudo detection and terminal background handling."""

import time

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_background_notify_metadata_is_checkpointed_after_session_setup(monkeypatch, tmp_path):
    """Gateway restart recovery needs watcher metadata persisted after spawn."""
    import tools.process_registry as pr_module
    from tools.process_registry import ProcessSession

    class FakeEnv:
        cwd = str(tmp_path)
        timeout = 10
        env = {}

    proc_session = ProcessSession(
        id="proc_checkpoint",
        command="sleep 60",
        task_id="default",
        session_key="discord-session",
        pid=12345,
        started_at=time.time(),
    )
    checkpoint_snapshots = []

    def fake_checkpoint():
        checkpoint_snapshots.append({
            "watcher_platform": proc_session.watcher_platform,
            "watcher_chat_id": proc_session.watcher_chat_id,
            "watcher_thread_id": proc_session.watcher_thread_id,
            "watcher_user_id": proc_session.watcher_user_id,
            "watcher_user_name": proc_session.watcher_user_name,
            "watcher_message_id": proc_session.watcher_message_id,
            "watcher_interval": proc_session.watcher_interval,
            "notify_on_complete": proc_session.notify_on_complete,
            "watch_patterns": list(proc_session.watch_patterns),
        })

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **_kw: FakeEnv())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *_a, **_kw: {"approved": True},
    )
    monkeypatch.setattr(
        pr_module.process_registry,
        "spawn_local",
        lambda **_kw: proc_session,
    )
    monkeypatch.setattr(pr_module.process_registry, "_write_checkpoint", fake_checkpoint)
    pr_module.process_registry.pending_watchers = []
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "thread-2")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-3")
    monkeypatch.setenv("HERMES_SESSION_USER_NAME", "Zongxin")
    monkeypatch.setenv("HERMES_SESSION_MESSAGE_ID", "message-4")

    try:
        result_json = terminal_tool.terminal_tool(
            command="sleep 60",
            background=True,
            task_id="discord-session",
            notify_on_complete=True,
        )
    finally:
        terminal_tool._active_environments.clear()
        terminal_tool._last_activity.clear()
        pr_module.process_registry.pending_watchers = []

    assert '"exit_code": 0' in result_json
    assert checkpoint_snapshots == [{
        "watcher_platform": "discord",
        "watcher_chat_id": "channel-1",
        "watcher_thread_id": "thread-2",
        "watcher_user_id": "user-3",
        "watcher_user_name": "Zongxin",
        "watcher_message_id": "message-4",
        "watcher_interval": 5,
        "notify_on_complete": True,
        "watch_patterns": [],
    }]


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_terminal_schema_advertises_persistent_env_state():
    description = terminal_tool.TERMINAL_TOOL_DESCRIPTION

    assert "exported environment variables persist between calls" in description
    assert "activate a virtualenv" in description
    assert "do not re-source the same environment before every command" in description
    assert "watch_patterns" in description
    assert "do not rely on them to survive a gateway restart" in description


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_actual_sudo_after_leading_env_assignment_is_rewritten(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("DEBUG=1 sudo whoami")

    assert transformed == "DEBUG=1 sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_cached_sudo_password_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool._set_cached_sudo_password("cached-pass")

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("echo ok && sudo whoami")

    assert transformed == "echo ok && sudo -S -p '' whoami"
    assert sudo_stdin == "cached-pass\n"


def test_registered_sudo_callback_is_used_without_interactive_env(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: False)

    calls = []

    def sudo_callback():
        calls.append("called")
        return "callback-pass"

    terminal_tool.set_sudo_password_callback(sudo_callback)
    try:
        transformed, sudo_stdin = terminal_tool._transform_sudo_command(
            "echo ok | sudo tee /tmp/hermes-test"
        )
    finally:
        terminal_tool.set_sudo_password_callback(None)

    assert calls == ["called"]
    assert transformed == "echo ok | sudo -S -p '' tee /tmp/hermes-test"
    assert sudo_stdin == "callback-pass\n"


def test_cached_sudo_password_isolated_by_session_key(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    terminal_tool._set_cached_sudo_password("alpha-pass")

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-b")
    assert terminal_tool._get_cached_sudo_password() == ""

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    assert terminal_tool._get_cached_sudo_password() == "alpha-pass"


def test_passwordless_sudo_skips_interactive_prompt_and_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError(
            "interactive sudo prompt should not run when sudo -n already works"
        )

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: True, raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo whoami")

    assert transformed == "sudo whoami"
    assert sudo_stdin is None


def test_passwordless_sudo_probe_rechecks_local_terminal(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    calls = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result(0 if len(calls) == 1 else 1)

    monkeypatch.setattr(terminal_tool.subprocess, "run", fake_run)

    assert terminal_tool._sudo_nopasswd_works() is True
    assert terminal_tool._sudo_nopasswd_works() is False
    assert len(calls) == 2
    assert calls[0][0] == ["sudo", "-n", "true"]
    assert calls[1][0] == ["sudo", "-n", "true"]


def test_passwordless_sudo_probe_is_disabled_for_nonlocal_terminal_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    def _fail_run(*_args, **_kwargs):
        raise AssertionError("host sudo probe must not run for non-local terminal envs")

    monkeypatch.setattr(terminal_tool.subprocess, "run", _fail_run)

    assert terminal_tool._sudo_nopasswd_works() is False


def test_validate_workdir_allows_windows_drive_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project") is None
    assert terminal_tool._validate_workdir("C:/Users/Alice/project") is None


def test_validate_workdir_allows_windows_unc_paths():
    assert terminal_tool._validate_workdir(r"\\server\share\project") is None


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_get_env_config_ignores_bad_docker_json_for_local_backend(monkeypatch):
    """Docker-only JSON env vars must not break the default local backend."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", "not-json")
    monkeypatch.setenv("TERMINAL_DOCKER_FORWARD_ENV", "not-json")
    monkeypatch.setenv("TERMINAL_DOCKER_EXTRA_ARGS", "not-json")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "local"
    assert config["docker_volumes"] == []
    assert config["docker_env"] == {}
    assert config["docker_forward_env"] == []
    assert config["docker_extra_args"] == []


def test_get_env_config_ignores_bad_docker_json_for_ssh_backend(monkeypatch):
    """Non-container remote backends should also ignore Docker-only JSON."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", "not-json")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["docker_volumes"] == []
    assert config["docker_env"] == {}


def test_get_env_config_still_rejects_bad_docker_json_for_docker_backend(monkeypatch):
    """Selecting Docker should keep the existing actionable config error."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")

    try:
        terminal_tool._get_env_config()
    except ValueError as exc:
        assert "TERMINAL_DOCKER_VOLUMES" in str(exc)
    else:
        raise AssertionError("Docker backend must validate TERMINAL_DOCKER_VOLUMES")


def test_sudo_wrong_password_failure_detects_rejection_output():
    output = (
        "sudo: Authentication failed, try again.\n\n"
        "sudo: maximum 3 incorrect authentication attempts\n"
    )
    assert terminal_tool._sudo_wrong_password_failure(output) is True


def test_sudo_wrong_password_failure_ignores_tty_required_message():
    output = "sudo: a terminal is required to authenticate"
    assert terminal_tool._sudo_wrong_password_failure(output) is False


def test_invalidate_cached_sudo_on_auth_failure_clears_session_cache(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    terminal_tool._set_cached_sudo_password("wrong-pass")

    cleared = terminal_tool._invalidate_cached_sudo_on_auth_failure(
        "sudo apt install fprintd",
        "sudo: Authentication failed, try again.",
    )

    assert cleared is True
    assert terminal_tool._get_cached_sudo_password() == ""


def test_invalidate_cached_sudo_on_auth_failure_keeps_env_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "from-env")
    terminal_tool._set_cached_sudo_password("wrong-pass")

    cleared = terminal_tool._invalidate_cached_sudo_on_auth_failure(
        "sudo true",
        "sudo: Authentication failed, try again.",
    )

    assert cleared is False
    assert terminal_tool._get_cached_sudo_password() == "wrong-pass"


def test_transform_sudo_command_pipes_one_password_line_per_invocation(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command(
        "sudo true && sudo whoami"
    )

    assert transformed == "sudo -S -p '' true && sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\ntestpass\n"


def test_count_real_sudo_invocations_ignores_mentions(monkeypatch):
    assert terminal_tool._count_real_sudo_invocations("grep sudo README.md") == 0
    assert terminal_tool._count_real_sudo_invocations("sudo a; sudo b") == 2
