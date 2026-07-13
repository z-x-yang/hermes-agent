//! Filesystem paths + logging setup.
//!
//! Mirrors `hermes_constants.get_hermes_home()` from the Python CLI. Fresh
//! installs use the Evelyn namespace; an existing Hermes root remains the
//! fallback so state is not split across two homes.
//!
//! IMPORTANT: this must match exactly. Drift here means install.ps1
//! writes to one place and the installer reads from another, breaking
//! the bootstrap-complete check.

use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::process::Command;
use tracing_appender::non_blocking::WorkerGuard;

fn nonempty_env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).and_then(|value| {
        if value.to_string_lossy().trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(value))
        }
    })
}

fn select_home(preferred: PathBuf, legacy: &[PathBuf]) -> PathBuf {
    if preferred.is_dir() {
        return preferred;
    }
    for candidate in legacy {
        if candidate.is_dir() {
            return candidate.clone();
        }
    }
    preferred
}

/// Returns the canonical data home. EVELYN_HOME is preferred; HERMES_HOME is
/// the compatibility override for existing automation and profiles.
pub fn hermes_home() -> PathBuf {
    if let Some(override_path) = nonempty_env("EVELYN_HOME") {
        return override_path;
    }
    if let Some(override_path) = nonempty_env("HERMES_HOME") {
        return override_path;
    }

    #[cfg(target_os = "windows")]
    {
        let home = dirs::home_dir();
        let local_app_data = dirs::data_local_dir()
            .or_else(|| home.as_ref().map(|path| path.join("AppData").join("Local")));
        if let Some(local_app_data) = local_app_data {
            let preferred = local_app_data.join("evelyn");
            let legacy_local = local_app_data.join("hermes");
            let legacy_dot = home.map(|path| path.join(".hermes"));
            let legacy = legacy_dot
                .into_iter()
                .fold(vec![legacy_local], |mut paths, path| {
                    paths.push(path);
                    paths
                });
            return select_home(preferred, &legacy);
        }
    }

    // macOS + Linux.
    #[cfg(not(target_os = "windows"))]
    if let Some(home) = dirs::home_dir() {
        return select_home(home.join(".evelyn"), &[home.join(".hermes")]);
    }

    // Last resort — current dir, almost certainly wrong but at least
    // doesn't panic.
    PathBuf::from(".evelyn")
}

pub fn log_dir() -> PathBuf {
    hermes_home().join("logs")
}

pub fn log_path() -> PathBuf {
    log_dir().join("bootstrap-installer.log")
}

pub fn bootstrap_cache_dir() -> PathBuf {
    hermes_home().join("bootstrap-cache")
}

/// Stable location the installer copies itself to after a successful install.
/// The desktop app re-invokes this with `--update`, and the start-menu /
/// desktop shortcuts can point users back to it. Lives directly under
/// HERMES_HOME so it survives repo checkout deletion (unlike anything under
/// hermes-agent/).
///
/// On Windows this is `%LOCALAPPDATA%\hermes\hermes-setup.exe`; on other
/// platforms the extension differs but the directory is the same.
pub fn installer_dest() -> PathBuf {
    let name = if cfg!(target_os = "windows") {
        "hermes-setup.exe"
    } else {
        "hermes-setup"
    };
    hermes_home().join(name)
}

/// Marker the updater writes for the duration of an in-app update and removes
/// when it finishes (see update.rs `UpdateMarkerGuard`). A freshly-launched
/// desktop checks this before spawning its own local backend: spawning one
/// mid-update re-locks the venv shim and triggers `force_kill_other_hermes`,
/// which then kills that legitimate backend in a respawn loop (#50238).
///
/// Lives directly under HERMES_HOME (same rationale as `installer_dest`) so the
/// Electron desktop — which resolves HERMES_HOME identically and pins it into
/// the updater's env — agrees on the exact path.
pub fn update_in_progress_marker() -> PathBuf {
    hermes_home().join(".hermes-update-in-progress")
}

/// Copy the currently-running installer binary to `installer_dest()` so it's
/// available for future `--update` runs and shortcut launches.
///
/// No-ops (returns Ok) when the running exe is ALREADY the destination — which
/// is exactly the case during an `--update` run (the desktop launched us FROM
/// that path), where copying onto ourselves would be a Windows sharing
/// violation. Best-effort: a failure here must not fail the install, so the
/// caller logs and continues.
pub fn copy_self_to_hermes_home() -> std::io::Result<()> {
    let src = std::env::current_exe()?;
    let dest = installer_dest();

    // Skip if we're already running from the destination (update re-invocation
    // or a prior copy). canonicalize both so symlinks / 8.3 short paths / case
    // differences don't trick us into a self-copy.
    let same = match (src.canonicalize(), dest.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => src == dest,
    };
    if same {
        tracing::info!(?dest, "installer already at destination; skipping self-copy");
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(&src, &dest)?;
    repair_macos_installer_helper(&dest);
    tracing::info!(?src, ?dest, "copied installer to HERMES_HOME");
    Ok(())
}

#[cfg(target_os = "macos")]
fn repair_macos_installer_helper(path: &Path) {
    // The staged helper may inherit quarantine from the downloaded installer.
    // Desktop later launches this exact file for in-app updates, so make it
    // executable before the update handoff reaches LaunchServices/Gatekeeper.
    let _ = Command::new("/usr/bin/xattr")
        .args(["-cr"])
        .arg(path)
        .status();

    let verify = Command::new("/usr/bin/codesign")
        .arg("--verify")
        .arg(path)
        .status();

    if !matches!(verify, Ok(status) if status.success()) {
        let _ = Command::new("/usr/bin/codesign")
            .args(["--force", "--sign", "-"])
            .arg(path)
            .status();
    }
}

#[cfg(not(target_os = "macos"))]
fn repair_macos_installer_helper(_path: &Path) {}

/// Where install.ps1 writes the bootstrap-complete marker (existence-only file
/// the Electron app also checks). Per main.cjs:
///   const BOOTSTRAP_COMPLETE_MARKER = path.join(ACTIVE_HERMES_ROOT, '.hermes-bootstrap-complete')
/// We don't always know ACTIVE_HERMES_ROOT until install.ps1 reports it, so
/// this is a probe helper, not a definitive path.
pub fn likely_bootstrap_marker(install_root: &Path) -> PathBuf {
    install_root.join(".hermes-bootstrap-complete")
}

/// Initializes tracing to bootstrap-installer.log under HERMES_HOME/logs/.
/// Returns a guard that flushes the appender on drop — keep it alive for
/// the lifetime of the process.
pub fn init_logging() -> Option<WorkerGuard> {
    let dir = log_dir();
    if let Err(err) = std::fs::create_dir_all(&dir) {
        // No log dir → log to stderr only. Don't panic; the installer
        // should still be usable on an exotic filesystem.
        eprintln!("[hermes-setup] could not create log dir {dir:?}: {err}");
        return None;
    }

    let file_appender = tracing_appender::rolling::never(&dir, "bootstrap-installer.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = tracing_subscriber::EnvFilter::try_from_env("HERMES_BOOTSTRAP_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(true)
        .init();

    Some(guard)
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn get_log_path() -> String {
    log_path().to_string_lossy().into_owned()
}

#[tauri::command]
pub fn get_hermes_home() -> String {
    hermes_home().to_string_lossy().into_owned()
}

#[tauri::command]
pub fn open_log_dir(app: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let path = log_dir();
    app.opener()
        .open_path(path.to_string_lossy(), None::<&str>)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::select_home;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sandbox(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("evelyn-home-{name}-{nonce}"));
        fs::create_dir_all(&root).expect("sandbox");
        root
    }

    #[test]
    fn fresh_install_selects_evelyn() {
        let root = sandbox("fresh");
        assert_eq!(
            select_home(root.join(".evelyn"), &[root.join(".hermes")]),
            root.join(".evelyn")
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn existing_hermes_root_is_legacy_fallback() {
        let root = sandbox("legacy");
        fs::create_dir(root.join(".hermes")).expect("legacy root");
        assert_eq!(
            select_home(root.join(".evelyn"), &[root.join(".hermes")]),
            root.join(".hermes")
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn first_existing_legacy_candidate_wins() {
        let root = sandbox("legacy-order");
        let local_legacy = root.join("local-hermes");
        let dot_legacy = root.join(".hermes");
        fs::create_dir(&local_legacy).expect("local legacy root");
        fs::create_dir(&dot_legacy).expect("dot legacy root");
        assert_eq!(
            select_home(root.join(".evelyn"), &[local_legacy.clone(), dot_legacy]),
            local_legacy
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn existing_evelyn_root_wins_when_both_exist() {
        let root = sandbox("both");
        fs::create_dir(root.join(".evelyn")).expect("evelyn root");
        fs::create_dir(root.join(".hermes")).expect("hermes root");
        assert_eq!(
            select_home(root.join(".evelyn"), &[root.join(".hermes")]),
            root.join(".evelyn")
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn preferred_file_does_not_shadow_legacy_directory() {
        let root = sandbox("preferred-file");
        fs::create_dir_all(&root).expect("sandbox root");
        fs::write(root.join(".evelyn"), b"not a directory").expect("preferred file");
        let legacy = root.join(".hermes");
        fs::create_dir(&legacy).expect("legacy root");
        assert_eq!(select_home(root.join(".evelyn"), &[legacy.clone()]), legacy);
        fs::remove_dir_all(root).ok();
    }
}
