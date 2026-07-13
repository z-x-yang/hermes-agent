"""Product-branding contract for the private Evelyn distribution.

Evelyn is the user-facing product. Hermes Agent remains the upstream/runtime
compatibility namespace so existing scripts, protocols, bundle identifiers, and
configuration continue to work.
"""

from pathlib import Path
import json
import re
import struct
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _text(relative_path: str):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_evelyn_is_the_preferred_cli_without_breaking_hermes():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]

    assert scripts["evelyn"] == "hermes_cli.main:main"
    assert scripts["hermes"] == scripts["evelyn"]
    assert scripts["hermes-agent"] == "run_agent:main"

    from hermes_cli._parser import build_top_level_parser

    parser, _, _ = build_top_level_parser()
    help_text = parser.format_help()
    assert parser.prog == "evelyn"
    assert parser.description is not None
    assert "Evelyn" in parser.description
    assert "    evelyn" in help_text

    install_sh = _text("scripts/install.sh")
    assert '"$command_link_dir/evelyn"' in install_sh
    assert '"$command_link_dir/hermes"' in install_sh
    assert "command -v evelyn && command -v hermes" in install_sh

    manual_setup = _text("setup-hermes.sh")
    assert 'EVELYN_BIN="$SCRIPT_DIR/venv/bin/evelyn"' in manual_setup
    assert '"$COMMAND_LINK_DIR/evelyn"' in manual_setup
    assert '"$COMMAND_LINK_DIR/hermes"' in manual_setup
    assert ".evelyn-migrated-backup" in manual_setup
    assert "ln -sf" not in manual_setup


def test_desktop_builds_as_evelyn_with_hermes_compatibility_ids():
    package = _json("apps/desktop/package.json")
    build = package["build"]

    assert package["name"] == "hermes"  # npm workspace compatibility
    assert package["productName"] == "Evelyn"
    assert package["description"] == "Evelyn — powered by Hermes Agent."
    assert build["productName"] == "Evelyn"
    assert build["executableName"] == "Evelyn"
    assert build["artifactName"].startswith("Evelyn-")

    # These are stable compatibility identities, not user-facing branding.
    assert build["appId"] == "com.nousresearch.hermes"
    assert build["protocols"][0]["schemes"] == ["hermes"]

    photon_sidecar = _text("plugins/platforms/photon/sidecar/index.mjs")
    assert "X-Hermes-Sidecar-Token" in photon_sidecar
    assert "X-Evelyn-Sidecar-Token" not in photon_sidecar

    mac_info = build["mac"]["extendInfo"]
    assert mac_info["CFBundleDisplayName"] == "Evelyn"
    assert mac_info["CFBundleExecutable"] == "Evelyn"
    assert mac_info["CFBundleName"] == "Evelyn"
    assert build["dmg"]["title"] == "Install Evelyn"
    assert build["nsis"]["shortcutName"] == "Evelyn"
    assert build["nsis"]["uninstallDisplayName"] == "Evelyn"


def test_desktop_runtime_and_installer_display_evelyn():
    assert "const APP_NAME = 'Evelyn'" in _text("apps/desktop/electron/main.cjs")
    assert "<title>Evelyn</title>" in _text("apps/desktop/index.html")

    tauri = _json("apps/bootstrap-installer/src-tauri/tauri.conf.json")
    assert tauri["productName"] == "Evelyn"
    assert tauri["app"]["windows"][0]["title"] == "Evelyn"
    assert tauri["identifier"] == "com.nousresearch.hermes.setup"
    assert 'name = "Evelyn-Setup"' in _text("apps/bootstrap-installer/src-tauri/Cargo.toml")


def test_visual_identity_uses_discord_derived_evelyn_assets():
    def png_shape(relative_path: str) -> tuple[int, int, int]:
        data = (ROOT / relative_path).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n"), relative_path
        width, height = struct.unpack(">II", data[16:24])
        color_type = data[25]
        return width, height, color_type

    assert png_shape("apps/desktop/assets/icon.png")[:2] == (1024, 1024)
    assert png_shape("apps/desktop/public/apple-touch-icon.png")[:2] == (1024, 1024)
    assert png_shape("apps/desktop/public/evelyn-brand-mark.png") == (1024, 1024, 6)
    assert png_shape("apps/bootstrap-installer/public/evelyn-brand-mark.png") == (1024, 1024, 6)
    assert png_shape("apps/bootstrap-installer/src-tauri/icons/32x32.png")[:2] == (32, 32)
    assert png_shape("apps/bootstrap-installer/src-tauri/icons/128x128.png")[:2] == (128, 128)
    assert png_shape("apps/bootstrap-installer/src-tauri/icons/128x128@2x.png")[:2] == (256, 256)

    assert (ROOT / "apps/desktop/assets/icon.icns").read_bytes().startswith(b"icns")
    assert (ROOT / "apps/desktop/assets/icon.ico").read_bytes().startswith(b"\x00\x00\x01\x00")
    assert (ROOT / "apps/bootstrap-installer/src-tauri/icons/icon.icns").read_bytes().startswith(b"icns")
    assert (ROOT / "apps/bootstrap-installer/src-tauri/icons/icon.ico").read_bytes().startswith(b"\x00\x00\x01\x00")

    for component in (
        "apps/desktop/src/components/brand-mark.tsx",
        "apps/bootstrap-installer/src/components/brand-mark.tsx",
    ):
        source = _text(component)
        assert "evelyn-brand-mark.png" in source
        assert "nous-girl" not in source

    for retired in (
        "apps/desktop/public/nous-girl.jpg",
        "apps/bootstrap-installer/public/nous-girl.jpg",
        "apps/desktop/public/hermes.png",
        "apps/desktop/public/hermes-sprite.png",
        "apps/desktop/public/hermes-frames",
    ):
        assert not (ROOT / retired).exists(), retired

    banner = _text("hermes_cli/banner.py")
    assert "EVELYN_PORTRAIT" in banner
    assert "☾" in banner
    assert "HERMES_CADUCEUS = EVELYN_PORTRAIT" in banner
    assert "else EVELYN_PORTRAIT" in banner
    assert "_hero = EVELYN_PORTRAIT" in banner
    assert "⣴⣾⣿⣿⣇⠸⣿⣿" not in banner  # retired caduceus art


def test_secondary_user_interfaces_display_evelyn():
    assert "name: 'Evelyn'" in _text("ui-tui/src/theme.ts")
    web_locale = _text("web/src/i18n/en.ts")
    assert 'brand: "Evelyn"' in web_locale
    assert 'brandShort: "E"' in web_locale
    assert "Gateway online — Evelyn is back and ready." in _text("gateway/run.py")


def test_desktop_launchers_prefer_evelyn_and_accept_legacy_hermes_artifacts():
    migration_surfaces = [
        "apps/desktop/electron/main.cjs",
        "hermes_cli/main.py",
        "scripts/install.sh",
        "scripts/install.ps1",
        "apps/bootstrap-installer/src-tauri/src/bootstrap.rs",
        "apps/bootstrap-installer/src-tauri/src/update.rs",
    ]
    for relative_path in migration_surfaces:
        source = _text(relative_path)
        assert "Evelyn.app" in source or "Evelyn.exe" in source or '"Evelyn"' in source, relative_path
        assert "Hermes.app" in source or "Hermes.exe" in source or '"Hermes"' in source, relative_path

    gui_uninstall = _text("hermes_cli/gui_uninstall.py")
    assert 'Path("/Applications/Evelyn.app")' in gui_uninstall
    assert 'Path("/Applications/Hermes.app")' in gui_uninstall
    assert '"Application Support" / "Evelyn"' in gui_uninstall
    assert 'primary.with_name("Hermes")' in gui_uninstall
    assert '"Evelyn.lnk"' in gui_uninstall
    assert '"Hermes.lnk"' in gui_uninstall

    install_ps1 = _text("scripts/install.ps1")
    assert "'Evelyn.lnk'" in install_ps1
    assert "'Hermes.lnk'" in install_ps1
    assert "Move-Item -LiteralPath $legacyLnk -Destination $legacyBackup" in install_ps1
    assert ".evelyn-migrated-backup" in install_ps1
    assert "Remove-Item -LiteralPath $legacyLnk" not in install_ps1

    desktop_main = _text("apps/desktop/electron/main.cjs")
    assert "findOnPath('evelyn') || findOnPath('hermes')" in desktop_main
    assert "IS_WINDOWS ? 'evelyn.exe' : 'evelyn'" in desktop_main
    assert "IS_WINDOWS ? 'hermes.exe' : 'hermes'" in desktop_main
    assert "macosAppMigrationPaths(targetApp)" in desktop_main
    assert "Could not archive legacy app" in desktop_main
    assert desktop_main.index("app.requestSingleInstanceLock()") < desktop_main.index(
        "migrateLegacyDesktopStateForApp({"
    )

    desktop_harness = _text("apps/desktop/scripts/test-desktop.mjs")
    assert "'Evelyn.app'" in desktop_harness
    assert "'Evelyn.exe'" in desktop_harness
    assert "`Evelyn-${PACKAGE_JSON.version}" in desktop_harness


def test_localized_product_copy_uses_evelyn_not_legacy_brand():
    desktop_locales = [
        ROOT / "apps/desktop/src/i18n/en.ts",
        ROOT / "apps/desktop/src/i18n/zh.ts",
        ROOT / "apps/desktop/src/i18n/zh-hant.ts",
        ROOT / "apps/desktop/src/i18n/ja.ts",
    ]
    web_locales = [
        path
        for path in (ROOT / "web/src/i18n").glob("*.ts")
        if path.name not in {"index.ts", "types.ts"} and not path.name.endswith(".test.ts")
    ]
    assert web_locales
    for path in [*desktop_locales, *web_locales]:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"\bHermes\b", source), path.relative_to(ROOT)
