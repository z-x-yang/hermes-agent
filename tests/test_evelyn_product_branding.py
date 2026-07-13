"""Product-branding contract for the private Evelyn distribution.

Evelyn is the user-facing product. Hermes Agent remains the upstream/runtime
compatibility namespace so existing scripts, protocols, bundle identifiers, and
configuration continue to work.
"""

from pathlib import Path
import json
import os
import re
import struct
import subprocess
import tomllib
import zlib


ROOT = Path(__file__).resolve().parents[1]


def _json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _text(relative_path: str):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _png_corner_alphas(data: bytes) -> tuple[int, int, int, int]:
    """Decode 8-bit RGBA PNG rows and return the four corner alpha values."""
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    width = height = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += length + 12
        if kind == b"IHDR":
            width, height, depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            assert depth == 8 and color_type == 6 and interlace == 0
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break

    assert width and height
    raw = zlib.decompress(bytes(compressed))
    stride = width * 4
    rows: list[bytearray] = []

    def paeth(a: int, b: int, c: int) -> int:
        estimate = a + b - c
        distances = (abs(estimate - a), abs(estimate - b), abs(estimate - c))
        return (a, b, c)[distances.index(min(distances))]

    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        filtered = raw[cursor : cursor + stride]
        cursor += stride
        row = bytearray(stride)
        previous = rows[-1] if rows else bytearray(stride)
        for index, value in enumerate(filtered):
            left = row[index - 4] if index >= 4 else 0
            above = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            predictor = {
                0: 0,
                1: left,
                2: above,
                3: (left + above) // 2,
                4: paeth(left, above, upper_left),
            }[filter_type]
            row[index] = (value + predictor) & 0xFF
        rows.append(row)

    return rows[0][3], rows[0][-1], rows[-1][3], rows[-1][-1]


def _icns_png_payloads(data: bytes) -> list[bytes]:
    assert data.startswith(b"icns")
    payloads = []
    offset = 8
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        chunk = data[offset + 8 : offset + length]
        if chunk.startswith(b"\x89PNG\r\n\x1a\n"):
            payloads.append(chunk)
        offset += length
    return payloads


def test_default_agent_identity_and_help_use_evelyn_skill_surface():
    prompt_builder = _text("agent/prompt_builder.py")

    assert '"You are Evelyn, an intelligent AI assistant' in prompt_builder
    assert (
        "Load the `evelyn-agent` skill with skill_view(name='evelyn-agent')"
        in prompt_builder
    )
    assert "skill_view(name='hermes-agent')" not in prompt_builder
    assert "load the `hermes-agent` skill" not in prompt_builder
    assert "Load the `hermes-agent` skill" not in prompt_builder
    background_review = _text("agent/background_review.py")
    assert "e.g. 'evelyn-agent'" in background_review
    assert "e.g. 'hermes-agent'" not in background_review


def test_bundled_skill_ids_use_evelyn_without_old_active_aliases():
    renamed = {
        "skills/autonomous-ai-agents/evelyn-agent/SKILL.md": "evelyn-agent",
        "skills/software-development/evelyn-agent-skill-authoring/SKILL.md": (
            "evelyn-agent-skill-authoring"
        ),
    }
    old_paths = (
        "skills/autonomous-ai-agents/hermes-agent",
        "skills/software-development/hermes-agent-skill-authoring",
    )

    for relative_path, skill_name in renamed.items():
        text = _text(relative_path)
        assert re.search(rf"(?m)^name: {re.escape(skill_name)}$", text)
    for relative_path in old_paths:
        assert not (ROOT / relative_path).exists()

    generated_surface = "\n".join(
        (
            _text("website/docs/reference/skills-catalog.md"),
            _text("website/sidebars.ts"),
            _text(
                "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/"
                "reference/skills-catalog.md"
            ),
        )
    )
    assert "autonomous-ai-agents-evelyn-agent" in generated_surface
    assert "software-development-evelyn-agent-skill-authoring" in generated_surface
    assert "autonomous-ai-agents-hermes-agent" not in generated_surface
    assert "software-development-hermes-agent-skill-authoring" not in generated_surface
    old_ids = {
        "hermes-agent",
        "hermes-agent-skill-authoring",
        "hermes-context-audit",
        "hermes-profile-distribution",
        "hermes-provider-routing-ops",
        "hermes-runtime-ops",
        "hermes-workbench-analytics",
        "hermes-context-compression-engineering",
        "hermes-cron-development",
        "hermes-delegation-engineering",
    }
    reference_markers = ("related_skills", "Related skills", "Sibling skills")
    roots = (
        ROOT / "skills",
        ROOT / "optional-skills",
        ROOT / "website/docs/user-guide/skills",
    )
    stale_references = []
    for root in roots:
        for path in root.rglob("*.md"):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if any(marker in line for marker in reference_markers) and any(
                    re.search(rf"(?<![a-z0-9-]){re.escape(old)}(?![a-z0-9-])", line)
                    for old in old_ids
                ):
                    stale_references.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert stale_references == []


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

    for relative_path in (
        "apps/desktop/assets/icon.png",
        "apps/bootstrap-installer/src-tauri/icons/32x32.png",
        "apps/bootstrap-installer/src-tauri/icons/128x128.png",
        "apps/bootstrap-installer/src-tauri/icons/128x128@2x.png",
    ):
        assert _png_corner_alphas((ROOT / relative_path).read_bytes()) == (0, 0, 0, 0)

    assert (ROOT / "apps/desktop/assets/icon.icns").read_bytes().startswith(b"icns")
    assert (ROOT / "apps/desktop/assets/icon.ico").read_bytes().startswith(b"\x00\x00\x01\x00")
    assert (ROOT / "apps/bootstrap-installer/src-tauri/icons/icon.icns").read_bytes().startswith(b"icns")
    assert (ROOT / "apps/bootstrap-installer/src-tauri/icons/icon.ico").read_bytes().startswith(b"\x00\x00\x01\x00")

    for relative_path in (
        "apps/desktop/assets/icon.icns",
        "apps/bootstrap-installer/src-tauri/icons/icon.icns",
    ):
        payloads = _icns_png_payloads((ROOT / relative_path).read_bytes())
        assert payloads, relative_path
        assert all(_png_corner_alphas(payload) == (0, 0, 0, 0) for payload in payloads)

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


def test_fresh_installs_prefer_evelyn_home_with_hermes_fallback():
    install_sh = _text("scripts/install.sh")
    install_ps1 = _text("scripts/install.ps1")
    setup_sh = _text("setup-hermes.sh")
    node_bootstrap = _text("scripts/lib/node-bootstrap.sh")
    desktop_main = _text("apps/desktop/electron/main.cjs")
    desktop_bootstrap = _text("apps/desktop/electron/bootstrap-runner.cjs")
    bootstrap_runtime = _text("apps/bootstrap-installer/src-tauri/src/bootstrap.rs")
    bootstrap_paths = _text("apps/bootstrap-installer/src-tauri/src/paths.rs")

    assert "--evelyn-home" in install_sh
    assert "--hermes-home" in install_sh
    assert "$HOME/.evelyn" in install_sh
    assert "$HOME/.hermes" in install_sh

    assert "EVELYN_HOME" in install_ps1
    assert "HERMES_HOME" in install_ps1
    assert r"\evelyn" in install_ps1
    assert r"\hermes" in install_ps1
    assert "Test-Path $PreferredHome -PathType Container" in install_ps1
    assert "Test-Path $LocalLegacyHome -PathType Container" in install_ps1
    assert "Test-Path $DotLegacyHome -PathType Container" in install_ps1

    for source in (desktop_main, bootstrap_paths):
        assert "EVELYN_HOME" in source
        assert "HERMES_HOME" in source
        assert "evelyn" in source
        assert "hermes" in source

    # Desktop backend, updater handoffs, streamed update commands, and other
    # official children must all receive one pinned root under both aliases.
    assert desktop_main.count("EVELYN_HOME: HERMES_HOME") >= 4
    assert desktop_bootstrap.count("EVELYN_HOME: hermesHome") >= 2
    assert bootstrap_runtime.count("Some(hermes_home.as_str())") >= 2

    for source in (setup_sh, node_bootstrap):
        assert "EVELYN_HOME" in source
        assert "HERMES_HOME" in source
        assert "$HOME/.evelyn" in source
        assert "$HOME/.hermes" in source


def test_node_bootstrap_resolves_fresh_and_legacy_homes(tmp_path):
    script = ROOT / "scripts/lib/node-bootstrap.sh"

    def resolve(extra_env=None):
        env = {**os.environ, "HOME": str(tmp_path)}
        env.pop("EVELYN_HOME", None)
        env.pop("HERMES_HOME", None)
        env.update(extra_env or {})
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; printf "%s|%s" "$EVELYN_HOME" "$HERMES_HOME"',
                "bash",
                str(script),
            ],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        return result.stdout

    assert resolve() == f"{tmp_path / '.evelyn'}|{tmp_path / '.evelyn'}"
    (tmp_path / ".hermes").mkdir()
    assert resolve() == f"{tmp_path / '.hermes'}|{tmp_path / '.hermes'}"
    (tmp_path / ".evelyn").mkdir()
    assert resolve() == f"{tmp_path / '.evelyn'}|{tmp_path / '.evelyn'}"
    assert resolve(
        {"EVELYN_HOME": str(tmp_path / "explicit"), "HERMES_HOME": str(tmp_path / "legacy")}
    ) == f"{tmp_path / 'explicit'}|{tmp_path / 'explicit'}"


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
