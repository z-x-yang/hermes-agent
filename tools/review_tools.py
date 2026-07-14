from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ReviewCapsuleError(ValueError):
    """The caller supplied an invalid or unsafe review capsule."""


class ReviewToolError(RuntimeError):
    """A sealed review tool call violates its invocation contract."""


_ALLOWED_TARGET_MODES = frozenset({"uncommitted", "base", "commit"})
_ALLOWED_EXTERNAL_SCOPES = frozenset({"none", "authoritative_docs_only"})
_ALLOWED_GIT_OPERATIONS = frozenset(
    {"status", "rev_parse", "merge_base", "diff", "show", "log", "blame", "grep", "ls_files"}
)
_SHELL_META_RE = re.compile(r"[\x00-\x1f\x7f;$|&`<>]")
_SAFE_REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@{}~^+\-]*$")
_MAX_FILE_LINES = 2_000
_MAX_SEARCH_RESULTS = 200
_MAX_GIT_OUTPUT = 100_000
_MAX_FINDING_LINES = 20
_MAX_SEARCH_FILE_BYTES = 2_000_000


@dataclass
class ReviewInvocationContext:
    root: Path
    capsule: dict[str, Any]
    capsule_digest: str
    scoped_paths: tuple[str, ...]
    target_mode: str
    target_base: str | None
    target_commit: str | None
    external_reference_scope: str
    local_paths: set[str] = field(default_factory=set)
    git_operations: set[str] = field(default_factory=set)
    web_urls: set[str] = field(default_factory=set)
    report: dict[str, Any] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


_CONTEXTS: dict[str, ReviewInvocationContext] = {}
_CONTEXTS_LOCK = threading.RLock()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _strict_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReviewCapsuleError(f"{label} must be an object")
    return dict(value)


def _relative_path(root: Path, raw: Any, *, require_exists: bool = False) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw.strip():
        raise ReviewToolError("path must be a non-empty relative string")
    text = raw.strip().replace("\\", "/")
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReviewToolError("path must stay inside the trusted review root")
    if _SHELL_META_RE.search(text):
        raise ReviewToolError("path contains forbidden control or shell metacharacters")
    resolved_root = root.resolve(strict=True)
    try:
        resolved = (resolved_root / candidate).resolve(strict=require_exists)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ReviewToolError("path escapes the trusted review root") from exc
    return candidate.as_posix().lstrip("./") or ".", resolved


def _validate_revision(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewCapsuleError(f"review_target.{label} must be a non-empty revision")
    revision = value.strip()
    if revision.startswith("-") or not _SAFE_REV_RE.fullmatch(revision):
        raise ReviewCapsuleError(f"review_target.{label} is not a safe revision")
    return revision


def parse_review_capsule(prompt: str, *, root: str | Path) -> ReviewInvocationContext:
    try:
        resolved_root = Path(root).expanduser().resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ReviewCapsuleError("trusted review root must be an existing directory") from exc
    if not resolved_root.is_dir():
        raise ReviewCapsuleError("trusted review root must be an existing directory")
    try:
        capsule = _strict_object(json.loads(prompt), label="review capsule")
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReviewCapsuleError("Reviewer prompt must be one strict JSON review capsule") from exc

    required = {
        "original_ask_or_approved_contract",
        "acceptance_criteria_and_invariants",
        "relevant_repo_rules",
        "review_target",
        "verification_evidence",
        "known_baseline_failures",
        "external_reference_scope",
    }
    missing = sorted(required - capsule.keys())
    unknown = sorted(capsule.keys() - required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ReviewCapsuleError("invalid review capsule fields: " + "; ".join(details))

    contract = capsule["original_ask_or_approved_contract"]
    if not isinstance(contract, str) or not contract.strip():
        raise ReviewCapsuleError("original_ask_or_approved_contract must be non-empty")
    for key, nonempty in (
        ("acceptance_criteria_and_invariants", True),
        ("relevant_repo_rules", False),
        ("verification_evidence", True),
        ("known_baseline_failures", False),
    ):
        value = capsule[key]
        if not isinstance(value, list) or (nonempty and not value):
            raise ReviewCapsuleError(f"{key} must be {'a non-empty' if nonempty else 'an'} array")
    if any(not isinstance(item, str) or not item.strip() for item in capsule["acceptance_criteria_and_invariants"]):
        raise ReviewCapsuleError("acceptance criteria must be non-empty strings")
    if any(not isinstance(item, str) or not item.strip() for item in capsule["relevant_repo_rules"]):
        raise ReviewCapsuleError("relevant repo rules must be non-empty strings")
    if any(not isinstance(item, str) or not item.strip() for item in capsule["known_baseline_failures"]):
        raise ReviewCapsuleError("known baseline failures must be non-empty strings")
    for item in capsule["verification_evidence"]:
        if not isinstance(item, dict) or set(item) != {"command", "result", "status"}:
            raise ReviewCapsuleError("verification evidence must contain command/result/status")
        if item["status"] not in {"pass", "fail", "baseline"}:
            raise ReviewCapsuleError("verification evidence status is invalid")
        if (
            not isinstance(item["command"], str)
            or not item["command"].strip()
            or not isinstance(item["result"], str)
            or not item["result"].strip()
        ):
            raise ReviewCapsuleError(
                "verification evidence command/result must be non-empty strings"
            )

    target = _strict_object(capsule["review_target"], label="review_target")
    mode = target.get("mode")
    if mode not in _ALLOWED_TARGET_MODES:
        raise ReviewCapsuleError("review_target.mode is invalid")
    allowed_target_fields = {"mode", "paths"}
    base = commit = None
    if mode == "base":
        allowed_target_fields.add("base")
        base = _validate_revision(target.get("base"), label="base")
    elif mode == "commit":
        allowed_target_fields.add("commit")
        commit = _validate_revision(target.get("commit"), label="commit")
    if set(target) != allowed_target_fields:
        raise ReviewCapsuleError("review_target fields do not match its mode")
    raw_paths = target.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ReviewCapsuleError("review_target.paths must be a non-empty array")
    paths: list[str] = []
    for raw_path in raw_paths:
        try:
            relative, _resolved = _relative_path(resolved_root, raw_path, require_exists=False)
        except ReviewToolError as exc:
            raise ReviewCapsuleError(str(exc)) from exc
        if relative == ".":
            raise ReviewCapsuleError("review_target.paths must be narrower than the repository root")
        if relative not in paths:
            paths.append(relative)

    external_scope = capsule["external_reference_scope"]
    if external_scope not in _ALLOWED_EXTERNAL_SCOPES:
        raise ReviewCapsuleError("external_reference_scope is invalid")

    canonical = json.loads(_canonical_json(capsule))
    digest = hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()
    return ReviewInvocationContext(
        root=resolved_root,
        capsule=canonical,
        capsule_digest=digest,
        scoped_paths=tuple(paths),
        target_mode=mode,
        target_base=base,
        target_commit=commit,
        external_reference_scope=external_scope,
    )


def capture_review_target_digest(context: ReviewInvocationContext) -> str:
    """Return a content digest for the frozen scoped target without writing state."""
    digest = hashlib.sha256()
    digest.update(context.capsule_digest.encode("ascii"))
    for scope in context.scoped_paths:
        relative, resolved = _relative_path(context.root, scope, require_exists=False)
        digest.update(relative.encode("utf-8"))
        if not resolved.exists() and not resolved.is_symlink():
            digest.update(b"<missing>")
            continue
        candidates: list[Path] = []
        if resolved.is_file() or resolved.is_symlink():
            candidates = [resolved]
        else:
            for directory, dirnames, filenames in os.walk(resolved, followlinks=False):
                dirnames[:] = sorted(
                    name for name in dirnames if not (Path(directory) / name).is_symlink()
                )
                candidates.extend(Path(directory) / name for name in sorted(filenames))
        for candidate in candidates:
            rel = candidate.relative_to(context.root).as_posix()
            digest.update(rel.encode("utf-8"))
            if candidate.is_symlink():
                digest.update(b"<symlink>")
                digest.update(os.readlink(candidate).encode("utf-8", errors="replace"))
                continue
            try:
                with candidate.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise ReviewToolError(f"cannot snapshot review target {rel}: {exc}") from exc
    try:
        status = subprocess.run(
            ["git", "--no-pager", "status", "--porcelain=v1", "-z", "--", *context.scoped_paths],
            cwd=context.root,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReviewToolError(f"cannot snapshot review target git status: {exc}") from exc
    if status.returncode != 0:
        raise ReviewToolError("cannot snapshot review target git status")
    digest.update(status.stdout)
    return digest.hexdigest()


def register_review_context(task_id: str, context: ReviewInvocationContext) -> None:
    if not isinstance(task_id, str) or not task_id:
        raise ReviewToolError("review task id must be non-empty")
    if not isinstance(context, ReviewInvocationContext):
        raise ReviewToolError("invalid trusted review context")
    with _CONTEXTS_LOCK:
        if task_id in _CONTEXTS:
            raise ReviewToolError("review context is already registered")
        _CONTEXTS[task_id] = context


def clear_review_context(task_id: str) -> None:
    with _CONTEXTS_LOCK:
        _CONTEXTS.pop(str(task_id or ""), None)


def get_review_context(task_id: str | None) -> ReviewInvocationContext:
    with _CONTEXTS_LOCK:
        context = _CONTEXTS.get(str(task_id or ""))
    if context is None:
        raise ReviewToolError("sealed review tool called outside a Reviewer invocation")
    return context


def _ensure_not_completed(context: ReviewInvocationContext) -> None:
    if context.report is not None:
        raise ReviewToolError("review already completed; no further tool calls are allowed")


def review_read_file(args: Mapping[str, Any], *, task_id: str | None) -> str:
    context = get_review_context(task_id)
    with context.lock:
        _ensure_not_completed(context)
        allowed = {"path", "offset", "limit"}
        if set(args) - allowed:
            raise ReviewToolError("review_read_file received unsupported fields")
        relative, resolved = _relative_path(context.root, args.get("path"), require_exists=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise ReviewToolError("review_read_file path must be a regular in-root file")
        offset = args.get("offset", 1)
        limit = args.get("limit", 500)
        if type(offset) is not int or offset < 1:
            raise ReviewToolError("offset must be a positive integer")
        if type(limit) is not int or not 1 <= limit <= _MAX_FILE_LINES:
            raise ReviewToolError(f"limit must be between 1 and {_MAX_FILE_LINES}")
        try:
            with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError as exc:
            raise ReviewToolError(f"cannot read review file: {exc}") from exc
        selected = lines[offset - 1 : offset - 1 + limit]
        content = "\n".join(f"{offset + index}|{line}" for index, line in enumerate(selected))
        context.local_paths.add(relative)
        return _canonical_json(
            {"path": relative, "content": content, "total_lines": len(lines), "truncated": offset - 1 + len(selected) < len(lines)}
        )


def review_search_files(args: Mapping[str, Any], *, task_id: str | None) -> str:
    context = get_review_context(task_id)
    with context.lock:
        _ensure_not_completed(context)
        allowed = {"pattern", "path", "file_glob", "limit"}
        if set(args) - allowed:
            raise ReviewToolError("review_search_files received unsupported fields")
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ReviewToolError("pattern must be a non-empty regex")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ReviewToolError(f"invalid search regex: {exc}") from exc
        raw_path = args.get("path", ".")
        relative_base, resolved_base = _relative_path(context.root, raw_path, require_exists=True)
        glob_pattern = args.get("file_glob")
        if glob_pattern is not None and (not isinstance(glob_pattern, str) or not glob_pattern):
            raise ReviewToolError("file_glob must be a non-empty string")
        limit = args.get("limit", 50)
        if type(limit) is not int or not 1 <= limit <= _MAX_SEARCH_RESULTS:
            raise ReviewToolError(f"limit must be between 1 and {_MAX_SEARCH_RESULTS}")

        if resolved_base.is_file():
            try:
                candidates = (
                    [resolved_base]
                    if resolved_base.stat().st_size <= _MAX_SEARCH_FILE_BYTES
                    else []
                )
            except OSError:
                candidates = []
        else:
            candidates = []
            for directory, dirnames, filenames in os.walk(resolved_base, followlinks=False):
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name != ".git" and not (Path(directory) / name).is_symlink()
                ]
                for filename in sorted(filenames):
                    candidate = Path(directory) / filename
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    if glob_pattern and not fnmatch.fnmatch(filename, glob_pattern):
                        continue
                    try:
                        if candidate.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                            continue
                    except OSError:
                        continue
                    candidates.append(candidate)
        matches: list[dict[str, Any]] = []
        for candidate in sorted(candidates):
            try:
                candidate.resolve(strict=True).relative_to(context.root)
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            rel = candidate.relative_to(context.root).as_posix()
            for line_number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append({"path": rel, "line": line_number, "text": line[:1_000]})
                    context.local_paths.add(rel)
                    if len(matches) >= limit:
                        return _canonical_json({"matches": matches, "truncated": True})
        return _canonical_json({"matches": matches, "truncated": False, "base": relative_base})


def _path_within_scope(context: ReviewInvocationContext, raw: Any) -> str:
    relative, _resolved = _relative_path(context.root, raw, require_exists=False)
    if not any(relative == scope or relative.startswith(scope.rstrip("/") + "/") for scope in context.scoped_paths):
        raise ReviewToolError("git path is outside the frozen review target scope")
    return relative


def _git_target_revision(context: ReviewInvocationContext) -> str:
    if context.target_mode == "commit":
        assert context.target_commit is not None
        return context.target_commit
    return "HEAD"


def _git_paths(context: ReviewInvocationContext, raw_path: Any | None) -> list[str]:
    if raw_path is None:
        return list(context.scoped_paths)
    return [_path_within_scope(context, raw_path)]


def _run_git(context: ReviewInvocationContext, operation: str, argv: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "--no-pager", *argv],
            cwd=context.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReviewToolError(f"review_git {operation} failed: {exc}") from exc
    stdout = (completed.stdout or "")[:_MAX_GIT_OUTPUT]
    stderr = (completed.stderr or "")[:10_000]
    allowed_returncodes = {0, 1} if operation == "grep" else {0}
    if completed.returncode not in allowed_returncodes:
        raise ReviewToolError(
            f"review_git {operation} returned {completed.returncode}: {stderr.strip()}"
        )
    context.git_operations.add(operation)
    return _canonical_json(
        {"operation": operation, "returncode": completed.returncode, "stdout": stdout, "stderr": stderr, "truncated": len(completed.stdout or "") > _MAX_GIT_OUTPUT}
    )


def review_git(args: Mapping[str, Any], *, task_id: str | None) -> str:
    context = get_review_context(task_id)
    with context.lock:
        _ensure_not_completed(context)
        operation = args.get("operation")
        if operation not in _ALLOWED_GIT_OPERATIONS:
            raise ReviewToolError("review_git operation is not allowed")
        allowed_fields = {
            "status": {"operation", "path"},
            "rev_parse": {"operation"},
            "merge_base": {"operation"},
            "diff": {"operation", "path", "context_lines"},
            "show": {"operation", "path", "context_lines"},
            "log": {"operation", "path", "max_count"},
            "blame": {"operation", "path", "start_line", "end_line"},
            "grep": {"operation", "path", "pattern"},
            "ls_files": {"operation", "path"},
        }[operation]
        if set(args) - allowed_fields:
            raise ReviewToolError("review_git received unsupported fields for this operation")
        paths = _git_paths(context, args.get("path"))
        target_revision = _git_target_revision(context)

        if operation == "status":
            argv = ["status", "--short", "--branch", "--untracked-files=all", "--", *paths]
        elif operation == "rev_parse":
            argv = ["rev-parse", "--verify", f"{target_revision}^{{commit}}"]
        elif operation == "merge_base":
            left = context.target_base or target_revision
            argv = ["merge-base", left, target_revision]
        elif operation == "diff":
            context_lines = args.get("context_lines", 3)
            if type(context_lines) is not int or not 0 <= context_lines <= 20:
                raise ReviewToolError("context_lines must be between 0 and 20")
            argv = ["diff", "--no-ext-diff", "--no-textconv", f"--unified={context_lines}"]
            if context.target_mode == "base":
                assert context.target_base is not None
                argv.extend([context.target_base, "HEAD"])
            elif context.target_mode == "commit":
                assert context.target_commit is not None
                argv.extend([f"{context.target_commit}^", context.target_commit])
            else:
                argv.append("HEAD")
            argv.extend(["--", *paths])
        elif operation == "show":
            context_lines = args.get("context_lines", 3)
            if type(context_lines) is not int or not 0 <= context_lines <= 20:
                raise ReviewToolError("context_lines must be between 0 and 20")
            argv = ["show", "--no-ext-diff", "--no-textconv", f"--unified={context_lines}", target_revision, "--", *paths]
        elif operation == "log":
            max_count = args.get("max_count", 20)
            if type(max_count) is not int or not 1 <= max_count <= 100:
                raise ReviewToolError("max_count must be between 1 and 100")
            argv = ["log", "--no-decorate", "--oneline", f"--max-count={max_count}", "--", *paths]
        elif operation == "blame":
            if len(paths) != 1 or args.get("path") is None:
                raise ReviewToolError("blame requires one explicit scoped path")
            start = args.get("start_line")
            end = args.get("end_line")
            if type(start) is not int or type(end) is not int or start < 1 or end < start:
                raise ReviewToolError("blame requires a valid positive line range")
            if end - start + 1 > 200:
                raise ReviewToolError("blame line range is too wide")
            argv = ["blame", "--line-porcelain", "-L", f"{start},{end}", "--", paths[0]]
        elif operation == "grep":
            pattern = args.get("pattern")
            if not isinstance(pattern, str) or not pattern or _SHELL_META_RE.search(pattern):
                raise ReviewToolError("grep pattern is empty or contains forbidden shell input")
            argv = ["grep", "-n", "--no-color", "-e", pattern, "--", *paths]
        else:
            argv = ["ls-files", "--", *paths]
        return _run_git(context, operation, argv)


def record_reviewer_web_result(task_id: str | None, *, urls: list[str]) -> None:
    context = get_review_context(task_id)
    with context.lock:
        _ensure_not_completed(context)
        if context.external_reference_scope != "authoritative_docs_only":
            raise ReviewToolError("web access is disabled by the frozen review capsule")
        for url in urls:
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                context.web_urls.add(url)


def assert_reviewer_web_allowed(task_id: str | None) -> ReviewInvocationContext | None:
    with _CONTEXTS_LOCK:
        context = _CONTEXTS.get(str(task_id or ""))
    if context is None:
        return None
    with context.lock:
        _ensure_not_completed(context)
        if context.external_reference_scope != "authoritative_docs_only":
            raise ReviewToolError("web access is disabled by the frozen review capsule")
    return context


def _validate_report(context: ReviewInvocationContext, raw: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {"findings", "overall_correctness", "explanation", "confidence", "sources_used"}
    if set(raw) != expected_fields:
        raise ReviewToolError("review report fields do not match the sealed schema")
    findings = raw.get("findings")
    if not isinstance(findings, list):
        raise ReviewToolError("findings must be an array")
    overall = raw.get("overall_correctness")
    if overall not in {"correct", "incorrect", "unverified"}:
        raise ReviewToolError("overall_correctness is invalid")
    if findings and overall != "incorrect":
        raise ReviewToolError(
            "overall_correctness must be incorrect when findings are reported"
        )
    if not findings and overall == "incorrect":
        raise ReviewToolError(
            "overall_correctness cannot be incorrect without a finding"
        )
    explanation = raw.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ReviewToolError("explanation must be non-empty")
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ReviewToolError("report confidence must be between 0 and 1")

    sources = raw.get("sources_used")
    if not isinstance(sources, dict) or set(sources) != {"local_paths", "git_operations", "web_urls"}:
        raise ReviewToolError("sources_used fields do not match the sealed schema")
    for key in ("local_paths", "git_operations", "web_urls"):
        if not isinstance(sources[key], list) or any(not isinstance(item, str) for item in sources[key]):
            raise ReviewToolError(f"sources_used.{key} must be a string array")
    claimed_local = set(sources["local_paths"])
    claimed_git = set(sources["git_operations"])
    claimed_web = set(sources["web_urls"])
    code_inspection_operations = {"diff", "show", "blame", "grep"}
    if not claimed_local and not (claimed_git & code_inspection_operations):
        raise ReviewToolError(
            "review must inspect and cite target code through a local file or "
            "diff/show/blame/grep operation"
        )
    if not claimed_local.issubset(context.local_paths):
        raise ReviewToolError("sources_used.local_paths contains an untraced path")
    if not claimed_git.issubset(context.git_operations):
        raise ReviewToolError("sources_used.git_operations contains an untraced operation")
    if not claimed_web.issubset(context.web_urls):
        raise ReviewToolError("sources_used.web_urls contains an untraced URL")

    canonical_findings: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {"title", "body", "priority", "confidence", "code_location", "evidence"}:
            raise ReviewToolError("finding fields do not match the sealed schema")
        priority = finding["priority"]
        if type(priority) is not int or priority not in {0, 1, 2}:
            raise ReviewToolError("finding priority must be 0, 1, or 2")
        title = finding["title"]
        if not isinstance(title, str) or not title.startswith(f"[P{priority}] ") or len(title.split()) < 3:
            raise ReviewToolError("finding title must begin with the matching [P0-P2] prefix")
        body = finding["body"]
        if not isinstance(body, str) or not body.strip():
            raise ReviewToolError("finding body must name a concrete failure scenario")
        finding_confidence = finding["confidence"]
        if isinstance(finding_confidence, bool) or not isinstance(finding_confidence, (int, float)) or not 0 <= finding_confidence <= 1:
            raise ReviewToolError("finding confidence must be between 0 and 1")
        location = finding["code_location"]
        if not isinstance(location, dict) or set(location) != {"path", "start_line", "end_line"}:
            raise ReviewToolError("code_location fields do not match the sealed schema")
        path = _path_within_scope(context, location["path"])
        start = location["start_line"]
        end = location["end_line"]
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ReviewToolError("code_location line range is invalid")
        if end - start + 1 > _MAX_FINDING_LINES:
            raise ReviewToolError("code_location line range is too wide")
        _relative, resolved_location = _relative_path(
            context.root, path, require_exists=False
        )
        if resolved_location.exists() and resolved_location.is_file():
            try:
                with resolved_location.open(
                    "r", encoding="utf-8", errors="replace"
                ) as handle:
                    total_lines = sum(1 for _line in handle)
            except OSError as exc:
                raise ReviewToolError(
                    f"cannot validate code_location path: {exc}"
                ) from exc
            if end > total_lines:
                raise ReviewToolError("code_location exceeds the current file")
        evidence = finding["evidence"]
        if not isinstance(evidence, list) or not evidence or any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ReviewToolError("finding evidence must be a non-empty string array")
        if path not in context.local_paths and not context.git_operations:
            raise ReviewToolError("finding location has no traced local or git evidence")
        canonical_findings.append(
            {
                "title": title.strip(),
                "body": body.strip(),
                "priority": priority,
                "confidence": float(finding_confidence),
                "code_location": {"path": path, "start_line": start, "end_line": end},
                "evidence": [item.strip() for item in evidence],
            }
        )
    return {
        "findings": canonical_findings,
        "overall_correctness": overall,
        "explanation": explanation.strip(),
        "confidence": float(confidence),
        "sources_used": {
            "local_paths": sorted(claimed_local),
            "git_operations": sorted(claimed_git),
            "web_urls": sorted(claimed_web),
        },
    }


def report_review_findings(args: Mapping[str, Any], *, task_id: str | None) -> str:
    context = get_review_context(task_id)
    with context.lock:
        if context.report is not None:
            raise ReviewToolError("report_review_findings may be called only once")
        report = _validate_report(context, args)
        context.report = report
        return _canonical_json({"accepted": True, "report": report})


# Registration is intentionally last: raw source tools must already own exact policy
# identities before the sealed derived aliases are admitted.
from tools import file_tools as _file_tools  # noqa: E402,F401
from tools import terminal_tool as _terminal_tool  # noqa: E402,F401
from tools.registry import registry  # noqa: E402
from tools.tool_effects import ResultRetention, ToolEffect, builtin_policy_descriptor  # noqa: E402


def _raw_identity(name: str) -> str:
    identity = registry.resolved_policy_identity(name)
    if not identity:
        raise RuntimeError(f"raw tool policy identity missing for sealed reviewer source {name}")
    return identity


REVIEW_READ_FILE_SCHEMA = {
    "name": "review_read_file",
    "description": "Read a bounded text-file slice inside the trusted review repository root.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_FILE_LINES},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}
REVIEW_SEARCH_FILES_SCHEMA = {
    "name": "review_search_files",
    "description": "Search text files inside the trusted review repository root without spill or telemetry.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "file_glob": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_SEARCH_RESULTS},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
}
REVIEW_GIT_SCHEMA = {
    "name": "review_git",
    "description": "Inspect the frozen review target through a structured read-only git broker; no shell string is accepted.",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": sorted(_ALLOWED_GIT_OPERATIONS)},
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "max_count": {"type": "integer", "minimum": 1, "maximum": 100},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}
REPORT_REVIEW_FINDINGS_SCHEMA = {
    "name": "report_review_findings",
    "description": "Submit the one validated structured result that completes a Reviewer invocation.",
    "parameters": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "priority": {"type": "integer", "enum": [0, 1, 2]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "code_location": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "start_line": {"type": "integer", "minimum": 1},
                                "end_line": {"type": "integer", "minimum": 1},
                            },
                            "required": ["path", "start_line", "end_line"],
                            "additionalProperties": False,
                        },
                        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    },
                    "required": ["title", "body", "priority", "confidence", "code_location", "evidence"],
                    "additionalProperties": False,
                },
            },
            "overall_correctness": {"type": "string", "enum": ["correct", "incorrect", "unverified"]},
            "explanation": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "sources_used": {
                "type": "object",
                "properties": {
                    "local_paths": {"type": "array", "items": {"type": "string"}},
                    "git_operations": {"type": "array", "items": {"type": "string"}},
                    "web_urls": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["local_paths", "git_operations", "web_urls"],
                "additionalProperties": False,
            },
        },
        "required": ["findings", "overall_correctness", "explanation", "confidence", "sources_used"],
        "additionalProperties": False,
    },
}


def _register(name: str, schema: dict[str, Any], handler, *, source_name: str) -> None:
    registry.register(
        name=name,
        toolset="review",
        schema=schema,
        handler=lambda args, _handler=handler, **kw: _handler(args, task_id=kw.get("task_id")),
        check_fn=lambda: True,
        emoji="🔎",
        max_result_size_chars=_MAX_GIT_OUTPUT,
        descriptor=builtin_policy_descriptor(
            name=name,
            schema=schema,
            handler=handler,
            effects={ToolEffect.READ_LOCAL},
            retention=ResultRetention.NO_SPILL,
            required_parent_any_of={_raw_identity(source_name)},
        ),
    )


_register("review_read_file", REVIEW_READ_FILE_SCHEMA, review_read_file, source_name="read_file")
_register("review_search_files", REVIEW_SEARCH_FILES_SCHEMA, review_search_files, source_name="search_files")
_register("review_git", REVIEW_GIT_SCHEMA, review_git, source_name="terminal")
_register("report_review_findings", REPORT_REVIEW_FINDINGS_SCHEMA, report_review_findings, source_name="read_file")
