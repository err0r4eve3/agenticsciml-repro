from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INSPECTOR_VERSION = "strategy_fidelity.v1"

_TEXT_CRITERIA_ALIASES = {
    "required_terms": {
        "require",
        "requires",
        "required",
        "required_terms",
        "require_terms",
        "must_include",
        "include_terms",
    },
    "forbidden_terms": {
        "forbid",
        "forbids",
        "forbidden",
        "forbidden_terms",
        "forbid_terms",
        "must_avoid",
        "avoid_terms",
        "exclude_terms",
    },
    "required_imports": {"required_imports", "require_imports", "must_import"},
    "forbidden_imports": {"forbidden_imports", "forbid_imports", "must_not_import"},
    "required_call_names": {"required_calls", "require_calls", "required_call_names", "must_call"},
    "forbidden_call_names": {"forbidden_calls", "forbid_calls", "forbidden_call_names", "must_not_call"},
}


@dataclass(slots=True)
class _CodeFacts:
    source_digest: str | None
    term_text: str
    compact_term_text: str
    imports: set[str]
    calls: set[str]
    syntax_error: str | None


def strategy_locks_from_readiness(readiness_report: dict[str, Any]) -> list[dict[str, Any]]:
    locks = readiness_report.get("manual_strategy_locks") if isinstance(readiness_report, dict) else []
    if not isinstance(locks, list):
        return []
    return [dict(lock) for lock in locks if isinstance(lock, dict)]


def inspect_solution_strategy(
    solution_path: Path,
    strategy_locks: list[dict[str, Any]],
) -> dict[str, object]:
    locks = [_normalize_lock(lock, index) for index, lock in enumerate(strategy_locks)]
    facts = _code_facts(solution_path)
    checks: list[dict[str, object]] = []

    if not locks:
        checks.append(
            _check(
                "strategy-fidelity.no-locks",
                None,
                "strategy_lock",
                "info",
                True,
                "No manual strategy locks were provided for implementation fidelity inspection.",
            )
        )
    for lock in locks:
        checks.extend(_inspect_lock(lock, facts))

    failed_blockers = [check for check in checks if check["severity"] == "blocker" and not check["passed"]]
    failed_warnings = [check for check in checks if check["severity"] == "warning" and not check["passed"]]
    status = "blocked" if failed_blockers else "passed_with_warnings" if failed_warnings else "passed"
    return {
        "schema_version": 1,
        "inspector_version": INSPECTOR_VERSION,
        "solution_file": solution_path.name,
        "solution_sha256": facts.source_digest,
        "status": status,
        "execution_allowed": not failed_blockers,
        "summary": {
            "lock_count": len(locks),
            "auditable_lock_count": sum(1 for lock in locks if _has_criteria(_criteria_for_lock(lock))),
            "check_count": len(checks),
            "failed_blocker_count": len(failed_blockers),
            "failed_warning_count": len(failed_warnings),
        },
        "checks": checks,
        "claim_boundary": (
            "Strategy fidelity inspection is a deterministic pre-execution guardrail. "
            "It checks only explicit machine-auditable lock criteria and is not scientific validation."
        ),
    }


def _inspect_lock(lock: dict[str, object], facts: _CodeFacts) -> list[dict[str, object]]:
    criteria = _criteria_for_lock(lock)
    lock_id = str(lock["lock_id"])
    if not _has_criteria(criteria):
        return [
            _check(
                f"strategy-fidelity.{_slug(lock_id)}.not-auditable",
                lock_id,
                "strategy_lock",
                "info",
                True,
                "Manual strategy lock has no explicit machine-auditable criteria; it remains recorded as planning context.",
            )
        ]

    checks: list[dict[str, object]] = []
    if facts.syntax_error and (
        criteria["required_imports"]
        or criteria["forbidden_imports"]
        or criteria["required_call_names"]
        or criteria["forbidden_call_names"]
    ):
        checks.append(
            _check(
                f"strategy-fidelity.{_slug(lock_id)}.syntax",
                lock_id,
                "code_parse",
                _severity(lock),
                False,
                f"Could not inspect imports or calls because solution.py has a syntax error: {facts.syntax_error}",
            )
        )
        return checks

    for term in criteria["required_terms"]:
        checks.append(
            _check(
                f"strategy-fidelity.{_slug(lock_id)}.required-term.{_slug(term)}",
                lock_id,
                "required_terms",
                _severity(lock),
                _contains_term(facts, term),
                f"Required strategy term is present: {term}",
                failure_message=f"Required strategy term not found in solution.py: {term}",
                criteria={"term": term},
            )
        )
    for term in criteria["forbidden_terms"]:
        present = _contains_term(facts, term)
        checks.append(
            _check(
                f"strategy-fidelity.{_slug(lock_id)}.forbidden-term.{_slug(term)}",
                lock_id,
                "forbidden_terms",
                _severity(lock),
                not present,
                f"Forbidden strategy term is absent: {term}",
                failure_message=f"Forbidden strategy term found in solution.py: {term}",
                criteria={"term": term},
            )
        )
    for import_name in criteria["required_imports"]:
        matches = _matching_symbols(facts.imports, import_name)
        checks.append(
            _check(
                f"strategy-fidelity.{_slug(lock_id)}.required-import.{_slug(import_name)}",
                lock_id,
                "required_imports",
                _severity(lock),
                bool(matches),
                f"Required import is present: {import_name}",
                failure_message=f"Required import not found in solution.py: {import_name}",
                criteria={"import": import_name},
                evidence={"matches": matches},
            )
        )
    for import_name in criteria["forbidden_imports"]:
        matches = _matching_symbols(facts.imports, import_name)
        checks.append(
            _check(
                f"strategy-fidelity.{_slug(lock_id)}.forbidden-import.{_slug(import_name)}",
                lock_id,
                "forbidden_imports",
                _severity(lock),
                not matches,
                f"Forbidden import is absent: {import_name}",
                failure_message=f"Forbidden import found in solution.py: {import_name}",
                criteria={"import": import_name},
                evidence={"matches": matches},
            )
        )
    for call_name in criteria["required_call_names"]:
        matches = _matching_symbols(facts.calls, call_name)
        checks.append(
            _check(
                f"strategy-fidelity.{_slug(lock_id)}.required-call.{_slug(call_name)}",
                lock_id,
                "required_call_names",
                _severity(lock),
                bool(matches),
                f"Required call is present: {call_name}",
                failure_message=f"Required call not found in solution.py: {call_name}",
                criteria={"call": call_name},
                evidence={"matches": matches},
            )
        )
    for call_name in criteria["forbidden_call_names"]:
        matches = _matching_symbols(facts.calls, call_name)
        checks.append(
            _check(
                f"strategy-fidelity.{_slug(lock_id)}.forbidden-call.{_slug(call_name)}",
                lock_id,
                "forbidden_call_names",
                _severity(lock),
                not matches,
                f"Forbidden call is absent: {call_name}",
                failure_message=f"Forbidden call found in solution.py: {call_name}",
                criteria={"call": call_name},
                evidence={"matches": matches},
            )
        )
    return checks


def _has_criteria(criteria: dict[str, list[str]]) -> bool:
    return any(criteria.values())


def _normalize_lock(lock: dict[str, Any], index: int) -> dict[str, object]:
    lock_id = _first_text(lock.get("lock_id"), lock.get("id")) or f"manual_lock_{index + 1:03d}"
    return {
        "lock_id": lock_id,
        "kind": _first_text(lock.get("kind")) or "constraint",
        "text": _first_text(lock.get("text"), lock.get("description")) or "",
        "required": bool(lock.get("required", True)),
        "inspection": lock.get("inspection") if isinstance(lock.get("inspection"), dict) else {},
        **{key: lock.get(key) for key in _criteria_keys() if key in lock},
    }


def _criteria_for_lock(lock: dict[str, object]) -> dict[str, list[str]]:
    payloads = []
    inspection = lock.get("inspection")
    if isinstance(inspection, dict):
        payloads.append(inspection)
    payloads.append(lock)

    criteria = {key: [] for key in _criteria_keys()}
    for payload in payloads:
        for key in _criteria_keys():
            criteria[key].extend(_string_list(payload.get(key)))

    text = lock.get("text")
    if isinstance(text, str) and text.strip():
        for key, values in _criteria_from_text(text).items():
            criteria[key].extend(values)

    return {key: _dedupe(values) for key, values in criteria.items()}


def _criteria_from_text(text: str) -> dict[str, list[str]]:
    criteria = {key: [] for key in _criteria_keys()}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z_ -]+)\s*[:=]\s*(.+?)\s*$", line)
        if not match:
            continue
        raw_key = match.group(1).strip().lower().replace("-", "_").replace(" ", "_")
        raw_values = match.group(2)
        for criteria_key, aliases in _TEXT_CRITERIA_ALIASES.items():
            if raw_key in aliases:
                criteria[criteria_key].extend(_split_terms(raw_values))
    return criteria


def _code_facts(solution_path: Path) -> _CodeFacts:
    if not solution_path.exists():
        return _CodeFacts(None, "", "", set(), set(), f"missing generated solution file: {solution_path.name}")

    source = solution_path.read_text(encoding="utf-8")
    term_text = _term_search_text(source)
    imports: set[str] = set()
    calls: set[str] = set()
    syntax_error = None
    try:
        tree = ast.parse(source, filename=str(solution_path))
    except SyntaxError as exc:
        syntax_error = str(exc)
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
                    imports.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    imports.add(node.module.split(".", 1)[0])
                for alias in node.names:
                    if node.module:
                        imports.add(f"{node.module}.{alias.name}")
                    imports.add(alias.name)
            elif isinstance(node, ast.Call):
                call_name = _call_name(node.func)
                if call_name:
                    calls.add(call_name)
    return _CodeFacts(
        source_digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        term_text=term_text.lower(),
        compact_term_text=_compact_text(term_text),
        imports=imports,
        calls=calls,
        syntax_error=syntax_error,
    )


def _term_search_text(source: str) -> str:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return " ".join(token.string for token in tokens if token.type != tokenize.COMMENT)
    except tokenize.TokenError:
        return source


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _check(
    check_id: str,
    lock_id: str | None,
    category: str,
    severity: str,
    passed: bool,
    success_message: str,
    *,
    failure_message: str | None = None,
    criteria: dict[str, object] | None = None,
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "lock_id": lock_id,
        "category": category,
        "severity": severity,
        "passed": passed,
        "message": success_message if passed else failure_message or success_message,
        "criteria": criteria or {},
        "evidence": evidence or {},
    }


def _criteria_keys() -> tuple[str, ...]:
    return (
        "required_terms",
        "forbidden_terms",
        "required_imports",
        "forbidden_imports",
        "required_call_names",
        "forbidden_call_names",
    )


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _split_terms(value)
    if not isinstance(value, list):
        return []
    terms = []
    for item in value:
        if isinstance(item, str) and item.strip():
            terms.append(_clean_term(item))
    return [term for term in terms if term]


def _split_terms(value: str) -> list[str]:
    terms = [_clean_term(term) for term in re.split(r"[,;\n]+", value)]
    return [term for term in terms if term]


def _clean_term(value: str) -> str:
    return value.strip().strip("`'\"")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(value)
    return normalized


def _contains_term(facts: _CodeFacts, term: str) -> bool:
    lowered = term.lower()
    if lowered in facts.term_text:
        return True
    compact = _compact_text(term)
    return bool(compact and compact in facts.compact_term_text)


def _matching_symbols(symbols: set[str], wanted: str) -> list[str]:
    normalized_wanted = _normalize_symbol(wanted)
    matches = [
        symbol
        for symbol in sorted(symbols)
        if _normalize_symbol(symbol) == normalized_wanted
        or _normalize_symbol(symbol).endswith("." + normalized_wanted)
    ]
    return matches


def _normalize_symbol(value: str) -> str:
    return value.strip().replace(" ", "").replace("-", "_").lower()


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _severity(lock: dict[str, object]) -> str:
    return "blocker" if lock.get("required", True) else "warning"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "item"


def _first_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
