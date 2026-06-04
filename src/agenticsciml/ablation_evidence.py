from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agenticsciml.evidence import SCIENTIFIC_CLAIM_NOT_SUPPORTED


ABLATION_EVIDENCE_SCHEMA_VERSION = 1
BASELINE_VARIANTS = {"root_only", "baseline"}


def has_ablation_output_source(source: Mapping[str, object]) -> bool:
    return any(
        isinstance(source.get(key), str) and str(source.get(key)).strip()
        for key in (
            "ablation_output_dir",
            "output_dir",
            "runs_csv_path",
            "runs_path",
            "summary_csv_path",
        )
    )


def build_multi_seed_ablation_verified_manifest(
    source: Mapping[str, object],
) -> dict[str, object]:
    """Build a verified evidence manifest from concrete ablation output files."""
    verifier = _verifier(source)
    paths = _source_paths(source)
    blockers: list[str] = []
    warnings: list[str] = []

    run_rows, run_blockers = _read_csv(paths["runs_csv"])
    summary_rows, summary_blockers = _read_csv(paths["summary_csv"])
    blockers.extend(f"ablation_runs.csv: {item}" for item in run_blockers)
    blockers.extend(f"ablation_summary.csv: {item}" for item in summary_blockers)

    if not verifier:
        blockers.append("verified_by or reviewer is required")
    if not paths["report_md"].is_file():
        warnings.append("ablation_report.md is missing")

    run_columns = set(run_rows[0].keys()) if run_rows else set()
    summary_columns = set(summary_rows[0].keys()) if summary_rows else set()
    if run_rows and not {"variant", "seed"}.issubset(run_columns):
        blockers.append("ablation_runs.csv must include variant and seed columns")
    if summary_rows and "variant" not in summary_columns:
        blockers.append("ablation_summary.csv must include variant column")

    seeds = _sorted_values(_collect_seeds(run_rows))
    variants = _sorted_values(_collect_variants(run_rows, summary_rows))
    baseline_variants = [variant for variant in variants if str(variant) in BASELINE_VARIANTS]
    ablation_variants = [variant for variant in variants if str(variant) not in BASELINE_VARIANTS]
    seed_coverage_by_variant = _seed_coverage_by_variant(run_rows)

    if len(seeds) < 2:
        blockers.append("at least two seeds are required in ablation_runs.csv")
    if len(ablation_variants) < 1:
        blockers.append("at least one non-baseline ablation variant is required")
    under_seeded = {
        variant: coverage
        for variant, coverage in seed_coverage_by_variant.items()
        if variant in ablation_variants and len(coverage) < 2
    }
    if under_seeded:
        blockers.append("each non-baseline ablation variant must include at least two seeds")

    expected_seeds = _optional_sequence(source, "expected_seeds")
    expected_variants = _optional_sequence(source, "expected_variants")
    if expected_seeds:
        missing = [seed for seed in expected_seeds if seed not in seeds]
        if missing:
            blockers.append(f"expected seeds missing from ablation_runs.csv: {_join_values(missing)}")
    if expected_variants:
        missing_variants = [variant for variant in expected_variants if variant not in variants]
        if missing_variants:
            blockers.append(
                f"expected variants missing from ablation outputs: {_join_values(missing_variants)}"
            )

    evidence_modes = _sorted_values(_collect_values(run_rows, summary_rows, "evidence_mode"))
    llm_modes = _sorted_values(_collect_values(run_rows, summary_rows, "llm_mode"))
    scientific_claims = _sorted_values(_collect_values(run_rows, summary_rows, "scientific_claim"))
    if any(claim != SCIENTIFIC_CLAIM_NOT_SUPPORTED for claim in scientific_claims):
        warnings.append("ablation output contains a scientific_claim value other than not_supported")

    source_artifacts = _artifact_descriptors(paths)
    artifact_blockers = [
        f"{name} is missing"
        for name, descriptor in source_artifacts.items()
        if name in {"runs_csv", "summary_csv"} and descriptor["exists"] is not True
    ]
    blockers.extend(artifact_blockers)

    verified = not blockers
    return {
        "schema_version": ABLATION_EVIDENCE_SCHEMA_VERSION,
        "source_type": "ablation_output",
        "verified": verified,
        "verified_by": verifier or None,
        "seed_count": len(seeds),
        "ablation_count": len(ablation_variants),
        "run_count": len(run_rows),
        "summary_variant_count": len(summary_rows),
        "seeds": seeds,
        "variants": variants,
        "baseline_variants": baseline_variants,
        "ablation_variants": ablation_variants,
        "seed_coverage_by_variant": seed_coverage_by_variant,
        "expected_seeds": expected_seeds,
        "expected_variants": expected_variants,
        "evidence_modes": evidence_modes,
        "llm_modes": llm_modes,
        "scientific_claims": scientific_claims,
        "source_artifacts": source_artifacts,
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": (
            "This manifest verifies local ablation output shape, seed coverage, and variant coverage. "
            "It does not prove paper-score improvement or scientific discovery."
        ),
    }


def _source_paths(source: Mapping[str, object]) -> dict[str, Path]:
    output_dir_value = source.get("ablation_output_dir") or source.get("output_dir") or ""
    output_dir = Path(str(output_dir_value)).expanduser() if str(output_dir_value).strip() else Path()
    runs_csv = _path_value(source, "runs_csv_path") or _path_value(source, "runs_path")
    summary_csv = _path_value(source, "summary_csv_path") or _path_value(source, "summary_path")
    report_md = _path_value(source, "report_md_path")
    return {
        "output_dir": output_dir,
        "runs_csv": runs_csv or output_dir / "ablation_runs.csv",
        "summary_csv": summary_csv or output_dir / "ablation_summary.csv",
        "report_md": report_md or output_dir / "ablation_report.md",
    }


def _path_value(source: Mapping[str, object], key: str) -> Path | None:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], [f"missing file: {path}"]
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return [], [f"could not read {path}: {exc}"]
    if not rows:
        return [], [f"no rows in {path}"]
    return rows, []


def _collect_seeds(rows: list[dict[str, str]]) -> list[object]:
    return [_normalize_seed(row.get("seed")) for row in rows if _normalize_seed(row.get("seed")) is not None]


def _collect_variants(
    run_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> list[str]:
    variants: list[str] = []
    for row in [*run_rows, *summary_rows]:
        value = (row.get("variant") or "").strip()
        if value:
            variants.append(value)
    return variants


def _seed_coverage_by_variant(rows: list[dict[str, str]]) -> dict[str, list[object]]:
    coverage: dict[str, list[object]] = {}
    for row in rows:
        variant = (row.get("variant") or "").strip()
        seed = _normalize_seed(row.get("seed"))
        if not variant or seed is None:
            continue
        coverage.setdefault(variant, []).append(seed)
    return {variant: _sorted_values(seeds) for variant, seeds in sorted(coverage.items())}


def _collect_values(
    run_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    key: str,
) -> list[str]:
    values: list[str] = []
    for row in [*run_rows, *summary_rows]:
        value = (row.get(key) or "").strip()
        if value:
            values.append(value)
    return values


def _optional_sequence(source: Mapping[str, object], key: str) -> list[object]:
    value = source.get(key)
    if not isinstance(value, list):
        return []
    return _sorted_values([_normalize_seed(item) if key == "expected_seeds" else str(item) for item in value])


def _normalize_seed(value: object) -> object | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _sorted_values(values: list[object]) -> list[object]:
    unique = {value for value in values if value not in (None, "")}
    return sorted(unique, key=lambda value: (str(type(value)), str(value)))


def _verifier(source: Mapping[str, object]) -> str:
    return str(
        source.get("verified_by")
        or source.get("reviewer")
        or source.get("verification_source")
        or ""
    ).strip()


def _artifact_descriptors(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    descriptors: dict[str, dict[str, object]] = {}
    for name in ("runs_csv", "summary_csv", "report_md"):
        path = paths[name]
        exists = path.is_file()
        descriptors[name] = {
            "path": str(path),
            "exists": exists,
            "sha256": _sha256(path) if exists else None,
        }
    return descriptors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_values(values: list[object]) -> str:
    return ", ".join(str(value) for value in values)
