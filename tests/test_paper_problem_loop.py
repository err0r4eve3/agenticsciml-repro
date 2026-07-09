from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.paper_problem_loop import _mapping_themes, write_paper_problem_loop_audit


def test_source_mapping_themes_ignore_substring_false_positives() -> None:
    themes = _mapping_themes("tangent behavior in biomedical science and engineering with neural operators")
    high_speed_themes = _mapping_themes("geometry-dependent hypersonic and supersonic flow surrogate")
    reliability_themes = _mapping_themes("operator reliability, stability, fidelity, and sensitivity diagnostics")
    fluid_themes = _mapping_themes("turbulent-flow super-resolution and sparse reconstruction")
    many_body_themes = _mapping_themes("many-body complex systems with message passing neural operators")
    multitask_themes = _mapping_themes("multi-task DeepONet with task coupling for PDE workflows")

    assert "agent" not in themes
    assert "engine" not in themes
    assert "biomedical" in themes
    assert "operator_learning" in themes
    assert "high_speed_flow" in high_speed_themes
    assert "reliability" in reliability_themes
    assert "safe_control" not in reliability_themes
    assert "fluid" in fluid_themes
    assert "many_body_operator" in many_body_themes
    assert "multitask_operator" in multitask_themes


def test_paper_problem_loop_audit_writes_passed_bilingual_artifact(tmp_path: Path) -> None:
    result = write_paper_problem_loop_audit(output_dir=tmp_path)

    audit_path = Path(result["paths"]["audit_json"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fno_case = next(item for item in audit["results"] if item["paper_id"] == "paper:fourier_neural_operator_parametric_pdes")

    assert audit_path == tmp_path / "agenticsciml_paper_problem_loop_audit.json"
    assert audit["passed"] is True
    assert audit["case_count"] == 10
    assert audit["source_review_ready_count"] == 10
    assert audit["prompt_quality_control_ready_count"] == 10
    assert audit["llm_context_ready_count"] == 10
    assert audit["llm_wiki_audit"]["status"] == "passed"
    assert audit["llm_wiki_audit"]["paper_problem_case_count"] == 10
    assert audit["llm_wiki_audit"]["source_candidate_node_count"] == 0
    assert audit["llm_wiki_audit"]["source_candidate_review_queue_count"] == 0
    assert audit["llm_wiki_audit"]["manual_editing"] is True
    assert audit["llm_wiki_audit"]["persistence"] == "account_scoped_json"
    assert audit["llm_wiki_audit"]["manual_edit_roundtrip"]["status"] == "passed"
    assert audit["llm_wiki_audit"]["manual_edit_roundtrip"]["title_changed"] is True
    assert audit["llm_wiki_audit"]["manual_edit_roundtrip"]["persistence"] == "account_scoped_json"
    assert (tmp_path / audit["llm_wiki_audit"]["artifacts"]["graph"]).exists()
    assert (tmp_path / audit["llm_wiki_audit"]["artifacts"]["audit"]).exists()
    assert (tmp_path / audit["llm_wiki_audit"]["artifacts"]["source_candidate_review_queue"]).exists()
    manual_edit_path = tmp_path / audit["llm_wiki_audit"]["artifacts"]["manual_edit_roundtrip"]
    assert json.loads(manual_edit_path.read_text(encoding="utf-8"))["title"].endswith("manual edit audit")
    assert all(item["source_review_status"] == "ready_for_source_audit" for item in audit["results"])
    assert fno_case["recommended_benchmark"] == "reaction_diffusion_operator_faithful_small"
    assert "fno_lite_operator" in fno_case["selected_algorithm_ids"]
    assert all(item["prompt_assembly_ready"] is True for item in audit["results"])
    assert (tmp_path / "summary.md").exists()
    assert fno_case["artifacts"] == {
        "source_review": "cases/10-fourier-neural-operator-parametric-pdes/source_review.json",
        "planner": "cases/10-fourier-neural-operator-parametric-pdes/planner.json",
        "reference_matrix": "cases/10-fourier-neural-operator-parametric-pdes/reference_matrix.json",
        "llm_context_pack": "cases/10-fourier-neural-operator-parametric-pdes/llm_context_pack.json",
    }
    assert json.loads((tmp_path / fno_case["artifacts"]["source_review"]).read_text(encoding="utf-8"))["status"] == (
        "ready_for_source_audit"
    )
    assert json.loads((tmp_path / fno_case["artifacts"]["planner"]).read_text(encoding="utf-8"))["status"] == (
        "catalog_benchmark_planned"
    )
    assert json.loads((tmp_path / fno_case["artifacts"]["reference_matrix"]).read_text(encoding="utf-8"))[
        "status"
    ] == "ready_for_offline_planning"
    fno_context = json.loads((tmp_path / fno_case["artifacts"]["llm_context_pack"]).read_text(encoding="utf-8"))
    assert fno_context["status"] == "ready_for_llm_context"
    assert fno_context["execution_prompt_contract"]["prompt_assembly_ready"] is True


def test_cli_paper_problem_loop_audit_writes_artifact(tmp_path: Path, cli_env: dict[str, str]) -> None:
    output_dir = tmp_path / "paper loop"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "paper-problem-loop-audit",
            "--output-dir",
            str(output_dir),
            "--fail-on-issues",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    audit_path = Path(result.stdout.strip())
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit_path == output_dir / "agenticsciml_paper_problem_loop_audit.json"
    assert audit["passed"] is True


def test_paper_problem_loop_audit_includes_source_collection_cache(tmp_path: Path) -> None:
    source_cache = tmp_path / "paper_source_collection.json"
    source_cache.write_text(
        json.dumps(
            {
                "status": "collected",
                "candidate_count": 1,
                "issue_count": 0,
                "query": "au:Karniadakis",
                "created_at": "2026-07-09T00:00:00Z",
                "candidates": [
                    {
                        "id": "2606.02427v1",
                        "title": "Spectral Audit of In-Context Operator Networks",
                        "url": "https://arxiv.org/abs/2606.02427",
                        "authors": ["Zhiwei Gao", "George Em Karniadakis"],
                        "published": "2026-06-01",
                        "summary": "Audits neural operator tangent spectra beyond prediction error.",
                        "real_problem": "Evaluate neural operator reliability beyond average prediction error.",
                        "real_problem_zh": "用稳定性和保真度诊断评估神经算子可靠性。",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = write_paper_problem_loop_audit(output_dir=tmp_path / "round", source_collection_path=source_cache)
    audit = json.loads(Path(result["paths"]["audit_json"]).read_text(encoding="utf-8"))

    assert audit["source_collection"] == {
        "status": "collected",
        "candidate_count": 1,
        "candidate_ids": ["source:2606.02427v1"],
        "issue_count": 0,
        "query": "au:Karniadakis",
        "created_at": "2026-07-09T00:00:00Z",
    }
    assert audit["source_candidate_count"] == 1
    assert audit["source_candidate_prompt_quality_control_ready_count"] == 1
    assert audit["source_candidate_context_ready_count"] == 1
    assert audit["source_candidate_manual_wiki_review_count"] == 1
    assert audit["llm_wiki_audit"]["source_candidate_node_count"] == 1
    assert audit["llm_wiki_audit"]["source_candidate_review_queue_count"] == 1
    source_candidate = audit["source_candidate_results"][0]
    assert source_candidate["source_mapping_status"] == "aligned"
    assert source_candidate["wiki_promotion_status"] == "manual_review_required"
    assert source_candidate["prompt_assembly_ready"] is True
    assert "bilingual_wiki_preserves_identifiers" in source_candidate["prompt_quality_control_ids"]
    assert source_candidate["artifacts"]["llm_context_pack"] == (
        "source_candidates/01-2606-02427v1/llm_context_pack.json"
    )
    assert (tmp_path / "round" / source_candidate["artifacts"]["planner"]).exists()
    source_review = json.loads((tmp_path / "round" / source_candidate["artifacts"]["source_review"]).read_text(encoding="utf-8"))
    assert source_review["source_type"] == "arxiv"
    assert source_review["mapping_audit"]["status"] == "aligned"
    assert "operator_learning" in source_review["mapping_audit"]["matched_themes"]
    graph = json.loads((tmp_path / "round" / audit["llm_wiki_audit"]["artifacts"]["graph"]).read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    wiki_node = nodes["source:2606.02427v1"]
    assert wiki_node["type"] == "source_candidate"
    assert wiki_node["title_zh"]
    assert wiki_node["description_zh"]
    assert wiki_node["real_problem_zh"] == "用稳定性和保真度诊断评估神经算子可靠性。"
    assert wiki_node["wiki_promotion_status"] == "manual_review_required"
    source_candidate_edge = next(
        edge
        for edge in graph["edges"]
        if edge["source"] == "source:2606.02427v1" and edge["target"] == "workflow:problem_intake"
    )
    assert source_candidate_edge["description_zh"]
    assert source_candidate_edge["description_zh"] != source_candidate_edge["description"]
    queue = json.loads(
        (tmp_path / "round" / audit["llm_wiki_audit"]["artifacts"]["source_candidate_review_queue"]).read_text(
            encoding="utf-8"
        )
    )
    assert queue["status"] == "manual_review_required"
    assert queue["graph_path"] == "llm_wiki/llm_wiki_okf.json"
    assert queue["editable_fields"] == [
        "title_zh",
        "description",
        "description_zh",
        "real_problem",
        "real_problem_zh",
        "tags",
        "tags_zh",
        "wiki_promotion_status",
    ]
    assert queue["nodes"][0]["id"] == "source:2606.02427v1"


def test_paper_problem_loop_audit_fails_on_unaligned_source_mapping(tmp_path: Path) -> None:
    source_cache = tmp_path / "paper_source_collection.json"
    source_cache.write_text(
        json.dumps(
            {
                "status": "collected",
                "candidate_count": 1,
                "issue_count": 0,
                "query": "au:Karniadakis",
                "created_at": "2026-07-09T00:00:00Z",
                "candidates": [
                    {
                        "id": "2602.12706v1",
                        "title": "Physics-Informed Laplace Neural Operator for Solving Partial Differential Equations",
                        "summary": "PILNO uses virtual inputs for small-data and OOD PDE operator generalization.",
                        "url": "http://arxiv.org/abs/2602.12706v1",
                        "authors": ["Heechang Kim", "George Em Karniadakis"],
                        "published": "2026-02-13",
                        "real_problem": "Audit long-horizon LLM workflows with verifiable outcomes.",
                        "real_problem_zh": "审计长周期 LLM 工作流。",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = write_paper_problem_loop_audit(output_dir=tmp_path / "round", source_collection_path=source_cache)
    audit = json.loads(Path(result["paths"]["audit_json"]).read_text(encoding="utf-8"))

    assert audit["passed"] is False
    assert audit["source_candidate_results"][0]["source_mapping_status"] == "needs_manual_review"
    assert any("source mapping audit is not aligned" in issue for issue in audit["issues"])


def test_cli_paper_problem_loop_audit_repeat_writes_round_dirs(tmp_path: Path, cli_env: dict[str, str]) -> None:
    output_dir = tmp_path / "paper loop"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "paper-problem-loop-audit",
            "--output-dir",
            str(output_dir),
            "--repeat",
            "--rounds",
            "2",
            "--interval-s",
            "0",
            "--fail-on-issues",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    audit_paths = [Path(line) for line in result.stdout.splitlines() if line.strip()]
    index = json.loads((output_dir / "paper_problem_loop_index.json").read_text(encoding="utf-8"))
    index_md = (output_dir / "paper_problem_loop_index.md").read_text(encoding="utf-8")
    health = json.loads((output_dir / "paper_problem_loop_health.json").read_text(encoding="utf-8"))

    assert len(audit_paths) == 2
    assert audit_paths[0].parent.parent == output_dir
    assert audit_paths[1].parent.parent == output_dir
    assert audit_paths[0].parent != audit_paths[1].parent
    assert all(json.loads(path.read_text(encoding="utf-8"))["passed"] is True for path in audit_paths)
    assert index["round_count"] == 2
    assert index["failed_round_count"] == 0
    assert index["latest"]["round_id"] == audit_paths[-1].parent.name
    assert index["latest"]["prompt_quality_control_ready_count"] == 10
    assert "# Paper Problem Loop Index" in index_md
    assert f"- Latest round: {audit_paths[-1].parent.name}" in index_md
    assert f"`{audit_paths[-1].parent.name}/agenticsciml_paper_problem_loop_audit.json`" in index_md
    assert health["status"] == "passed"
    assert health["latest_round_id"] == audit_paths[-1].parent.name
    assert health["issue_count"] == 0
    assert health["source_candidate_seen_count"] == 0
    assert health["source_candidate_current_seen_count"] is None
    assert health["source_candidate_pending_count"] is None
    assert health["source_candidate_pending_ids"] is None
    assert health["source_candidate_available_count"] is None
    assert health["source_candidate_coverage_ratio"] is None


def test_cli_paper_problem_loop_audit_repeat_rotates_source_candidates(
    tmp_path: Path, cli_env: dict[str, str]
) -> None:
    output_dir = tmp_path / "paper loop"
    source_cache = tmp_path / "source_collection.json"
    source_cache.write_text(
        json.dumps(
            {
                "status": "collected",
                "candidate_count": 4,
                "issue_count": 0,
                "query": "agentic sciml",
                "created_at": "2026-07-09T00:00:00Z",
                "candidates": [
                    {
                        "id": f"a{index}",
                        "title": f"Agentic SciML Candidate {index}",
                        "url": f"https://arxiv.org/abs/2606.0000{index}",
                        "authors": ["Test Author"],
                        "published": "2026-06-01",
                        "summary": "Agentic scientific machine learning benchmark candidate.",
                        "real_problem": "Audit real scientific ML planning coverage.",
                        "real_problem_zh": "审计真实科学机器学习规划覆盖。",
                    }
                    for index in range(1, 5)
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "paper-problem-loop-audit",
            "--output-dir",
            str(output_dir),
            "--source-collection-json",
            str(source_cache),
            "--source-candidate-limit",
            "2",
            "--repeat",
            "--rounds",
            "2",
            "--interval-s",
            "0",
            "--fail-on-issues",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    audit_paths = [Path(line) for line in result.stdout.splitlines() if line.strip()]
    first = json.loads(audit_paths[0].read_text(encoding="utf-8"))
    second = json.loads(audit_paths[1].read_text(encoding="utf-8"))
    index = json.loads((output_dir / "paper_problem_loop_index.json").read_text(encoding="utf-8"))
    health = json.loads((output_dir / "paper_problem_loop_health.json").read_text(encoding="utf-8"))

    assert first["source_candidate_offset"] == 0
    assert second["source_candidate_offset"] == 2
    assert first["source_candidate_selection_ids"] == ["source:a1", "source:a2"]
    assert second["source_candidate_selection_ids"] == ["source:a3", "source:a4"]
    assert index["latest"]["source_candidate_selection_ids"] == ["source:a3", "source:a4"]
    assert index["source_candidate_seen_count"] == 4
    assert index["source_candidate_seen_ids"] == ["source:a1", "source:a2", "source:a3", "source:a4"]
    assert index["source_candidate_current_seen_count"] == 4
    assert index["source_candidate_current_seen_ids"] == ["source:a1", "source:a2", "source:a3", "source:a4"]
    assert index["source_candidate_pending_count"] == 0
    assert index["source_candidate_pending_ids"] == []
    assert index["source_candidate_available_count"] == 4
    assert index["source_candidate_coverage_ratio"] == 1.0
    assert health["source_candidate_seen_count"] == 4
    assert health["source_candidate_current_seen_count"] == 4
    assert health["source_candidate_pending_count"] == 0
    assert health["source_candidate_pending_ids"] == []
    assert health["source_candidate_available_count"] == 4
    assert health["source_candidate_coverage_ratio"] == 1.0


def test_cli_paper_problem_loop_audit_source_coverage_uses_current_collection(
    tmp_path: Path, cli_env: dict[str, str]
) -> None:
    output_dir = tmp_path / "paper loop"
    source_cache = tmp_path / "source_collection.json"

    def write_source(ids: list[str]) -> None:
        source_cache.write_text(
            json.dumps(
                {
                    "status": "collected",
                    "candidate_count": len(ids),
                    "issue_count": 0,
                    "query": "agentic sciml",
                    "created_at": "2026-07-09T00:00:00Z",
                    "candidates": [
                        {
                            "id": source_id,
                            "title": f"Agentic SciML Candidate {source_id}",
                            "url": f"https://arxiv.org/abs/2606.{source_id}",
                            "authors": ["Test Author"],
                            "published": "2026-06-01",
                            "summary": "Agentic scientific machine learning benchmark candidate.",
                            "real_problem": "Audit real scientific ML planning coverage.",
                            "real_problem_zh": "审计真实科学机器学习规划覆盖。",
                        }
                        for source_id in ids
                    ],
                }
            ),
            encoding="utf-8",
        )

    def run_one_round() -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "agenticsciml.cli",
                "paper-problem-loop-audit",
                "--output-dir",
                str(output_dir),
                "--source-collection-json",
                str(source_cache),
                "--source-candidate-limit",
                "1",
                "--repeat",
                "--rounds",
                "1",
                "--interval-s",
                "0",
                "--fail-on-issues",
            ],
            check=True,
            text=True,
            capture_output=True,
            env=cli_env,
        )

    write_source(["a1", "a2", "a3", "a4"])
    run_one_round()
    write_source(["b1", "b2", "b3", "b4"])
    run_one_round()

    index = json.loads((output_dir / "paper_problem_loop_index.json").read_text(encoding="utf-8"))
    health = json.loads((output_dir / "paper_problem_loop_health.json").read_text(encoding="utf-8"))

    assert index["source_candidate_seen_count"] == 2
    assert index["source_candidate_available_count"] == 4
    assert index["source_candidate_current_seen_count"] == 1
    assert index["source_candidate_current_seen_ids"] == ["source:b2"]
    assert index["source_candidate_pending_count"] == 3
    assert index["source_candidate_pending_ids"] == ["source:b1", "source:b3", "source:b4"]
    assert index["source_candidate_coverage_ratio"] == 0.25
    assert health["source_candidate_pending_count"] == 3
    assert health["source_candidate_pending_ids"] == ["source:b1", "source:b3", "source:b4"]


def test_cli_verify_paper_problem_loop_health_checks_latest_artifacts(
    tmp_path: Path, cli_env: dict[str, str]
) -> None:
    output_dir = tmp_path / "paper loop"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "paper-problem-loop-audit",
            "--output-dir",
            str(output_dir),
            "--repeat",
            "--rounds",
            "1",
            "--interval-s",
            "0",
            "--fail-on-issues",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    ok = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-paper-problem-loop",
            str(output_dir),
            "--max-age-s",
            "86400",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )
    health = json.loads(ok.stdout)
    assert health["status"] == "passed"

    index = json.loads((output_dir / "paper_problem_loop_index.json").read_text(encoding="utf-8"))
    audit_path = output_dir / index["latest"]["audit_json"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    (audit_path.parent / audit["llm_wiki_audit"]["artifacts"]["graph"]).unlink()

    broken = subprocess.run(
        [sys.executable, "-m", "agenticsciml.cli", "verify-paper-problem-loop", str(output_dir)],
        text=True,
        capture_output=True,
        env=cli_env,
    )
    broken_health = json.loads(broken.stdout)

    assert broken.returncode == 1
    assert broken_health["status"] == "failed"
    assert "missing llm wiki artifact: graph" in broken_health["issues"]


def test_cli_paper_problem_loop_audit_repeat_continues_existing_round_index(
    tmp_path: Path, cli_env: dict[str, str]
) -> None:
    output_dir = tmp_path / "paper loop"
    (output_dir / "round-0002-old").mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "paper-problem-loop-audit",
            "--output-dir",
            str(output_dir),
            "--repeat",
            "--rounds",
            "1",
            "--interval-s",
            "0",
            "--fail-on-issues",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    audit_path = Path(result.stdout.strip())

    assert audit_path.parent.name.startswith("round-0003-")


def test_cli_paper_problem_loop_audit_index_backfills_existing_rounds(
    tmp_path: Path, cli_env: dict[str, str]
) -> None:
    output_dir = tmp_path / "paper loop"
    old_round = output_dir / "round-0002-old"
    old_round.mkdir(parents=True)
    (old_round / "agenticsciml_paper_problem_loop_audit.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-09T00:00:00Z",
                "passed": True,
                "issues": [],
                "case_count": 10,
                "source_candidate_count": 0,
                "prompt_quality_control_ready_count": 10,
                "source_candidate_prompt_quality_control_ready_count": 0,
                "llm_wiki_audit": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "paper-problem-loop-audit",
            "--output-dir",
            str(output_dir),
            "--repeat",
            "--rounds",
            "1",
            "--interval-s",
            "0",
            "--fail-on-issues",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    index = json.loads((output_dir / "paper_problem_loop_index.json").read_text(encoding="utf-8"))

    assert index["round_count"] == 2
    assert index["rounds"][0]["round_id"] == "round-0002-old"
    assert index["latest"]["round_id"].startswith("round-0003-")


def test_cli_paper_problem_loop_audit_index_backfills_corrupt_index(
    tmp_path: Path, cli_env: dict[str, str]
) -> None:
    output_dir = tmp_path / "paper loop"
    old_round = output_dir / "round-0002-old"
    old_round.mkdir(parents=True)
    (old_round / "agenticsciml_paper_problem_loop_audit.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-09T00:00:00Z",
                "passed": True,
                "issues": [],
                "case_count": 10,
                "source_candidate_count": 0,
                "prompt_quality_control_ready_count": 10,
                "source_candidate_prompt_quality_control_ready_count": 0,
                "llm_wiki_audit": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "paper_problem_loop_index.json").write_text("{", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "paper-problem-loop-audit",
            "--output-dir",
            str(output_dir),
            "--repeat",
            "--rounds",
            "1",
            "--interval-s",
            "0",
            "--fail-on-issues",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    index = json.loads((output_dir / "paper_problem_loop_index.json").read_text(encoding="utf-8"))

    assert index["round_count"] == 2
    assert index["rounds"][0]["round_id"] == "round-0002-old"
    assert index["latest"]["round_id"].startswith("round-0003-")
