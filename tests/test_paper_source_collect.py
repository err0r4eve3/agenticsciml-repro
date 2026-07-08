from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.paper_source_collect import build_paper_source_collection, parse_arxiv_atom


ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2606.02427</id>
    <updated>2026-06-01T16:04:21Z</updated>
    <published>2026-06-01T16:04:21Z</published>
    <title>Spectral Audit of In-Context Operator Networks</title>
    <summary>Existing evaluations of neural operators rely primarily on prediction error.</summary>
    <author><name>Zhiwei Gao</name></author>
    <author><name>George Em Karniadakis</name></author>
    <category term="math.NA" />
  </entry>
</feed>
"""


def test_parse_arxiv_atom_builds_bilingual_real_problem_candidate() -> None:
    entries = parse_arxiv_atom(ATOM_FIXTURE)
    collection = build_paper_source_collection(
        entries=entries,
        query="au:Karniadakis",
        max_results=1,
        status="collected",
        error=None,
    )

    assert collection["status"] == "collected"
    assert collection["candidate_count"] == 1
    assert collection["issue_count"] == 0
    assert entries[0]["id"] == "2606.02427"
    assert entries[0]["published"] == "2026-06-01"
    assert entries[0]["real_problem_zh"]


def test_collect_paper_sources_cli_writes_failure_cache_without_network(tmp_path: Path, cli_env: dict[str, str]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "collect-paper-sources",
            "--output-dir",
            str(tmp_path),
            "--max-results",
            "0",
            "--fail-on-issues",
        ],
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert result.returncode == 1
    path = Path(result.stdout.strip())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "collection_failed"
    assert payload["issue_count"] == 1
