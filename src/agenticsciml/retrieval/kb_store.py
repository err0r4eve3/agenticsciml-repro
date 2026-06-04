from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class KnowledgeBaseEntry:
    entry_id: str
    title: str
    description: str
    content: str
    path: Path
    source_title: str = ""
    source_url_or_doi: str = ""
    source_type: str = "local_note"
    task_tags: tuple[str, ...] = ()
    implementation_snippet_available: bool = False

    def provenance(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "title": self.title,
            "source_title": self.source_title,
            "source_url_or_doi": self.source_url_or_doi,
            "source_type": self.source_type,
            "task_tags": list(self.task_tags),
            "implementation_snippet_available": self.implementation_snippet_available,
            "path": str(self.path),
        }


class KnowledgeBase:
    def __init__(self, entries: dict[str, KnowledgeBaseEntry], *, coverage_status: str | None = None):
        self.entries = entries
        self.coverage_status = coverage_status

    @classmethod
    def load(cls, kb_dir: Path) -> "KnowledgeBase":
        index_path = kb_dir / "index.json"
        if not index_path.exists():
            return cls({}, coverage_status="missing")
        records = json.loads(index_path.read_text(encoding="utf-8"))
        entries: dict[str, KnowledgeBaseEntry] = {}
        for record in records:
            path = kb_dir / record["file"]
            entries[record["id"]] = KnowledgeBaseEntry(
                entry_id=record["id"],
                title=record["title"],
                description=record["description"],
                content=path.read_text(encoding="utf-8"),
                path=path,
                source_title=str(record.get("source_title", "")),
                source_url_or_doi=str(record.get("source_url_or_doi", "")),
                source_type=str(record.get("source_type", "local_note")),
                task_tags=tuple(str(item) for item in record.get("task_tags", []) if isinstance(item, str)),
                implementation_snippet_available=bool(record.get("implementation_snippet_available", False)),
            )
        return cls(entries)

    def get(self, entry_id: str) -> KnowledgeBaseEntry:
        return self.entries[entry_id]

    def all(self) -> list[KnowledgeBaseEntry]:
        return list(self.entries.values())

    def random_entry(self, seed: int, salt: str = "") -> KnowledgeBaseEntry | None:
        entries = sorted(self.entries.values(), key=lambda entry: entry.entry_id)
        if not entries:
            return None
        rng = random.Random(f"{seed}:{salt}")
        return entries[rng.randrange(len(entries))]

    def manifest(self) -> dict[str, object]:
        entries = sorted(self.entries.values(), key=lambda entry: entry.entry_id)
        entry_count = len(entries)
        if entry_count == 0:
            return {
                "schema_version": 1,
                "entry_count": 0,
                "paper_reference_entry_count": 70,
                "coverage_status": self.coverage_status or "empty",
                "paper_kb_equivalent": False,
                "provenance_complete": False,
                "provenance_complete_count": 0,
                "missing_provenance_entry_ids": [],
                "entries": [],
            }
        missing_provenance = [
            entry.entry_id
            for entry in entries
            if not (
                entry.source_title.strip()
                and entry.source_url_or_doi.strip()
                and entry.source_type.strip()
                and entry.task_tags
            )
        ]
        paper_kb_equivalent = entry_count >= 70 and not missing_provenance
        return {
            "schema_version": 1,
            "entry_count": entry_count,
            "paper_reference_entry_count": 70,
            "coverage_status": "paper_kb_equivalent" if paper_kb_equivalent else "local_kb_seed",
            "paper_kb_equivalent": paper_kb_equivalent,
            "provenance_complete": not missing_provenance,
            "provenance_complete_count": entry_count - len(missing_provenance),
            "missing_provenance_entry_ids": missing_provenance,
            "entries": [entry.provenance() for entry in entries],
        }


def kb_manifest_for_dir(kb_dir: Path) -> dict[str, object]:
    return KnowledgeBase.load(kb_dir).manifest()
