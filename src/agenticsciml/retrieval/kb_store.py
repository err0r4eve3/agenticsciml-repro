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


class KnowledgeBase:
    def __init__(self, entries: dict[str, KnowledgeBaseEntry]):
        self.entries = entries

    @classmethod
    def load(cls, kb_dir: Path) -> "KnowledgeBase":
        index_path = kb_dir / "index.json"
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
