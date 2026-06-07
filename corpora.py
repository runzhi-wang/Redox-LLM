"""Multi-corpus RAG configuration: OER, EO, and mixed retrieval."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config import INDEX_VERSION, OUTPUT_DIR, ROOT

RagMode = Literal["oer", "eo", "mixed"]

RAG_MODE_LABELS: dict[str, str] = {
    "oer": "OER",
    "eo": "EO",
    "mixed": "OER+EO 混合",
}


@dataclass(frozen=True)
class Corpus:
    key: str
    label: str
    md_dir: Path
    collection_name: str
    progress_path: Path


def _md_dir(env_key: str, default: Path) -> Path:
    override = os.getenv(env_key)
    return Path(override).expanduser() if override else default


OER_CORPUS = Corpus(
    key="oer",
    label="OER",
    md_dir=_md_dir("OER_RAG_OER_MD_DIR", ROOT / "OER" / "OER_md"),
    collection_name=os.getenv("OER_RAG_OER_COLLECTION", "oer_md"),
    progress_path=OUTPUT_DIR / "index_progress_oer.json",
)

EO_CORPUS = Corpus(
    key="eo",
    label="EO",
    md_dir=_md_dir("OER_RAG_EO_MD_DIR", ROOT / "EO" / "EO_md"),
    collection_name=os.getenv("OER_RAG_EO_COLLECTION", "eo_md"),
    progress_path=OUTPUT_DIR / "index_progress_eo.json",
)

CORPORA: dict[str, Corpus] = {
    "oer": OER_CORPUS,
    "eo": EO_CORPUS,
}

LEGACY_PROGRESS = OUTPUT_DIR / "index_progress.json"


def corpus_progress_path(corpus: Corpus) -> Path:
    if corpus.key == "oer" and LEGACY_PROGRESS.exists() and not corpus.progress_path.exists():
        return LEGACY_PROGRESS
    return corpus.progress_path


def count_indexed_papers(corpus: Corpus) -> int:
    path = corpus_progress_path(corpus)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            files = data.get("files") or {}
            done = sum(1 for item in files.values() if item.get("status") == "done")
            if done:
                return done
        except (json.JSONDecodeError, OSError):
            pass
    if corpus.md_dir.is_dir():
        return len(list(corpus.md_dir.glob("*.md")))
    return 0


def corpora_for_mode(mode: str) -> list[Corpus]:
    if mode == "mixed":
        return [OER_CORPUS, EO_CORPUS]
    return [CORPORA[mode]]


def mode_ready(mode: str, stats: dict[str, dict]) -> bool:
    keys = ["oer", "eo"] if mode == "mixed" else [mode]
    ready = [stats.get(k, {}).get("ok") for k in keys]
    return any(ready) if mode == "mixed" else bool(ready[0])
