"""Index progress, file hashing, citations, Chroma helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import INDEX_VERSION, OUTPUT_DIR, PROGRESS_PATH


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def format_cite(doi: str, section: str = "", heading_path: str = "") -> str:
    label = (section or "").strip()
    if not label and heading_path:
        label = heading_path.split(" > ")[-1].strip()
    if not label:
        label = "chunk"
    return f"[{doi} #{label}]"


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {"index_version": INDEX_VERSION, "files": {}, "stats": {}}
    data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    data.setdefault("files", {})
    return data


def save_progress(progress: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    progress["index_version"] = INDEX_VERSION
    PROGRESS_PATH.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def decide_file_action(
    md_path: Path,
    progress: dict,
    *,
    rebuild_all: bool,
) -> str:
    """Return: skip | index | reindex"""
    if rebuild_all:
        return "index"

    entry = progress.get("files", {}).get(md_path.name)
    if not entry:
        return "index"

    current_hash = file_hash(md_path)
    if entry.get("index_version") != INDEX_VERSION:
        return "reindex"
    if entry.get("file_hash") != current_hash:
        return "reindex"
    if entry.get("status") == "done":
        return "skip"
    return "reindex"


def mark_file_done(
    progress: dict,
    md_path: Path,
    doi: str,
    chunk_count: int,
) -> None:
    progress.setdefault("files", {})[md_path.name] = {
        "doi": doi,
        "file_hash": file_hash(md_path),
        "index_version": INDEX_VERSION,
        "status": "done",
        "chunks": chunk_count,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_progress(progress)


def mark_file_failed(progress: dict, md_path: Path, error: str) -> None:
    progress.setdefault("files", {})[md_path.name] = {
        "file_hash": file_hash(md_path),
        "index_version": INDEX_VERSION,
        "status": "failed",
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_progress(progress)


def delete_chunks_for_doi(collection, doi: str) -> int:
    data = collection.get(where={"doi": doi}, include=[])
    ids = data.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def chunk_to_metadata(chunk) -> dict[str, Any]:
    return {
        "doi": chunk.doi,
        "source_file": chunk.source_file,
        "section": chunk.section,
        "heading_path": chunk.heading_path,
        "h1": chunk.h1,
        "h2": chunk.h2,
        "h3": chunk.h3,
        "h4": chunk.h4,
        "index_version": chunk.index_version,
        "has_metrics": chunk.has_metrics,
        "content_type": getattr(chunk, "content_type", "prose"),
        "token_count": chunk.token_count,
    }
