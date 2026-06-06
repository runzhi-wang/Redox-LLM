"""Chroma helpers: HNSW safety settings, health checks, and sqlite fallback."""

from __future__ import annotations

import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.errors import InternalError, NotFoundError

from config import CHROMA_DIR, COLLECTION_NAME

# Chroma 1.5.x compactor can write a broken HNSW pickle when sync_threshold is low.
# Keep WAL-only until the collection is far larger than our corpus (~10k chunks).
HNSW_CREATE_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:sync_threshold": 100_000,
    "hnsw:num_threads": 1,
}

HNSW_SAFE_METADATA = {
    "hnsw:sync_threshold": 100_000,
    "hnsw:num_threads": 1,
}

_CHUNK_INDEX_RE = re.compile(r"#(\d+)$")


def create_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _patch_collection_metadata(collection) -> None:
    meta = collection.metadata or {}
    needs_patch = any(
        meta.get(key) != value for key, value in HNSW_SAFE_METADATA.items()
    )
    if not needs_patch:
        return
    try:
        collection.modify(metadata=HNSW_SAFE_METADATA)
    except ValueError:
        # Existing collection may reject metadata that touches distance settings.
        pass


def open_collection(
    chroma: chromadb.PersistentClient,
    *,
    create_if_missing: bool = True,
):
    try:
        collection = chroma.get_collection(COLLECTION_NAME)
    except NotFoundError:
        if not create_if_missing:
            raise
        collection = chroma.create_collection(
            name=COLLECTION_NAME,
            metadata=HNSW_CREATE_METADATA,
        )
        return collection

    _patch_collection_metadata(collection)
    return chroma.get_collection(COLLECTION_NAME)


def _embedding_nonempty(vec: Any) -> bool:
    if vec is None:
        return False
    try:
        return len(vec) > 0
    except TypeError:
        return bool(vec)


def row_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doi": row.get("doi", ""),
        "source_file": row.get("source_file", ""),
        "section": row.get("section", ""),
        "heading_path": row.get("heading_path", ""),
        "index_version": row.get("index_version", ""),
        "has_metrics": row.get("has_metrics", False),
        "content_type": row.get("content_type", "prose"),
        "token_count": row.get("token_count", 0),
    }


def vector_index_healthy(collection) -> bool:
    try:
        if collection.count() == 0:
            return True
        data = collection.get(limit=1, include=["embeddings"])
        embeddings = data.get("embeddings")
        if embeddings is None:
            return False
        if len(embeddings) == 0:
            return False
        return _embedding_nonempty(embeddings[0])
    except InternalError:
        return False


def probe_collection(collection, *, limit: int = 1) -> bool:
    """True when reads work (count alone is not enough on Chroma 1.5.x)."""
    try:
        collection.get(limit=limit, include=["metadatas"])
        return True
    except InternalError:
        return False


def _vector_segment_ids() -> list[str]:
    db_path = CHROMA_DIR / "chroma.sqlite3"
    if not db_path.is_file():
        return []
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id
            FROM segments
            WHERE scope = 'VECTOR'
            """
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        con.close()


def repair_hnsw_segments() -> bool:
    """
    Delete corrupted on-disk HNSW segment dirs so Chroma rebuilds from WAL/sqlite.
    Returns True when at least one segment dir was removed.
    """
    repaired = False
    for segment_id in _vector_segment_ids():
        segment_dir = CHROMA_DIR / segment_id
        if segment_dir.exists():
            backup = segment_dir.with_name(f"{segment_id}_broken_{int(time.time())}")
            try:
                segment_dir.rename(backup)
            except OSError:
                shutil.rmtree(segment_dir, ignore_errors=True)
            repaired = True
    return repaired


def ensure_collection_readable(
    chroma: Optional[chromadb.PersistentClient] = None,
    collection=None,
):
    """
    Ensure collection.get() works. Rebuilds broken HNSW segments when needed.
    Returns (client, collection).
    """
    own_client = chroma is None
    client = chroma or create_chroma_client()
    col = collection or open_collection(client)

    if probe_collection(col):
        return client, col

    print(
        "[WARN] Chroma HNSW index unreadable — rebuilding vector segment from WAL…",
        flush=True,
    )
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass

    repair_hnsw_segments()
    client = create_chroma_client()
    col = open_collection(client)
    if not probe_collection(col):
        raise RuntimeError(
            "Chroma index still unreadable after HNSW repair. "
            "Close other Python processes using chroma_db and rerun build_index.py."
        )
    print("[INFO] Chroma HNSW index repaired.", flush=True)
    return client, col


def chunk_index_from_id(chunk_id: str) -> int:
    match = _CHUNK_INDEX_RE.search(chunk_id)
    return int(match.group(1)) if match else 0


def _metadata_value(row: tuple) -> Any:
    _id, key, string_value, int_value, float_value, bool_value = row
    if string_value is not None:
        return string_value
    if int_value is not None:
        return int_value
    if float_value is not None:
        return float_value
    if bool_value is not None:
        return bool(bool_value)
    return ""


def fetch_chunks_from_sqlite() -> list[dict]:
    """Read chunks directly from chroma.sqlite3 when the HNSW reader is broken."""
    db_path = CHROMA_DIR / "chroma.sqlite3"
    if not db_path.is_file():
        return []

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT e.embedding_id, m.key,
                   m.string_value, m.int_value, m.float_value, m.bool_value
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
            """
        )

        by_id: dict[str, dict[str, Any]] = {}
        for embedding_id, key, s, i, f, b in cur.fetchall():
            row = by_id.setdefault(
                embedding_id,
                {
                    "chunk_id": embedding_id,
                    "doi": "",
                    "source_file": "",
                    "heading_path": "",
                    "section": "",
                    "has_metrics": False,
                    "content_type": "prose",
                    "index_version": "",
                    "token_count": "",
                    "text": "",
                },
            )
            value = _metadata_value((None, key, s, i, f, b))
            if key == "chroma:document":
                row["text"] = value
            elif key in row:
                row[key] = value

        rows = list(by_id.values())
        for row in rows:
            row["char_count"] = len(row.get("text") or "")
        rows.sort(key=lambda r: (r["doi"], chunk_index_from_id(r["chunk_id"])))
        return rows
    finally:
        con.close()


def repair_vectors_from_sqlite(*, batch_size: int = 32) -> int:
    """
    Rebuild Chroma vector index from sqlite text/metadata when HNSW binaries are missing.
    Keeps chunk text in sqlite; re-embeds and re-adds all records sequentially.
    """
    from embedder import embed_texts
    from tqdm import tqdm

    rows = fetch_chunks_from_sqlite()
    if not rows:
        return 0

    client = create_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=HNSW_CREATE_METADATA,
    )

    for start in tqdm(
        range(0, len(rows), batch_size),
        desc="Re-embedding chunks",
        unit="batch",
    ):
        batch = rows[start : start + batch_size]
        texts = [row["text"] for row in batch]
        embeddings = embed_texts(texts, is_query=False)
        collection.add(
            ids=[row["chunk_id"] for row in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[row_to_metadata(row) for row in batch],
        )

    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass

    ensure_collection_readable()
    return len(rows)


def fetch_all_chunks(*, include_embeddings: bool = False) -> list[dict]:
    """Load all chunks via Chroma API, with sqlite fallback."""
    include = ["documents", "metadatas"]
    if include_embeddings:
        include.append("embeddings")

    client, collection = ensure_collection_readable()
    try:
        if include_embeddings and not vector_index_healthy(collection):
            print(
                "[WARN] Vector index missing — rebuilding embeddings from sqlite text…",
                flush=True,
            )
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            repair_vectors_from_sqlite()
            client, collection = ensure_collection_readable()

        try:
            data = collection.get(include=include)
        except InternalError:
            rows = fetch_chunks_from_sqlite()
            if rows:
                print(
                    f"[WARN] Used sqlite fallback for export ({len(rows)} chunks).",
                    flush=True,
                )
                return rows
            raise

        rows: list[dict] = []
        embeddings = data.get("embeddings")
        if embeddings is None:
            embeddings = []
        for idx, (chunk_id, doc, meta) in enumerate(
            zip(
                data["ids"],
                data["documents"],
                data["metadatas"],
            )
        ):
            meta = meta or {}
            text = doc or ""
            row = {
                "chunk_id": chunk_id,
                "doi": meta.get("doi", ""),
                "source_file": meta.get("source_file", ""),
                "heading_path": meta.get("heading_path", ""),
                "section": meta.get("section", ""),
                "has_metrics": meta.get("has_metrics", False),
                "content_type": meta.get("content_type", "prose"),
                "index_version": meta.get("index_version", ""),
                "token_count": meta.get("token_count", ""),
                "char_count": len(text),
                "text": text,
            }
            if (
                include_embeddings
                and idx < len(embeddings)
                and _embedding_nonempty(embeddings[idx])
            ):
                row["embedding_dim"] = len(embeddings[idx])
                row["embedding"] = list(embeddings[idx])
            rows.append(row)

        rows.sort(key=lambda r: (r["doi"], chunk_index_from_id(r["chunk_id"])))
        return rows
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
