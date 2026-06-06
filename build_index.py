"""Build Chroma vector index with resume, v2 chunking, graceful stop."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.errors import InternalError, NotFoundError
from tqdm import tqdm

from chunking import chunk_markdown, doi_from_filename, warmup_tokenizer
from config import (
    CHROMA_DIR,
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    COLLECTION_NAME,
    EMBED_API_MAX_RETRIES,
    EMBED_API_PARALLEL,
    EMBED_API_TIMEOUT_SEC,
    EMBED_BACKEND,
    EMBED_BATCH_TIMEOUT_SEC,
    ETA_WINDOW,
    INDEX_FILE_PARALLEL,
    INDEX_FILE_TIMEOUT_SEC,
    INDEX_VERSION,
    MD_DIR,
    OUTPUT_DIR,
    is_cloud_embed,
)
from chroma_utils import (
    HNSW_CREATE_METADATA,
    create_chroma_client,
    ensure_collection_readable,
    open_collection,
    probe_collection,
    repair_hnsw_segments,
    repair_vectors_from_sqlite,
    vector_index_healthy,
)
from embedder import embed_model_name, embed_texts
from index_utils import (
    chunk_to_metadata,
    decide_file_action,
    delete_chunks_for_doi,
    load_progress,
    mark_file_done,
    mark_file_failed,
    save_progress,
)

_stop_requested = False
_current_doi: Optional[str] = None
_chroma_lock = threading.Lock()
_progress_lock = threading.Lock()
_timed_out_files: set[str] = set()
_timed_out_lock = threading.Lock()


def _request_stop(signum, frame) -> None:  # noqa: ARG001
    global _stop_requested
    _stop_requested = True
    print("\n[INFO] Stop requested — finishing in-flight work then exiting…", file=sys.stderr)


def _wipe_collection(chroma) -> None:
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass
    chroma.create_collection(
        name=COLLECTION_NAME,
        metadata=HNSW_CREATE_METADATA,
    )


def _repair_chroma_db() -> tuple[chromadb.PersistentClient, object]:
    if CHROMA_DIR.exists():
        backup = CHROMA_DIR.with_name(
            f"{CHROMA_DIR.name}_corrupt_{int(time.time())}"
        )
        try:
            CHROMA_DIR.rename(backup)
            print(f"[INFO] Moved corrupt DB to {backup}", file=sys.stderr)
        except OSError:
            try:
                shutil.rmtree(CHROMA_DIR)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot repair chroma_db (files in use). "
                    f"Close Streamlit/other Python processes and retry. ({exc})"
                ) from exc
    chroma = create_chroma_client()
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata=HNSW_CREATE_METADATA,
    )
    return chroma, collection


def _reset_progress_after_chroma_repair(progress: dict) -> None:
    """Chroma empty after repair — do not skip files marked done in progress."""
    progress["files"] = {}
    progress["index_version"] = INDEX_VERSION
    save_progress(progress)


def _resolve_action(collection, md_path: Path, progress: dict, rebuild_all: bool) -> str:
    action = decide_file_action(md_path, progress, rebuild_all=rebuild_all)
    if action != "index":
        return action

    doi = doi_from_filename(md_path)
    try:
        probe = collection.get(where={"doi": doi}, include=["metadatas"], limit=1)
    except InternalError:
        return "index"
    if probe.get("ids") and probe.get("metadatas"):
        old_ver = (probe["metadatas"][0] or {}).get("index_version", "v1")
        if old_ver != INDEX_VERSION:
            return "reindex"
    return action


def _index_one_file(
    md_path: Path,
    collection,
    progress: dict,
    action: str,
) -> dict:
    """Index a single MD file. Thread-safe Chroma/progress updates via locks."""
    global _current_doi

    doi = doi_from_filename(md_path)
    t0 = time.time()
    result = {
        "file": md_path.name,
        "ok": False,
        "chunks": 0,
        "elapsed": 0.0,
        "error": "",
        "action": action,
    }

    try:
        if action == "reindex":
            with _chroma_lock:
                deleted = delete_chunks_for_doi(collection, doi)
            if deleted:
                result["reindex_removed"] = deleted

        _current_doi = doi
        chunks = chunk_markdown(md_path)
        if not chunks:
            raise RuntimeError("no chunks produced")

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts, is_query=False)

        with _timed_out_lock:
            if md_path.name in _timed_out_files:
                result["error"] = "completed after per-file timeout (ignored)"
                return result

        with _chroma_lock:
            collection.add(
                ids=[c.chunk_id for c in chunks],
                documents=texts,
                embeddings=embeddings,
                metadatas=[chunk_to_metadata(c) for c in chunks],
            )

        with _timed_out_lock:
            if md_path.name in _timed_out_files:
                result["error"] = "completed after per-file timeout (ignored)"
                return result

        with _progress_lock:
            mark_file_done(progress, md_path, doi, len(chunks))

        result["ok"] = True
        result["chunks"] = len(chunks)
        result["elapsed"] = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        with _chroma_lock:
            delete_chunks_for_doi(collection, doi)
        err = str(exc)
        with _progress_lock:
            mark_file_failed(progress, md_path, err)
        result["error"] = err
        result["elapsed"] = time.time() - t0
    finally:
        _current_doi = None

    return result


def _make_timeout_result(
    md_path: Path,
    progress: dict,
    collection,
    action: str,
    started_at: float,
) -> dict:
    err = f"index timeout after {INDEX_FILE_TIMEOUT_SEC:.0f}s"
    doi = doi_from_filename(md_path)
    with _chroma_lock:
        delete_chunks_for_doi(collection, doi)
    with _timed_out_lock:
        _timed_out_files.add(md_path.name)
    with _progress_lock:
        mark_file_failed(progress, md_path, err)
    return {
        "file": md_path.name,
        "ok": False,
        "chunks": 0,
        "elapsed": time.time() - started_at,
        "error": err,
        "action": action,
    }


def _run_pending_jobs(
    pending: list[tuple[Path, str]],
    collection,
    progress: dict,
    file_workers: int,
    handle_result,
) -> None:
    if not pending:
        return

    if file_workers <= 1:
        for md_path, action in pending:
            if _stop_requested:
                break
            started = time.time()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    _index_one_file, md_path, collection, progress, action
                )
                try:
                    res = future.result(timeout=INDEX_FILE_TIMEOUT_SEC)
                except TimeoutError:
                    res = _make_timeout_result(
                        md_path, progress, collection, action, started
                    )
                    tqdm.write(f"TIMEOUT {md_path.name}: {res['error']}")
            handle_result(res)
            if _stop_requested:
                tqdm.write("[INFO] Stopped after completing current file.")
                break
        return

    future_meta: dict[Future, tuple[Path, str, float]] = {}
    with ThreadPoolExecutor(max_workers=file_workers) as pool:
        for md_path, action in pending:
            if _stop_requested:
                break
            future = pool.submit(
                _index_one_file, md_path, collection, progress, action
            )
            future_meta[future] = (md_path, action, time.time())

        pending_futures = set(future_meta.keys())
        while pending_futures and not _stop_requested:
            done, _ = wait(pending_futures, timeout=1.0, return_when="FIRST_COMPLETED")
            now = time.time()

            for future in list(pending_futures):
                md_path, action, started = future_meta[future]
                if future in done:
                    pending_futures.discard(future)
                    res = future.result()
                    handle_result(res)
                    continue
                if now - started >= INDEX_FILE_TIMEOUT_SEC:
                    pending_futures.discard(future)
                    res = _make_timeout_result(
                        md_path, progress, collection, action, started
                    )
                    tqdm.write(f"TIMEOUT {md_path.name}: {res['error']}")
                    handle_result(res)

    if _stop_requested:
        tqdm.write("[INFO] Stop requested; in-flight files may still finish.")


def main() -> int:
    global _current_doi
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build OER literature Chroma index")
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Delete index and progress, rebuild from scratch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume: skip up-to-date files (default)",
    )
    args = parser.parse_args()
    rebuild_all = args.rebuild_all

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)

    md_files = sorted(MD_DIR.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {MD_DIR}")
        return 1

    file_workers = max(1, INDEX_FILE_PARALLEL if is_cloud_embed() else 1)

    print(f"Index version: {INDEX_VERSION}")
    print(f"Embedding backend: {EMBED_BACKEND} ({embed_model_name()})")
    if is_cloud_embed():
        print(f"API parallel batches: {EMBED_API_PARALLEL} | file workers: {file_workers}")
        print(
            f"Timeouts: embed {EMBED_API_TIMEOUT_SEC:.0f}s/batch "
            f"(wait {EMBED_BATCH_TIMEOUT_SEC:.0f}s), "
            f"retry {EMBED_API_MAX_RETRIES}, "
            f"per-file {INDEX_FILE_TIMEOUT_SEC:.0f}s"
        )
    print(f"Mode: {'rebuild-all' if rebuild_all else 'resume'}")

    # Chroma must initialize before transformers on Windows (reverse order segfaults).
    chroma = create_chroma_client()

    if rebuild_all:
        _wipe_collection(chroma)
        progress = {"index_version": INDEX_VERSION, "files": {}, "stats": {}}
        save_progress(progress)
    else:
        progress = load_progress()
        if progress.get("index_version") != INDEX_VERSION:
            print(
                f"[INFO] Progress version {progress.get('index_version')} != "
                f"{INDEX_VERSION}; outdated files will be reindexed."
            )

    collection = open_collection(chroma)
    if not probe_collection(collection):
        if repair_hnsw_segments():
            try:
                chroma.close()
            except Exception:  # noqa: BLE001
                pass
            chroma = create_chroma_client()
            collection = open_collection(chroma)
        if not probe_collection(collection):
            print(
                "[WARN] Chroma index corrupted (hnsw load failed). "
                "Rebuilding empty chroma_db; all papers will be reindexed.",
                file=sys.stderr,
            )
            chroma, collection = _repair_chroma_db()
            if not rebuild_all:
                _reset_progress_after_chroma_repair(progress)

    print("Loading BGE-M3 tokenizer (once)...", flush=True)
    warmup_tokenizer()

    failed: list[dict[str, str]] = []
    total_chunks = 0
    skipped = 0
    recent_times: deque[float] = deque(maxlen=ETA_WINDOW)

    pending: list[tuple[Path, str]] = []
    for md_path in md_files:
        action = _resolve_action(collection, md_path, progress, rebuild_all)
        if action == "skip":
            skipped += 1
        else:
            pending.append((md_path, action))

    print(
        f"Files: {len(md_files)} total | to index {len(pending)} | skip {skipped}",
        flush=True,
    )
    if not pending:
        print("Nothing to index — collection is up to date.", flush=True)

    pbar = tqdm(total=len(pending), desc="Indexing MD", unit="file")

    def _handle_result(res: dict) -> None:
        nonlocal total_chunks
        if res.get("reindex_removed"):
            tqdm.write(
                f"Reindex {res['file']}: removed {res['reindex_removed']} old chunks"
            )
        if res["ok"]:
            total_chunks += res["chunks"]
            recent_times.append(res["elapsed"])
            avg = sum(recent_times) / len(recent_times)
            done_count = sum(
                1 for f in progress.get("files", {}).values() if f.get("status") == "done"
            )
            remaining = max(0, len(md_files) - done_count - skipped)
            eta_min = (remaining * avg) / 60 if recent_times else 0
            pbar.set_postfix(
                last_s=f"{res['elapsed']:.0f}s",
                avg_s=f"{avg:.0f}s",
                eta_min=f"{eta_min:.0f}",
                skip=skipped,
            )
        else:
            failed.append({"file": res["file"], "error": res["error"]})
            tqdm.write(f"FAILED {res['file']}: {res['error']}")
        pbar.update(1)

    _run_pending_jobs(
        pending,
        collection,
        progress,
        file_workers,
        _handle_result,
    )

    pbar.close()

    try:
        chroma.close()
    except Exception:  # noqa: BLE001
        pass
    verify_client, final_collection = ensure_collection_readable()
    if not vector_index_healthy(final_collection):
        print(
            "[WARN] Vector index empty after build — re-embedding all chunks…",
            file=sys.stderr,
        )
        try:
            verify_client.close()
        except Exception:  # noqa: BLE001
            pass
        repaired = repair_vectors_from_sqlite()
        print(f"[INFO] Rebuilt vectors for {repaired} chunks.", file=sys.stderr)

    done_files = sum(
        1 for f in progress.get("files", {}).values() if f.get("status") == "done"
    )
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "index_version": INDEX_VERSION,
        "chunk_min_tokens": CHUNK_MIN_TOKENS,
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "md_dir": str(MD_DIR),
        "files_total": len(md_files),
        "files_done": done_files,
        "files_skipped": skipped,
        "files_failed": len(failed),
        "chunks_indexed_this_run": total_chunks,
        "collection": COLLECTION_NAME,
        "chroma_dir": str(CHROMA_DIR),
        "embed_model": embed_model_name(),
        "embed_backend": EMBED_BACKEND,
        "embed_api_parallel": EMBED_API_PARALLEL if is_cloud_embed() else 0,
        "index_file_parallel": file_workers,
        "stopped_early": _stop_requested,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "index_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if failed:
        failed_path = OUTPUT_DIR / "failed_files.csv"
        with failed_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "error"])
            writer.writeheader()
            writer.writerows(failed)

    print(
        f"Done this run: +{total_chunks} chunks | "
        f"total done {done_files}/{len(md_files)} | skipped {skipped} | "
        f"failed {len(failed)}"
    )
    if _stop_requested:
        print("Resume later with: python build_index.py  (default --resume)")
    if failed:
        print(f"Failures logged to {OUTPUT_DIR / 'failed_files.csv'}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
