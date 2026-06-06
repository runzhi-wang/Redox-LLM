"""Unified embedding: local BGE-M3 worker or cloud BGE-M3 API (parallel batches)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from config import (
    CLOUD_EMBED_MODEL,
    EMBED_API_MAX_RETRIES,
    EMBED_API_PARALLEL,
    EMBED_API_RETRY_BACKOFF_SEC,
    EMBED_API_TIMEOUT_SEC,
    EMBED_BACKEND,
    EMBED_BATCH_SIZE,
    EMBED_BATCH_TIMEOUT_SEC,
    EMBED_MAX_CHARS_INDEX,
    EMBED_MAX_CHARS_QUERY,
    INDEX_VERSION,
    LOCAL_EMBED_DIR,
    LOCAL_EMBED_PYTHON,
    get_embed_client,
    is_cloud_embed,
)

_worker_lock = threading.Lock()
_worker_proc: Optional[subprocess.Popen] = None


def _local_python() -> Path:
    custom = Path(LOCAL_EMBED_PYTHON) if LOCAL_EMBED_PYTHON else None
    if custom and custom.is_file():
        return custom
    default = Path(LOCAL_EMBED_DIR) / "pyenv" / "Scripts" / "python.exe"
    if default.is_file():
        return default
    raise RuntimeError(
        f"Local embed Python not found. Expected: {default}. "
        "Run D:\\bge-m3-local\\install.bat first."
    )


def _drain_stderr(proc: subprocess.Popen) -> None:
    assert proc.stderr is not None
    for _line in proc.stderr:
        pass


def _start_worker() -> subprocess.Popen:
    python = _local_python()
    worker = Path(LOCAL_EMBED_DIR) / "embed_worker.py"
    if not worker.is_file():
        raise RuntimeError(f"Missing embed_worker.py at {worker}")

    proc = subprocess.Popen(
        [str(python), "-u", str(worker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=str(LOCAL_EMBED_DIR),
        bufsize=1,
    )
    assert proc.stderr is not None
    while True:
        line = proc.stderr.readline()
        if not line:
            proc.kill()
            raise RuntimeError("Embed worker stderr closed before ready.")
        print(line.rstrip(), file=sys.stderr)
        if "ready" in line.lower():
            break

    threading.Thread(target=_drain_stderr, args=(proc,), daemon=True).start()
    return proc


def _get_worker() -> subprocess.Popen:
    global _worker_proc
    if _worker_proc is None or _worker_proc.poll() is not None:
        _worker_proc = _start_worker()
    return _worker_proc


def _prepare_texts(texts: list[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        return texts
    return [t[:max_chars] if len(t) > max_chars else t for t in texts]


def _embed_via_worker(
    texts: list[str],
    is_query: bool,
    max_chars: int,
) -> list[list[float]]:
    payload = json.dumps(
        {
            "texts": texts,
            "batch_size": EMBED_BATCH_SIZE,
            "is_query": is_query,
            "max_chars": max_chars,
        },
        ensure_ascii=False,
    )
    with _worker_lock:
        proc = _get_worker()
        assert proc.stdin and proc.stdout
        proc.stdin.write(payload + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Embed worker returned no output.")
        if proc.poll() is not None:
            raise RuntimeError("Embed worker exited unexpectedly.")
    data = json.loads(line)
    return data["embeddings"]


def _is_retryable_api_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "TimeoutError",
    }:
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("timeout", "timed out", "connection", "429", "502", "503", "504")
    )


def _embed_api_batch(batch: list[str]) -> list[list[float]]:
    last_err: Exception | None = None
    for attempt in range(1, EMBED_API_MAX_RETRIES + 1):
        try:
            client = get_embed_client()
            response = client.embeddings.create(model=CLOUD_EMBED_MODEL, input=batch)
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= EMBED_API_MAX_RETRIES or not _is_retryable_api_error(exc):
                raise
            wait_s = EMBED_API_RETRY_BACKOFF_SEC * attempt
            print(
                f"[WARN] embed API attempt {attempt}/{EMBED_API_MAX_RETRIES} failed: "
                f"{exc}; retry in {wait_s:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_s)
    raise RuntimeError(f"embed API failed after retries: {last_err}")


def _embed_via_api(texts: list[str], max_chars: int) -> list[list[float]]:
    prepared = _prepare_texts(texts, max_chars)
    if not prepared:
        return []

    if len(prepared) == 1:
        return _embed_api_batch(prepared)

    batches: list[list[str]] = []
    for start in range(0, len(prepared), EMBED_BATCH_SIZE):
        batches.append(prepared[start : start + EMBED_BATCH_SIZE])

    workers = min(EMBED_API_PARALLEL, len(batches))
    ordered: list[Optional[list[list[float]]]] = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_embed_api_batch, batch): idx for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                ordered[idx] = future.result(timeout=EMBED_BATCH_TIMEOUT_SEC)
            except TimeoutError as exc:
                raise RuntimeError(
                    f"embed batch timeout after {EMBED_BATCH_TIMEOUT_SEC:.0f}s "
                    f"(batch {idx + 1}/{len(batches)})"
                ) from exc

    merged: list[list[float]] = []
    for part in ordered:
        if part:
            merged.extend(part)
    return merged


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    if not texts:
        return []

    max_chars = EMBED_MAX_CHARS_QUERY if is_query else EMBED_MAX_CHARS_INDEX

    if EMBED_BACKEND == "local":
        return _embed_via_worker(texts, is_query=is_query, max_chars=max_chars)

    return _embed_via_api(texts, max_chars=max_chars)


def embed_model_name() -> str:
    trunc_index = (
        "full text"
        if EMBED_MAX_CHARS_INDEX == 0
        else f"max {EMBED_MAX_CHARS_INDEX} chars"
    )
    if is_cloud_embed():
        return (
            f"{CLOUD_EMBED_MODEL} (cloud, parallel={EMBED_API_PARALLEL}, "
            f"timeout={EMBED_API_TIMEOUT_SEC:.0f}s, retry={EMBED_API_MAX_RETRIES}, "
            f"{trunc_index})"
        )
    return f"BAAI/bge-m3 (local, {trunc_index}, {INDEX_VERSION})"
