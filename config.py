"""Shared configuration for OER literature RAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
_md_override = os.getenv("OER_RAG_MD_DIR")
MD_DIR = Path(_md_override).expanduser() if _md_override else ROOT / "md"
_chroma_override = os.getenv("OER_RAG_CHROMA_DIR")
CHROMA_DIR = (
    Path(_chroma_override).expanduser()
    if _chroma_override
    else ROOT / "oer_rag" / "chroma_db"
)
COLLECTION_NAME = "oer_md"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROGRESS_PATH = OUTPUT_DIR / "index_progress.json"

_env_path = Path(__file__).resolve().parent / ".env"
# Bat may set EMBED_BACKEND before Python starts; keep that after loading .env.
_runtime_embed_backend = os.getenv("EMBED_BACKEND") or os.getenv(
    "OER_RAG_EMBED_BACKEND"
)
# override=True: project .env API key wins over stale system OPENAI_API_KEY.
load_dotenv(_env_path, override=True)
if _runtime_embed_backend:
    os.environ["EMBED_BACKEND"] = _runtime_embed_backend

# Index schema version — bump when chunking / embed strategy changes.
INDEX_VERSION = os.getenv("OER_RAG_INDEX_VERSION", "v2.2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "120"))

# Cloud embed: per-request timeout + manual retry (index build).
EMBED_API_TIMEOUT_SEC = float(os.getenv("OER_RAG_EMBED_API_TIMEOUT_SEC", "90"))
EMBED_API_MAX_RETRIES = int(os.getenv("OER_RAG_EMBED_API_MAX_RETRIES", "3"))
EMBED_API_RETRY_BACKOFF_SEC = float(
    os.getenv("OER_RAG_EMBED_API_RETRY_BACKOFF_SEC", "2")
)
EMBED_BATCH_TIMEOUT_SEC = float(os.getenv("OER_RAG_EMBED_BATCH_TIMEOUT_SEC", "120"))
# Skip one MD file if total indexing exceeds this (other files continue).
INDEX_FILE_TIMEOUT_SEC = float(os.getenv("OER_RAG_INDEX_FILE_TIMEOUT_SEC", "480"))
CHAT_MODEL_OPTIONS = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5",
    "gpt-5.4",
]
DEFAULT_CHAT_MODEL = "gpt-5-mini"
_default_chat = os.getenv("OER_RAG_DEFAULT_CHAT_MODEL") or os.getenv(
    "OPENAI_MODEL", DEFAULT_CHAT_MODEL
)
CHAT_MODEL = (
    _default_chat
    if _default_chat in CHAT_MODEL_OPTIONS
    else DEFAULT_CHAT_MODEL
)
ADMIN_KEY = os.getenv("OER_RAG_ADMIN_KEY", "")
CHAT_LOG_PATH = OUTPUT_DIR / "chat_logs.jsonl"
CHAT_LOG_EXPORT_PATH = OUTPUT_DIR / "chat_logs_export.xlsx"
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_raw_backend = (
    os.getenv("EMBED_BACKEND") or os.getenv("OER_RAG_EMBED_BACKEND") or "local"
).lower()
# cloud / api → remote BGE-M3; local → GPU worker
EMBED_BACKEND = "cloud" if _raw_backend in ("cloud", "api") else "local"

CLOUD_EMBED_MODEL = os.getenv("OER_RAG_CLOUD_EMBED_MODEL") or os.getenv(
    "CLOUD_EMBED_MODEL", "bge-m3"
)

LOCAL_EMBED_DIR = os.getenv("LOCAL_EMBED_DIR", r"D:\bge-m3-local")
LOCAL_EMBED_PYTHON = os.getenv(
    "LOCAL_EMBED_PYTHON", r"D:\bge-m3-local\pyenv\Scripts\python.exe"
)
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL") or os.getenv(
    "OER_RAG_LOCAL_EMBED_MODEL", "BAAI/bge-m3"
)
LOCAL_EMBED_DEVICE = os.getenv("LOCAL_EMBED_DEVICE") or os.getenv(
    "OER_RAG_LOCAL_EMBED_DEVICE", "cuda"
)
LOCAL_TOKENIZER_PATH = os.getenv(
    "LOCAL_TOKENIZER_PATH",
    str(Path(LOCAL_EMBED_DIR) / "models" / "modelscope" / "BAAI" / "bge-m3"),
)

TOP_K = int(os.getenv("OER_RAG_TOP_K", "8"))
_default_batch = "64" if EMBED_BACKEND == "local" else "16"
EMBED_BATCH_SIZE = int(os.getenv("OER_RAG_EMBED_BATCH", _default_batch))
EMBED_API_PARALLEL = int(os.getenv("OER_RAG_EMBED_API_PARALLEL", "25"))
_default_file_parallel = "25" if EMBED_BACKEND == "cloud" else "1"
INDEX_FILE_PARALLEL = int(
    os.getenv("OER_RAG_INDEX_FILE_PARALLEL", _default_file_parallel)
)

# 0 = embed full chunk text.
EMBED_MAX_CHARS_INDEX = int(os.getenv("OER_RAG_EMBED_MAX_CHARS_INDEX", "0"))
_query_default = os.getenv("OER_RAG_EMBED_MAX_CHARS_QUERY") or os.getenv(
    "OER_RAG_LOCAL_EMBED_MAX_LENGTH", "512"
)
EMBED_MAX_CHARS_QUERY = int(_query_default)

# Structural chunking: section → paragraph → merge → sentence split at MAX.
CHUNK_MIN_TOKENS = int(os.getenv("OER_RAG_CHUNK_MIN_TOKENS", "300"))
CHUNK_TARGET_TOKENS = int(os.getenv("OER_RAG_CHUNK_TARGET_TOKENS", "750"))
CHUNK_MAX_TOKENS = int(os.getenv("OER_RAG_CHUNK_MAX_TOKENS", "1024"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("OER_RAG_CHUNK_OVERLAP_TOKENS", "100"))

ETA_WINDOW = int(os.getenv("OER_RAG_ETA_WINDOW", "10"))

os.environ.setdefault("LOCAL_EMBED_MODEL", LOCAL_EMBED_MODEL)


def is_cloud_embed() -> bool:
    return EMBED_BACKEND == "cloud"


def get_client(*, timeout: float | None = None) -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            f"OPENAI_API_KEY not set. Copy .env.example to {_env_path} and fill in your key."
        )
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_retries=OPENAI_MAX_RETRIES,
        timeout=timeout if timeout is not None else OPENAI_TIMEOUT_SEC,
    )


def get_embed_client() -> OpenAI:
    return get_client(timeout=EMBED_API_TIMEOUT_SEC)
