#!/usr/bin/env bash
# Rebuild Chroma vector index inside Docker (uses cloud BGE-M3 by default).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and fill in secrets first."
  exit 1
fi

echo "Rebuilding index (this may take a while)..."
docker compose run --rm \
  -e EMBED_BACKEND=cloud \
  -e OER_RAG_MD_DIR=/data/md \
  -e OER_RAG_CHROMA_DIR=/data/chroma_db \
  oer-rag python -u build_index.py

echo "Index rebuild finished. Restart app: docker compose up -d"
