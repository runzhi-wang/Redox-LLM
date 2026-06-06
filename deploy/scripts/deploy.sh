#!/usr/bin/env bash
# Pull latest code and restart the OER RAG stack.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy .env.example and fill in secrets first."
  exit 1
fi

git pull --ff-only

docker compose pull nginx 2>/dev/null || true
docker compose up -d --build

echo "Deploy complete. Streamlit: http://$(hostname -I | awk '{print $1}'):${STREAMLIT_PORT:-8501}"
