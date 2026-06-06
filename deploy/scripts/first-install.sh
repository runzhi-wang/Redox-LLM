#!/usr/bin/env bash
# First-time server setup for Redox-LLM (Tencent Lighthouse / Ubuntu).
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/redox-llm}"
REPO="${REPO:-https://github.com/runzhi-wang/Redox-LLM.git}"

echo "==> Installing Docker (if missing)..."
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi
if ! docker compose version >/dev/null 2>&1; then
  apt-get update
  apt-get install -y docker-compose-plugin git
fi

echo "==> Cloning app to ${APP_DIR}..."
mkdir -p "$(dirname "$APP_DIR")"
if [[ ! -d "${APP_DIR}/.git" ]]; then
  git clone "$REPO" "$APP_DIR"
else
  git -C "$APP_DIR" pull --ff-only
fi
cd "$APP_DIR"

mkdir -p data/md data/chroma_db data/output

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ""
  echo "!! Edit secrets before starting:"
  echo "   nano ${APP_DIR}/.env"
  echo "   Required: OPENAI_API_KEY, OER_RAG_ADMIN_KEY"
  echo ""
fi

export MD_DATA_DIR="${APP_DIR}/data/md"
export CHROMA_DATA_DIR="${APP_DIR}/data/chroma_db"
export CHAT_OUTPUT_DIR="${APP_DIR}/data/output"

chmod +x deploy/scripts/*.sh

echo "==> Building and starting containers..."
docker compose up -d --build

echo ""
echo "Done. Open: http://$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}'):8501"
echo "If unreachable, open TCP 8501 in the cloud firewall / security group."
