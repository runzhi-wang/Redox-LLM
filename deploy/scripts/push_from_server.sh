#!/usr/bin/env bash
# Push latest app files from server working copy to GitHub (after deploy key is added).
set -euo pipefail

APP_SRC="${APP_SRC:-/home/ubuntu/redox-llm}"
WORK="${WORK:-/home/ubuntu/redox-llm-git}"
KEY="$HOME/.ssh/redox_llm_deploy"
REPO="git@github.com:runzhi-wang/Redox-LLM.git"

export GIT_SSH_COMMAND="ssh -i ${KEY} -o StrictHostKeyChecking=accept-new"

if [[ ! -f "$KEY" ]]; then
  echo "Run bootstrap_github_key.sh first and add deploy key to GitHub."
  exit 1
fi

rm -rf "$WORK"
git clone "$REPO" "$WORK"
cd "$WORK"

rsync -a --delete \
  --exclude '.git' \
  --exclude 'chroma_db' \
  --exclude 'chroma_db_corrupt_*' \
  --exclude 'data' \
  --exclude 'output' \
  --exclude '.env' \
  --exclude '__pycache__' \
  "$APP_SRC"/ ./

if git diff --quiet && git diff --cached --quiet; then
  echo "No changes to push."
  exit 0
fi

git add -A
git config user.email "deploy@redox-llm.local"
git config user.name "Redox LLM Deploy"
git commit -m "Sync from server: fix README encoding and deploy scripts"
git push origin main
echo "Pushed to GitHub."
