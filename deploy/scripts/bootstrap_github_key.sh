#!/usr/bin/env bash
# One-time: create deploy key for pushing to GitHub from this server.
set -euo pipefail

KEY="$HOME/.ssh/redox_llm_deploy"
if [[ ! -f "$KEY" ]]; then
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "redox-llm-deploy"
fi
echo ""
echo "=== Add this Deploy Key to GitHub ==="
echo "Repo: runzhi-wang/Redox-LLM -> Settings -> Deploy keys -> Add"
echo "Title: redox-llm-server | Allow write access: YES"
echo ""
cat "${KEY}.pub"
echo ""
echo "After adding, run: ./deploy/scripts/push_from_server.sh"
