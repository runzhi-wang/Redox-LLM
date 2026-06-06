# Push to GitHub via cloud server (when local network cannot reach GitHub).
# Prereq: SSH key login to server works.
param(
    [string]$Server = "62.234.27.253",
    [string]$User = "ubuntu",
    [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519_redox"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$remote = "${User}@${Server}"

Write-Host "1) Upload latest code to server..."
ssh -i $Key $remote "mkdir -p /home/ubuntu/redox-llm"
scp -i $Key -r "$root\README.md" "$root\docker-compose.yml" "$root\.dockerignore" "${remote}:/home/ubuntu/redox-llm/"
scp -i $Key -r "$root\deploy" "${remote}:/home/ubuntu/redox-llm/"

Write-Host "2) Setup GitHub deploy key (one-time)..."
ssh -i $Key $remote "chmod +x /home/ubuntu/redox-llm/deploy/scripts/*.sh; bash /home/ubuntu/redox-llm/deploy/scripts/bootstrap_github_key.sh"

Write-Host ""
Write-Host ">>> Copy the deploy key above into GitHub:"
Write-Host "    https://github.com/runzhi-wang/Redox-LLM/settings/keys"
Write-Host "    Enable WRITE access, then press Enter here..."
Read-Host

Write-Host "3) Push from server to GitHub..."
ssh -i $Key $remote "bash /home/ubuntu/redox-llm/deploy/scripts/push_from_server.sh"
Write-Host "Done. Refresh https://github.com/runzhi-wang/Redox-LLM"
