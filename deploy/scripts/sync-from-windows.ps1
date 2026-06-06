# Upload local corpus, vector index, and .env to the cloud server.
# Usage (PowerShell, will prompt for SSH password):
#   .\deploy\scripts\sync-from-windows.ps1 -Server 62.234.27.253 -User root

param(
    [Parameter(Mandatory = $true)]
    [string]$Server,
    [string]$User = "root",
    [string]$AppDir = "/opt/redox-llm",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$chroma = Join-Path $ProjectRoot "chroma_db"
$envFile = Join-Path $ProjectRoot ".env"
$corpus = "E:\Desktop\Nature药物发现\OER\OER_md"

$remote = "${User}@${Server}"
$sshTarget = "${remote}:${AppDir}"

Write-Host "Project: $ProjectRoot"
Write-Host "Server:  $remote"
Write-Host ""

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "OpenSSH client not found. Install: Settings -> Apps -> Optional features -> OpenSSH Client"
}

Write-Host "==> 1/4 Run first-install on server (Docker + git clone)..."
Get-Content (Join-Path $PSScriptRoot "first-install.sh") -Raw | ssh $remote "bash -s"

Write-Host "==> 2/4 Upload literature (.md)..."
if (Test-Path $corpus) {
    ssh $remote "mkdir -p ${AppDir}/data/md"
    scp -r "$corpus\*" "${sshTarget}/data/md/"
} else {
    Write-Warning "Corpus not found at $corpus — skip or set path manually."
}

Write-Host "==> 3/4 Upload chroma_db (~300MB, may take a few minutes)..."
if (Test-Path $chroma) {
    ssh $remote "mkdir -p ${AppDir}/data/chroma_db"
    scp -r "$chroma\*" "${sshTarget}/data/chroma_db/"
} else {
    Write-Warning "chroma_db missing — run rebuild-index.sh on server after upload."
}

Write-Host "==> 4/4 Upload .env (API keys)..."
if (Test-Path $envFile) {
    scp "$envFile" "${sshTarget}/.env"
} else {
    Write-Warning ".env missing locally — edit ${AppDir}/.env on server manually."
}

Write-Host "==> Restart app..."
ssh $remote @"
cd ${AppDir}
export MD_DATA_DIR=${AppDir}/data/md
export CHROMA_DATA_DIR=${AppDir}/data/chroma_db
export CHAT_OUTPUT_DIR=${AppDir}/data/output
docker compose up -d --build
"@

Write-Host ""
Write-Host "Team URL: http://${Server}:8501"
Write-Host "If blocked, open port 8501 in Tencent Lighthouse firewall."
