$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已创建 .env，请填入 OPENAI_API_KEY 后重新运行。"
    exit 1
}

$py = "D:\bge-m3-local\pyenv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:EMBED_BACKEND = "cloud"
& $py -m pip install -r requirements.txt -q
& $py -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
