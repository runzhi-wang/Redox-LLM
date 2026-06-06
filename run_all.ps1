$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已创建 .env，请填入 OPENAI_API_KEY 后重新运行。"
    exit 1
}

python -m pip install -r requirements.txt -q
python build_index.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$q = "中性 pH 下哪些 OER 催化剂策略可提升稳定性并降低过电位？"
python ask.py $q
