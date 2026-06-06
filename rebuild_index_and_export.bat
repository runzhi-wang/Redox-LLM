@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
title OER RAG Rebuild Index

echo ============================================================
echo   OER RAG 重建索引 + 导出 Excel
echo ============================================================
echo.
echo 工作目录: %CD%
echo.

set "PY=D:\bge-m3-local\pyenv\Scripts\python.exe"
if exist "%PY%" (
    echo Python: %PY%
) else (
    echo Python: 系统默认
    set "PY=python"
)
echo.
echo 请选择嵌入模型:
echo   1 = 云端 BGE-M3  [dmxapi API, 25篇并行, 快速, 默认]
echo   2 = 本地 BGE-M3  [本地 RTX GPU, 无需API, 较慢]
echo.
set "EMBED_CHOICE="
set /p EMBED_CHOICE=请输入 1 或 2, 直接回车=1云端: 
if "!EMBED_CHOICE!"=="" set "EMBED_CHOICE=1"
if "!EMBED_CHOICE!"=="2" (
    set "EMBED_BACKEND=local"
    echo 已选择: 2 本地 BGE-M3 GPU
) else (
    set "EMBED_BACKEND=cloud"
    echo 已选择: 1 云端 BGE-M3
)
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python
    goto :fail
)

if not exist ".env" (
    echo [错误] 缺少 .env 文件
    goto :fail
)

if not exist "..\md" (
    echo [错误] 缺少 md 文件夹
    goto :fail
)

set MD_COUNT=0
for %%f in ("..\md\*.md") do set /a MD_COUNT+=1
echo 文献数量: !MD_COUNT!
echo.

if !MD_COUNT! EQU 0 (
    echo [错误] 没有 .md 文件
    goto :fail
)

echo [1/3] 安装依赖...
"%PY%" -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo [错误] pip install 失败
    goto :fail
)

echo [2/3] 构建索引...
"%PY%" -u build_index.py
set "BUILD_ERR=!ERRORLEVEL!"
if not "!BUILD_ERR!"=="0" (
    echo [错误] build_index.py 失败, 退出码: !BUILD_ERR!
    goto :fail
)

echo [3/3] 导出 Excel（含向量）...
"%PY%" -u export_chunks.py --with-vectors
if errorlevel 1 (
    echo [错误] export_chunks.py 失败
    goto :fail
)

echo.
echo 全部完成
set "ERR=0"
goto :done

:fail
echo.
echo 流程未完成，请查看上方错误信息
set "ERR=1"

:done
echo.
pause
endlocal & exit /b %ERR%
