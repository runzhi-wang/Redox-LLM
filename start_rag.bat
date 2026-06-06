@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title Electrode Material LLM

echo ============================================================
echo   Electrode Material LLM Platform
echo ============================================================
echo.
echo Workdir: %CD%
echo.

set "PY=D:\bge-m3-local\pyenv\Scripts\python.exe"
if exist "%PY%" (
    echo Python: %PY%
) else (
    echo Python: system default
    set "PY=python"
)

if not exist ".env" (
    echo [ERROR] Missing .env - copy .env.example and set OPENAI_API_KEY
    goto :fail
)

if not exist "chroma_db" (
    echo [ERROR] Missing chroma_db - run rebuild index bat first
    goto :fail
)

set "EMBED_BACKEND=cloud"

echo [1/3] Check Python...
"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    goto :fail
)

echo [2/3] Install deps...
"%PY%" -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] pip install failed
    goto :fail
)

"%PY%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] streamlit not installed
    goto :fail
)

echo [3/3] Starting Streamlit...
echo Open: http://localhost:8501
echo Press Ctrl+C to stop, or close this window.
echo.
"%PY%" -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
    echo.
    echo [ERROR] Streamlit exited with code %ERR%
    goto :fail
)
goto :done

:fail
set "ERR=1"
echo.
pause
exit /b %ERR%

:done
pause
endlocal & exit /b 0
