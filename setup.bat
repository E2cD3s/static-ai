@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || exit /b 1
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip wheel
pip install -r requirements.txt || exit /b 1

rem kokoro-onnx installs CPU onnxruntime; replace it with the CUDA build (same import name).
pip uninstall -y onnxruntime >nul 2>nul
pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.20,<1.24" || exit /b 1

python download_models.py kokoro || exit /b 1

if not exist config.yaml (
    copy config.example.yaml config.yaml >nul
    echo Created config.yaml - edit it with your Discord token and LLM endpoints.
)

echo.
echo Setup complete. Edit config.yaml, then run: run.bat
