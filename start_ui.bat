@echo off
REM Start the E-Commerce Churn Prediction UI
REM This script activates the virtual environment and launches the FastAPI + Gradio application

setlocal enabledelayedexpansion

echo Starting E-Commerce Churn Prediction UI...
echo.

REM Check if venv exists
if not exist "venv" (
    echo Virtual environment not found. Creating venv...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Display instructions
echo Virtual environment activated
echo.
echo Launching application...
echo.
echo The application will be available at:
echo   Web UI:  http://localhost:8000/ui
echo   API:     http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the FastAPI app with Gradio UI
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

pause
