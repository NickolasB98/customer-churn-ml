#!/usr/bin/env python3
"""
Start the E-Commerce Churn Prediction UI

This script activates the virtual environment (if needed) and launches
the FastAPI + Gradio application for making predictions.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("Starting E-Commerce Churn Prediction UI")
    print("=" * 60 + "\n")

    # Get project root
    project_root = Path(__file__).parent
    venv_path = project_root / "venv"

    # Create venv if it doesn't exist
    if not venv_path.exists():
        print("Virtual environment not found. Creating venv...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print("Installing dependencies...")

        # Determine pip executable path
        if sys.platform == "win32":
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:
            pip_exe = venv_path / "bin" / "pip"

        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"],
                      cwd=str(project_root), check=True)

    print("✓ Virtual environment ready\n")

    # Display instructions
    print("=" * 60)
    print("Application is starting...")
    print("=" * 60)
    print("\nThe application will be available at:")
    print("  🌐 Web UI:  http://localhost:8000/ui")
    print("  🔌 API:     http://localhost:8000")
    print("\nPress Ctrl+C to stop the server\n")
    print("=" * 60 + "\n")

    # Run the FastAPI app
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "src.app.main:app",
             "--host", "0.0.0.0", "--port", "8000"],
            cwd=str(project_root)
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


if __name__ == "__main__":
    main()
