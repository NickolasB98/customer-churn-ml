#!/bin/bash

# Start the E-Commerce Churn Prediction UI
# This script activates the virtual environment and launches the FastAPI + Gradio application

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting E-Commerce Churn Prediction UI...${NC}"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Display instructions
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""
echo -e "${BLUE}Launching application...${NC}"
echo ""
echo "The application will be available at:"
echo -e "${GREEN}  Web UI:  http://localhost:8000/ui${NC}"
echo -e "${GREEN}  API:     http://localhost:8000${NC}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the FastAPI app with Gradio UI
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
