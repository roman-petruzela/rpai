#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== 1. Pulling Ollama models ==="
# Pull primary LLM and embedding model for vector memory
ollama pull llama3.1:8b
ollama pull nomic-embed-text

echo "=== 2. Setting up Python environment ==="
# Create and activate virtual environment if requirements.txt exists
if [ -f "requirements.txt" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "=== 3. Starting Docker services ==="
# Launch Open WebUI container in background
docker compose up -d

echo "=== Setup completed successfully! ==="
