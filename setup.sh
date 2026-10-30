#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OLLAMA_HOST
CHAT_MODEL="llama3.1:8b"
EMBEDDING_MODEL="nomic-embed-text"

trap 'echo "[ERROR] Setup failed near line ${LINENO}." >&2' ERR

log() {
    printf '\n[SETUP] %s\n' "$1"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: $1" >&2
        exit 1
    fi
}

log "Checking required tools..."
require_command cargo
require_command rustc
require_command curl
require_command ollama

log "Checking Ollama availability..."
if ! curl --silent --show-error --fail "${OLLAMA_HOST%/}/api/tags" >/dev/null; then
    echo "[ERROR] Ollama is not reachable at ${OLLAMA_HOST}." >&2
    echo "[ERROR] Start Ollama first, for example with: ollama serve" >&2
    exit 1
fi

log "Pulling chat model: ${CHAT_MODEL}"
ollama pull "$CHAT_MODEL"

log "Pulling embedding model: ${EMBEDDING_MODEL}"
ollama pull "$EMBEDDING_MODEL"

log "Checking Rust project..."
cargo check

log "Building JARVIS..."
cargo build

if [[ "${SETUP_PYTHON:-false}" == "true" ]]; then
    log "Setting up optional Python environment..."
    require_command python3
    if [[ -f requirements.txt ]]; then
        python3 -m venv .venv
        .venv/bin/python -m pip install --upgrade pip
        .venv/bin/python -m pip install -r requirements.txt
    else
        echo "[PYTHON] requirements.txt not found; skipping."
    fi
fi

if [[ "${START_DOCKER_SERVICES:-false}" == "true" ]]; then
    log "Starting optional Docker services..."
    require_command docker
    docker compose up -d
fi

cat <<'BANNER'

==================================================
                    J.A.R.V.I.S.
==================================================
[SETUP] Development environment is ready.
[SETUP] Start the application with: cargo run
==================================================
BANNER
