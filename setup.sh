#!/bin/bash

echo "=== 1. Zakazuji uspávání notebooku ==="
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

echo "=== 2. Nastavuji Ollama síťový naslech ==="
sudo mkdir -p /etc/systemd/system/ollama.service.d/
echo -e "[Service]\nEnvironment=\"OLLAMA_HOST=0.0.0.0:11434\"" | sudo tee /etc/systemd/system/ollama.service.d/override.conf

sudo systemctl daemon-reload
sudo systemctl restart ollama

echo "=== 3. Stahuji AI modely ==="
ollama pull llama3.1:8b
ollama pull nomic-embed-text

echo "=== 4. Spouštím Docker službu ==="
sudo systemctl enable --now docker

echo "=== Hotovo! Nyní spusťte rozhraní příkazem: docker compose up -d ==="
