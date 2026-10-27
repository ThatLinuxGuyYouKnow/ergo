#!/bin/bash
# ergo EC2 deploy (Amazon Linux 2023). Run ON the instance, from the repo root.
# Installs Docker + compose, builds, starts, and waits for /health.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 installing docker =="
if ! command -v docker >/dev/null; then
  sudo dnf install -y docker docker-compose-plugin
  sudo systemctl enable --now docker
  sudo usermod -aG docker "${USER:-ec2-user}" || true
fi
docker --version
docker compose version

echo "== 2/5 env file =="
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit it for real AWS mode (ERGO_MOCK_*=0 + ERGO_BUCKET)."
fi

echo "== 3/5 data dirs =="
mkdir -p data/jobs data/review data/clips data/s3_mock

echo "== 4/5 build + start =="
docker compose build
docker compose up -d

echo "== 5/5 waiting for /health =="
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT:-8000}/health" >/dev/null; then
    echo "UP: $(curl -s "http://localhost:${PORT:-8000}/health")"
    echo "UI:  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<this-host>'):${PORT:-8000}/ui"
    exit 0
  fi
  sleep 4
done
echo "FAILED to become healthy. Logs:"
docker compose logs --tail=100
exit 1
