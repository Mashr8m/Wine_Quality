#!/usr/bin/env bash
# Docker rollback demonstration. Requires original UCI CSV and working Docker.
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py --offline --version wine-logreg-1.0.0
docker build -t wine-quality-service:1.0.0 .
python train.py --offline --version wine-logreg-1.1.0 --save-previous
docker build -t wine-quality-service:1.1.0 .

CID=""
cleanup() { if [[ -n "$CID" ]]; then docker rm -f "$CID" >/dev/null 2>&1 || true; fi; }
trap cleanup EXIT
wait_for_health() {
  for attempt in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8077/health; then echo; return 0; fi
    sleep 1
  done
  echo 'Контейнер не стал доступен за 30 попыток' >&2
  return 1
}

CID=$(docker run -d -p 8077:8077 wine-quality-service:1.1.0)
echo 'Сначала текущая версия:'
wait_for_health
docker rm -f "$CID" >/dev/null
CID=""
CID=$(docker run -d -p 8077:8077 wine-quality-service:1.0.0)
echo 'После отката:'
wait_for_health
echo 'Сверьте model_version: wine-logreg-1.0.0'
