#!/usr/bin/env bash
# Локальный smoke-тест ручек gateway. Требует поднятый стек (make up) и JWT_SECRET в .env.
set -euo pipefail

# Читаем только нужные переменные из .env через grep (БЕЗ сорсинга — значения со
# пробелами, напр. STUDENT_NAME=Иван Иванов, не ломают шелл).
read_env() {
  [ -f .env ] || return 0
  grep -E "^$1=" .env | head -n1 | cut -d= -f2- || true
}

JWT_SECRET="${JWT_SECRET:-$(read_env JWT_SECRET)}"
PORT="${GATEWAY_PORT:-$(read_env GATEWAY_PORT)}"
PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-http://localhost:$PORT}"

if [ -z "$JWT_SECRET" ]; then
  echo "!! JWT_SECRET не задан — заполните .env (cp .env.example .env)"; exit 1
fi

TOKEN="$(JWT_SECRET="$JWT_SECRET" python3 scripts/make_token.py)"

echo "== /health =="
curl -fsS "$BASE_URL/health"; echo

echo "== /info =="
curl -fsS -H "Authorization: Bearer $TOKEN" "$BASE_URL/info"; echo

echo "== /vision (нужен файл sample.jpg рядом) =="
if [ -f sample.jpg ]; then
  curl -fsS -H "Authorization: Bearer $TOKEN" -F "file=@sample.jpg" "$BASE_URL/vision"; echo
else
  echo "  пропущено: положите sample.jpg рядом, чтобы проверить /vision"
fi

echo "== /chat (stream) =="
curl -fsS -N -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":32,"temperature":0.0,"stream":true}' \
  "$BASE_URL/chat"; echo
