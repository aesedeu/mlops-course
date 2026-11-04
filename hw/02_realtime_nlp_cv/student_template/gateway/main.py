"""Real-time inference gateway — СКЕЛЕТ.

Готовы /health и /metrics. Остальные ручки реализуйте сами по контракту
(API_CONTRACT.md в корне этого каталога):
    /triton/health, /vllm/health, /metrics/triton, /metrics/vllm, /metrics/node,
    /info (Bearer, с полем gpu), /vision (Bearer, модель вашего варианта — make variant),
    /chat (Bearer, SSE).

Как структурировать код (auth, конфиг, инференс) — на ваше усмотрение.
"""
from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Real-time Inference Gateway")

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# TODO: реализуйте остальные ручки контракта (см. API_CONTRACT.md).
