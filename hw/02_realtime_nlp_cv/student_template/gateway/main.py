"""Real-time inference gateway — СКЕЛЕТ.

Готовы /health и /metrics. Остальные ручки реализуйте сами по контракту
(API_CONTRACT.md в корне этого каталога):
    /triton/health, /vllm/health, /metrics/triton, /metrics/vllm, /metrics/node,
    /info (Bearer, с полем gpu), /vision (Bearer, модель вашего варианта — make variant),
    /chat (Bearer, SSE).

Как структурировать код (auth, конфиг, инференс) — на ваше усмотрение.
"""
from __future__ import annotations

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx
import jwt
import numpy as np
import tritonclient.http as triton_http
import tritonclient.http.aio as triton_aio
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWTError
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator

Image.MAX_IMAGE_PIXELS = None


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"environment variable {name} is required")
    return value


@dataclass(frozen=True)
class Settings:
    jwt_secret: str
    student_name: str
    student_email: str
    cv_model: str
    llm_model: str
    gpu_name: str
    triton_url: str
    triton_metrics_url: str
    vllm_url: str
    vllm_metrics_url: str
    node_metrics_url: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            jwt_secret=required_env("JWT_SECRET"),
            student_name=required_env("STUDENT_NAME"),
            student_email=required_env("STUDENT_EMAIL"),
            cv_model=required_env("CV_MODEL"),
            llm_model=required_env("LLM_MODEL"),
            gpu_name=required_env("GPU_NAME"),
            triton_url=required_env("TRITON_URL"),
            triton_metrics_url=required_env("TRITON_METRICS_URL"),
            vllm_url=required_env("VLLM_URL").rstrip("/"),
            vllm_metrics_url=required_env("VLLM_METRICS_URL"),
            node_metrics_url=required_env("NODE_METRICS_URL"),
        )


settings = Settings.from_env()
labels = (
    Path(__file__)
    .with_name("imagenet_classes.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)
if len(labels) != 1000:
    raise RuntimeError("imagenet_classes.txt must contain exactly 1000 labels")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=40),
        timeout=httpx.Timeout(120.0, connect=5.0),
    )
    app.state.triton = triton_aio.InferenceServerClient(
        url=settings.triton_url,
        conn_limit=100,
        conn_timeout=30.0,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.triton.close()


app = FastAPI(title="Real-time Inference Gateway", lifespan=lifespan)
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)

bearer = HTTPBearer(auto_error=False)


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid or missing bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized()
    try:
        jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=["HS256"],
            options={"require": ["exp"]},
        )
    except PyJWTError as error:
        raise unauthorized() from error


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info", dependencies=[Depends(require_token)])
def info() -> dict:
    return {
        "student": {"name": settings.student_name, "email": settings.student_email},
        "models": {"vision": settings.cv_model, "llm": settings.llm_model},
        "gpu": settings.gpu_name,
        "version": "1.0",
    }


@app.get("/triton/health")
async def triton_health(request: Request) -> dict:
    try:
        ready = await request.app.state.triton.is_server_ready()
    except Exception:
        ready = False
    if not ready:
        raise HTTPException(status_code=503, detail="Triton is not ready")
    return {"backend": "triton", "ready": True, "model": settings.cv_model}


@app.get("/vllm/health")
async def vllm_health(request: Request) -> dict:
    try:
        response = await request.app.state.http.get(
            f"{settings.vllm_url}/models", timeout=5.0
        )
        response.raise_for_status()
        models = [
            item["id"] for item in response.json().get("data", []) if "id" in item
        ]
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        raise HTTPException(status_code=503, detail="vLLM is not ready") from None
    if settings.llm_model not in models:
        raise HTTPException(status_code=503, detail="Expected vLLM model is not ready")
    return {"backend": "vllm", "ready": True, "models": models}


async def proxy_metrics(request: Request, url: str) -> Response:
    try:
        upstream = await request.app.state.http.get(url, timeout=10.0)
        upstream.raise_for_status()
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502, detail="Metrics backend is unavailable"
        ) from None
    content_type = upstream.headers.get("content-type", "text/plain")
    return Response(content=upstream.content, headers={"content-type": content_type})


@app.get("/metrics/triton")
async def triton_metrics(request: Request) -> Response:
    return await proxy_metrics(request, settings.triton_metrics_url)


@app.get("/metrics/vllm")
async def vllm_metrics(request: Request) -> Response:
    return await proxy_metrics(request, settings.vllm_metrics_url)


@app.get("/metrics/node")
async def node_metrics(request: Request) -> Response:
    return await proxy_metrics(request, settings.node_metrics_url)


PREPROCESS = {
    "resnet50": (256, Image.Resampling.BILINEAR),
    "efficientnet_b0": (256, Image.Resampling.BICUBIC),
    "mobilenet_v3_large": (232, Image.Resampling.BILINEAR),
    "convnext_tiny": (236, Image.Resampling.BILINEAR),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def preprocess_image(data: bytes) -> np.ndarray:
    try:
        resize_size, interpolation = PREPROCESS[settings.cv_model]
    except KeyError as error:
        raise ValueError(f"unsupported CV model: {settings.cv_model}") from error

    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("file is not a valid image") from error

    width, height = image.size
    if width <= height:
        resized = (resize_size, int(resize_size * height / width))
    else:
        resized = (int(resize_size * width / height), resize_size)
    image = image.resize(resized, interpolation)

    left = int(round((image.width - 224) / 2.0))
    top = int(round((image.height - 224) / 2.0))
    image = image.crop((left, top, left + 224, top + 224))

    tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(((tensor - MEAN) / STD)[None, ...], dtype=np.float32)


@app.post("/vision", dependencies=[Depends(require_token)])
async def vision(request: Request, file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Image file is empty")
    try:
        tensor = await asyncio.to_thread(preprocess_image, data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    infer_input = triton_http.InferInput("INPUT__0", list(tensor.shape), "FP32")
    infer_input.set_data_from_numpy(tensor, binary_data=True)
    output = triton_http.InferRequestedOutput("OUTPUT__0", binary_data=True)

    started = time.perf_counter()
    try:
        result = await request.app.state.triton.infer(
            model_name=settings.cv_model,
            inputs=[infer_input],
            outputs=[output],
        )
        logits = result.as_numpy("OUTPUT__0")
        if logits is None or logits.shape != (1, 1000):
            raise ValueError("unexpected Triton output")
    except Exception:
        raise HTTPException(status_code=502, detail="Triton inference failed") from None
    latency_ms = (time.perf_counter() - started) * 1000.0

    logits = logits[0].astype(np.float64)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    top_ids = np.argsort(probabilities)[-5:][::-1]
    predictions = [
        {
            "class_id": int(class_id),
            "label": labels[class_id],
            "score": float(probabilities[class_id]),
        }
        for class_id in top_ids
    ]
    return {
        "model": settings.cv_model,
        "predictions": predictions,
        "latency_ms": latency_ms,
    }


async def stream_upstream(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            if chunk:
                yield chunk
    finally:
        await response.aclose()


@app.post("/chat", dependencies=[Depends(require_token)])
async def chat(request: Request):
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Request body must be valid JSON"
        ) from None
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise HTTPException(status_code=400, detail="messages must be an array")

    payload = dict(payload)
    payload["model"] = settings.llm_model
    payload.setdefault("max_tokens", 128)
    payload.setdefault("temperature", 0.0)
    payload.setdefault("stream", True)
    if not isinstance(payload["stream"], bool):
        raise HTTPException(status_code=400, detail="stream must be a boolean")
    streaming = payload["stream"]

    upstream_request = request.app.state.http.build_request(
        "POST",
        f"{settings.vllm_url}/chat/completions",
        json=payload,
        timeout=120.0,
    )
    try:
        upstream = await request.app.state.http.send(upstream_request, stream=streaming)
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="vLLM is unavailable") from None

    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        code = 400 if upstream.status_code < 500 else 502
        detail = (
            body.decode("utf-8", errors="replace")[:1000] or "vLLM rejected the request"
        )
        raise HTTPException(status_code=code, detail=detail)

    if not streaming:
        try:
            return JSONResponse(content=upstream.json())
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream_upstream(upstream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
