# API_CONTRACT — контракт gateway (источник истины для грейдера)

Публичный вход — FastAPI-gateway. `base_url = http://HOST:8080` (порт может быть любым, вы
передаёте полный `base_url` грейдеру). Грейдер проверяет **ровно** эти схемы — соблюдайте
имена полей, коды и формат SSE буквально.

Внутренние порты (docker-сеть, экспонировать не обязательно):
`triton` 8000/8001/8002, `vllm` 8000, `gateway` 8080.
Gateway → Triton: `triton:8000` (http). Gateway → vLLM: `http://vllm:8000/v1`.

## Авторизация

JWT **HS256**. Секрет `JWT_SECRET` — в `.env` студента. Токен генерируется `scripts/make_token.py`
(`make token`) и передаётся грейдеру как непрозрачный Bearer (секрет грейдеру неизвестен).
Сервер валидирует подпись своим секретом.

- Защищённые ручки (`/info`, `/vision`, `/chat`): заголовок `Authorization: Bearer <jwt>`.
- Без токена или с невалидным/просроченным → **401**. Грейдер присылает и поддельные токены:
  корректный по форме JWT, подписанный **чужим** секретом, и неподписанный `alg=none` —
  оба обязаны получить 401 (подпись проверяется, а не только формат).
- С валидным → **200** (или бизнес-код ручки).

---

## `GET /health` — без auth

Быстрая (< 1 c) проверка живости.

```bash
curl -s http://localhost:8080/health
```
```json
{"status": "ok"}
```
Код: `200`.

---

## `GET /metrics` — без auth

Prometheus-метрики gateway в text-exposition формате (`prometheus_fastapi_instrumentator`).

```bash
curl -s http://localhost:8080/metrics
```
```
# HELP http_requests_total ...
http_requests_total{handler="/vision",method="POST",status="2xx"} 12.0
http_request_duration_seconds_bucket{handler="/vision",le="0.4"} 11.0
...
```
Код: `200`, `Content-Type: text/plain`.

---

## `GET /triton/health` — без auth

Готовность бэкенда **Triton**. Gateway опрашивает Triton (`GET /v2/health/ready`).

```bash
curl -s http://localhost:8080/triton/health
```
```json
{"backend": "triton", "ready": true, "model": "efficientnet_b0"}
```
`model` — CV-модель вашего варианта (`CV_MODEL`, см. `make variant`); под этим же именем
модель загружена в Triton.
Коды: `200` — Triton готов; `503` — Triton недоступен/не готов.

---

## `GET /vllm/health` — без auth

Готовность бэкенда **vLLM**. Gateway опрашивает vLLM (`GET /v1/models`).

```bash
curl -s http://localhost:8080/vllm/health
```
```json
{"backend": "vllm", "ready": true, "models": ["Qwen2.5-1.5B-Instruct"]}
```
Коды: `200` — vLLM готов; `503` — vLLM недоступен.

---

## `GET /metrics/triton`, `GET /metrics/vllm`, `GET /metrics/node` — без auth

Проброс сырых Prometheus-метрик соответствующего компонента (gateway ходит в Triton
`:8002/metrics`, vLLM `/metrics`, node-exporter `:9100/metrics`).

```bash
curl -s http://localhost:8080/metrics/node   | head
```
Ответ: `200`, `Content-Type: text/plain` (метрики в exposition-формате).
`502` — если бэкенд недоступен. `/metrics/node` даёт метрики хоста (CPU/RAM/диск/сеть).

Отдавайте **сырые** метрики бэкенда без изменений. Грейдер по ним проверяет, что его
запросы дошли до настоящих Triton и vLLM на GPU:
- `/metrics/triton`: `nv_inference_count{model="<CV_MODEL>"}` растёт на число запросов
  `/vision`, есть `nv_inference_exec_count` (по нему считается dynamic batching) и GPU-метрики
  `nv_gpu_memory_total_bytes`;
- `/metrics/vllm`: `vllm:request_success_total{model_name="<LLM_MODEL>"}` растёт на число
  запросов `/chat`, есть `vllm:kv_cache_usage_perc` (в старых версиях — `vllm:gpu_cache_usage_perc`).

---

## `GET /info` — Bearer

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8080/info
```
```json
{
  "student": {"name": "Иван Иванов", "email": "student@edu.hse.ru"},
  "models": {"vision": "efficientnet_b0", "llm": "Qwen2.5-1.5B-Instruct"},
  "gpu": "NVIDIA A100-SXM4-40GB",
  "version": "1.0"
}
```
- `gpu` — **обязательное** поле: марка/модель/название видеокарты (напр. из `nvidia-smi
  --query-gpu=name`). Грейдер требует непустое значение и пишет его в отчёт.
- `student.email` — ваш **настоящий** email: по нему грейдер определяет вариант.
- `models.vision` — CV-модель вашего варианта (`CV_MODEL`), `models.llm` — served-model-name в vLLM.
- Коды: `200` с токеном; `401` без/с битым токеном.

---

## `POST /vision` — Bearer

Запрос: `multipart/form-data`, поле **`file`** = изображение (jpg/png).

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
     -F "file=@golden_retriever.jpg" \
     http://localhost:8080/vision
```

Ответ `200`:
```json
{
  "model": "efficientnet_b0",
  "predictions": [
    {"class_id": 207, "label": "golden retriever", "score": 0.92},
    {"class_id": 208, "label": "Labrador retriever", "score": 0.03},
    {"class_id": 220, "label": "cocker spaniel", "score": 0.01},
    {"class_id": 222, "label": "kuvasz", "score": 0.008},
    {"class_id": 852, "label": "tennis ball", "score": 0.006}
  ],
  "latency_ms": 12.3
}
```

- `predictions` — **ровно top-5**, отсортированы по `score` по убыванию.
- `score` — вероятность после `softmax` (в диапазоне 0..1).
- `class_id` — индекс ImageNet-1k в **стандартном порядке torchvision** (файл
  `imagenet_classes.txt`, 1000 строк `tench` … `toilet tissue`; строка N при 0-индексации =
  класс N). `label` — метка из того же файла. Проверка: `class_id 207 = golden retriever`.
- `latency_ms` — время инференса в мс (float).
- `model` — CV-модель вашего варианта.
- Грейдер сверяет `score` top-1 с эталоном **именно вашей модели** (допуск ~0.02). Совпадёт
  только при правильном препроцессинге этих весов: размер ресайза, интерполяция, кроп и
  нормализация у моделей разные — берите их из `weights.transforms()` torchvision.
- Принимаются любые картинки, которые открывает Pillow: PNG с прозрачностью, оттенки
  серого, CMYK, очень большие и очень маленькие — ответ `200` с top-5.

Коды ошибок: `401` (нет токена), `400` (пустой файл или не картинка),
`422` (нет поля `file`), `502` (ошибка инференса/Triton недоступен).

---

## `POST /chat` — Bearer

Запрос JSON (OpenAI-подобный):
```json
{
  "messages": [{"role": "user", "content": "Respond with exactly one word: PONG"}],
  "max_tokens": 128,
  "temperature": 0.0,
  "stream": true
}
```

### `stream: true` (обязательно, грейдер использует это)

Ответ: `Content-Type: text/event-stream`, OpenAI-совместимые чанки. Каждый чанк — строка:

```
data: {"choices":[{"delta":{"content":"PO"}}]}

data: {"choices":[{"delta":{"content":"NG"}}]}

data: [DONE]

```

- Каждая SSE-строка данных начинается с `data: ` и завершается пустой строкой (`\n\n`).
- Поток **обязан** завершиться строкой `data: [DONE]\n\n`.
- Контент собирается конкатенацией `choices[0].delta.content` по всем чанкам.
- `temperature = 0` → детерминированный ответ.

```bash
curl -s -N -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Respond with exactly one word: PONG"}],"max_tokens":16,"temperature":0.0,"stream":true}' \
     http://localhost:8080/chat
```

### `stream: false` (поддержать, необязательно для грейдера)

Ответ — обычный JSON vLLM (OpenAI chat.completion):
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}
}
```

Коды ошибок: `401` (нет токена), `400` (нет `messages`, битый JSON или vLLM отклонил
запрос — например, промпт длиннее контекста модели), `502` (vLLM недоступен или вернул
внутреннюю ошибку).

**Ошибку отдавайте HTTP-кодом**, а не `200` с ошибкой внутри стрима. Код можно вернуть,
пока клиенту ещё ничего не отправлено: сначала откройте запрос к vLLM и проверьте его
статус, и только потом начинайте стрим. Если поток оборвался уже посреди ответа — тогда
допустим чанк `data: {"error": {...}}`. Грейдер считает стрим с error-чанком неуспешным.

---

## Требование про `imagenet_classes.txt`

Файл `imagenet_classes.txt` (в `gateway/`) — 1000 меток в стандартном порядке torchvision,
**тот же**, что у грейдера. Не меняйте порядок и не пересобирайте файл: `class_id` в ответе
`/vision` должен соответствовать индексу строки в этом файле. ONNX-модель вашего варианта
экспортируется `scripts/export_model.py` (`make init`) из весов torchvision ImageNet-1k — у всех
вариантов порядок классов совпадает с этим файлом.
