# Celery-приложение: где брокер, где результаты, какие есть задачи.
# Этот файл импортируют ОБЕ стороны: и воркер (чтобы знать, что исполнять),
# и клиент (чтобы знать, какие задачи можно отправить).
import os
import socket
import time

from celery import Celery

REDIS = os.getenv("REDIS_URL", "redis://redis:6379")

app = Celery(
    "demo",
    broker=f"{REDIS}/0",   # очередь: сюда клиент кладёт сообщения «выполни задачу»
    backend=f"{REDIS}/1",  # хранилище результатов: сюда воркер пишет ответ
)


@app.task
def add(x, y):
    return x + y


@app.task
def slow_square(x):
    # Имитируем тяжёлую работу, чтобы было видно параллельность воркеров
    time.sleep(3)
    return {"x": x, "square": x * x, "worker": socket.gethostname()}


@app.task
def fail():
    raise ValueError("упали нарочно")
