FROM python:3.11-slim
RUN pip install --no-cache-dir "celery[redis]==5.4.0"
WORKDIR /app
COPY tasks.py client.py ./
