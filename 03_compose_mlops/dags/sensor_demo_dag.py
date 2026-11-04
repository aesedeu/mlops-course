"""
DAG: sensor_demo — ждём файл в S3 (MinIO) и обрабатываем его.

Что делает один запуск:
  1. wait_for_file — S3KeySensor ждёт, пока в бакете incoming появится
     файл по маске new/*.csv. Режим reschedule: между проверками задача
     отдаёт слот воркера (в UI она в статусе up_for_reschedule).
  2. process_files — читает найденные CSV через S3Hook и печатает сводку.
  3. archive_files — переносит файлы в processed/<run_id>/, чтобы следующий
     запуск снова ждал новый файл, а не срабатывал на старый.

Подключение к MinIO — Airflow connection minio_s3 (задан в docker-compose
через AIRFLOW_CONN_MINIO_S3). В коде DAG-а нет ни адресов, ни ключей.

Автор материалов: Чернов Евгений (НИУ ВШЭ).
Запуск — по триггеру, затем кладём файл:
  mc cp data/wine_batch.csv mlops/incoming/new/wine_batch.csv
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

log = logging.getLogger(__name__)

AWS_CONN_ID = "minio_s3"
BUCKET = "incoming"
PREFIX = "new/"


@dag(
    dag_id="sensor_demo",
    description="S3KeySensor: ждём CSV в MinIO, обрабатываем и архивируем",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["demo", "sensor", "s3"],
)
def sensor_demo():
    wait_for_file = S3KeySensor(
        task_id="wait_for_file",
        aws_conn_id=AWS_CONN_ID,
        bucket_name=BUCKET,
        bucket_key=f"{PREFIX}*.csv",
        wildcard_match=True,
        # reschedule: между проверками слот воркера свободен.
        # poke (по умолчанию): задача держит слот всё время ожидания.
        mode="reschedule",
        poke_interval=10,             # проверять каждые 10 с
        timeout=timedelta(minutes=15), # не дождались за 15 мин — задача падает
    )

    @task
    def process_files() -> list[str]:
        import io

        import pandas as pd
        from airflow.providers.amazon.aws.hooks.s3 import S3Hook

        hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        keys = [k for k in hook.list_keys(bucket_name=BUCKET, prefix=PREFIX) if k.endswith(".csv")]
        log.info("Найдено файлов: %d -> %s", len(keys), keys)

        for key in keys:
            df = pd.read_csv(io.StringIO(hook.read_key(key=key, bucket_name=BUCKET)))
            log.info("=" * 48)
            log.info("Файл s3://%s/%s: строк=%d, колонки=%s", BUCKET, key, len(df), list(df.columns))
            log.info("Средние значения:\n%s", df.mean(numeric_only=True).round(2).to_string())
            log.info("=" * 48)
        return keys  # список ключей уходит в XCom и достаётся следующей задаче

    @task
    def archive_files(keys: list[str], run_id: str | None = None) -> None:
        from airflow.providers.amazon.aws.hooks.s3 import S3Hook

        hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        for key in keys:
            dest = f"processed/{run_id}/{key.removeprefix(PREFIX)}"
            hook.copy_object(source_bucket_key=key, dest_bucket_key=dest,
                             source_bucket_name=BUCKET, dest_bucket_name=BUCKET)
            log.info("s3://%s/%s -> s3://%s/%s", BUCKET, key, BUCKET, dest)
        if keys:
            hook.delete_objects(bucket=BUCKET, keys=keys)

    keys = process_files()
    wait_for_file >> keys
    archive_files(keys)


sensor_demo()
