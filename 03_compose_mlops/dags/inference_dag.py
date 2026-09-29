"""
DAG: inference — батч-инференс «боевой» модели из MLflow Model Registry.

Что делает один запуск:
  1. Тянет модель по alias:  models:/wine-clf@PROD  из MLflow (и её версию).
  2. Прогоняет классификацию на нескольких сэмплах из датасета Wine.
  3. Печатает читаемый отчёт (по классам, доля предсказаний) в лог Airflow.
  4. Логирует метрики/параметры прогона в MLflow — отдельный run в эксперименте
     "inference" (число сэмплов, распределение предсказаний, версия PROD-модели).

Автор материалов: Чернов Евгений (НИУ ВШЭ).
Запуск — только по триггеру: schedule=None (см. `make trigger-infer`).
Предусловие: DAG train_model уже отработал и alias PROD проставлен.
Весь трекинг инференса — через self-hosted MLflow.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime

from airflow.decorators import dag, task

log = logging.getLogger(__name__)

MODEL_NAME = "wine-clf"
PROD_ALIAS = "PROD"
MODEL_URI = f"models:/{MODEL_NAME}@{PROD_ALIAS}"
INFERENCE_EXPERIMENT = "inference"


@dag(
    dag_id="inference",
    description="Батч-инференс wine-clf@PROD из MLflow Model Registry",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "inference", "mlflow"],
)
def inference():
    @task
    def run_inference() -> dict:
        import os

        import mlflow
        import numpy as np
        from mlflow import MlflowClient
        from sklearn.datasets import load_wine

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # --- 1. Узнаём версию PROD и загружаем боевую модель по alias -------
        client = MlflowClient(tracking_uri=tracking_uri)
        prod_version = client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS).version
        log.info("Загружаем модель из MLflow: %s (версия %s)", MODEL_URI, prod_version)
        model = mlflow.pyfunc.load_model(MODEL_URI)

        # --- 2. Берём несколько сэмплов для классификации ------------------
        data = load_wine(as_frame=True)
        X = data.data
        sample = X.sample(n=20, random_state=7)
        preds = model.predict(sample)
        preds = np.asarray(preds).astype(int).tolist()

        # --- 3. Формируем отчёт и печатаем в лог Airflow -------------------
        target_names = list(data.target_names)
        counts = Counter(preds)
        total = len(preds)

        log.info("=" * 48)
        log.info("ОТЧЁТ ИНФЕРЕНСА (модель %s, версия %s)", MODEL_URI, prod_version)
        log.info("Всего сэмплов: %d", total)
        for cls_idx in sorted(counts):
            name = target_names[cls_idx] if cls_idx < len(target_names) else str(cls_idx)
            n = counts[cls_idx]
            log.info("  класс %d (%s): %d предсказаний (%.1f%%)",
                     cls_idx, name, n, 100.0 * n / total)
        log.info("=" * 48)

        distribution = {str(k): counts[k] for k in sorted(counts)}

        # --- 4. Логируем метрики/параметры прогона в MLflow ----------------
        mlflow.set_experiment(INFERENCE_EXPERIMENT)
        with mlflow.start_run(run_name="batch_inference"):
            mlflow.log_params({
                "model_uri": MODEL_URI,
                "model_version": prod_version,
                "n_samples": total,
            })
            mlflow.set_tags({"stage": "inference", "model": MODEL_NAME, "alias": PROD_ALIAS})
            mlflow.log_metric("samples_total", total)
            # Распределение предсказаний по классам как отдельные метрики.
            for cls, n in distribution.items():
                mlflow.log_metric(f"class_{cls}_count", n)
        log.info("Метрики инференса залогированы в MLflow (эксперимент '%s').",
                 INFERENCE_EXPERIMENT)

        return {"total": total, "distribution": distribution, "model_version": prod_version}

    run_inference()


inference()
