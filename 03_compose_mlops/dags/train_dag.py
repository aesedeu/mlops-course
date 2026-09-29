"""
DAG: train_model — обучение классификатора на датасете Wine.

Что делает один запуск:
  1. Грузит sklearn wine dataset, делает train/test split.
  2. Обучает RandomForestClassifier, считает accuracy и f1 (macro).
  3. Логирует params / metrics / модель (с signature и input_example) в MLflow.
  4. Регистрирует модель в MLflow Model Registry как "wine-clf" и ставит alias "PROD".
     Артефакты модели физически уходят в MinIO (s3://mlflow) через MLflow.

Автор материалов: Чернов Евгений (НИУ ВШЭ).
Запуск — только по триггеру: schedule=None (см. `make trigger-train`).
Весь experiment tracking, метрики и Model Registry — через self-hosted MLflow.
"""
from __future__ import annotations

import logging
from datetime import datetime

from airflow.decorators import dag, task

log = logging.getLogger(__name__)

# Имя модели в MLflow Model Registry и alias «боевой» версии.
MODEL_NAME = "wine-clf"
PROD_ALIAS = "PROD"
MLFLOW_EXPERIMENT = "wine-training"


@dag(
    dag_id="train_model",
    description="Обучение RandomForest на Wine + логирование в MLflow",
    schedule=None,          # запуск только вручную / по триггеру
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "mlflow"],
)
def train_model():
    @task
    def train_and_register() -> dict:
        import os

        import mlflow
        import mlflow.sklearn
        from mlflow.models.signature import infer_signature
        from sklearn.datasets import load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split

        # --- 1. Данные ------------------------------------------------------
        data = load_wine(as_frame=True)
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        log.info("Wine dataset: train=%d, test=%d, признаков=%d",
                 len(X_train), len(X_test), X.shape[1])

        params = {"n_estimators": 200, "max_depth": 6, "random_state": 42}

        # --- 2. Обучение ----------------------------------------------------
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))
        metrics = {"accuracy": acc, "f1_macro": f1}
        log.info("Метрики на тесте: accuracy=%.4f f1_macro=%.4f", acc, f1)

        # --- 3. Логирование в MLflow + регистрация модели -------------------
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="train_rf") as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            # Теги — удобно фильтровать прогоны в MLflow UI.
            mlflow.set_tags({"stage": "training", "dataset": "wine", "model": MODEL_NAME})

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5],
                registered_model_name=MODEL_NAME,   # создаёт/обновляет запись в Registry
            )
            run_id = run.info.run_id
            log.info("MLflow run_id=%s залогирован в эксперимент '%s'",
                     run_id, MLFLOW_EXPERIMENT)

        # --- 4. Проставляем alias PROD последней зарегистрированной версии --
        version = _set_prod_alias(tracking_uri)

        return {"run_id": run_id, "version": version, **metrics}

    train_and_register()


def _set_prod_alias(tracking_uri: str) -> int:
    """Ставит alias PROD на самую свежую версию модели MODEL_NAME."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, PROD_ALIAS, latest.version)
    log.info("alias '%s' установлен на %s версии %s", PROD_ALIAS, MODEL_NAME, latest.version)
    return int(latest.version)


train_model()
