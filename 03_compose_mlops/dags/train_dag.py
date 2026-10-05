"""
DAG: train_model — обучение регрессора на синтетических данных.

Что делает один запуск:
  1. Генерирует датасет (по умолчанию 100 000 строк × 30 признаков, см.
     dags/mlops_lib/synth_data.py), делает train/test split.
  2. Обучает модель, выбранную в параметрах запуска (RandomForest или
     HistGradientBoosting), считает RMSE / MAE / R² на тесте.
  3. Логирует params / metrics / модель (с signature и input_example) в MLflow,
     плюс reference_stats.json — статистику признаков и предсказаний на обучении,
     эталон для мониторинга дрейфа при инференсе.
  4. Регистрирует модель в MLflow Model Registry как "synth-regressor" и ставит
     alias "PROD". Артефакты модели физически уходят в MinIO (s3://mlflow).

Параметры запуска (params) задаются без правки кода:
  * в UI: Trigger DAG w/ config — форма с полями;
  * в CLI: airflow dags trigger train_model --conf '{"model": "hist_gb", "n_estimators": 300}'
Не указанные параметры берутся по умолчанию.

Автор материалов: Чернов Евгений (НИУ ВШЭ).
Запуск — только по триггеру: schedule=None (см. `make trigger-train`).
Весь experiment tracking, метрики и Model Registry — через self-hosted MLflow.
"""
from __future__ import annotations

import logging
from datetime import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param

log = logging.getLogger(__name__)

# Имя модели в MLflow Model Registry и alias «боевой» версии.
MODEL_NAME = "synth-regressor"
PROD_ALIAS = "PROD"
MLFLOW_EXPERIMENT = "synth-training"


@dag(
    dag_id="train_model",
    description="Обучение регрессора на синтетике + логирование в MLflow",
    schedule=None,          # запуск только вручную / по триггеру
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training", "mlflow"],
    params={
        "model": Param("random_forest", enum=["random_forest", "hist_gb"],
                       description="random_forest — RandomForestRegressor, "
                                   "hist_gb — HistGradientBoostingRegressor"),
        "n_estimators": Param(100, type="integer", minimum=10, maximum=1000,
                              description="Число деревьев (для hist_gb — итераций бустинга)"),
        "max_depth": Param(8, type="integer", minimum=1, maximum=20,
                           description="Максимальная глубина дерева"),
        "learning_rate": Param(0.1, type="number", minimum=0.001, maximum=1,
                               description="Шаг бустинга (только для hist_gb)"),
        "n_samples": Param(100_000, type="integer", minimum=1_000, maximum=500_000,
                           description="Сколько строк сгенерировать"),
        "noise": Param(3.0, type="number", minimum=0, maximum=20,
                       description="Шум в целевой переменной: больше шум — хуже метрики"),
    },
)
def train_model():
    @task
    def train_and_register(params: dict | None = None) -> dict:
        import os
        import time

        import mlflow
        import mlflow.sklearn
        from mlflow.models.signature import infer_signature
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        from mlops_lib.synth_data import N_INFORMATIVE, make_dataset

        # --- 1. Данные ------------------------------------------------------
        X, y = make_dataset(n_samples=params["n_samples"], noise=params["noise"], seed=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        log.info("Синтетический датасет: train=%d, test=%d, признаков=%d (информативных %d), noise=%.1f",
                 len(X_train), len(X_test), X.shape[1], N_INFORMATIVE, params["noise"])

        # --- 2. Обучение ----------------------------------------------------
        if params["model"] == "random_forest":
            model_params = {"n_estimators": params["n_estimators"],
                            "max_depth": params["max_depth"], "random_state": 42}
            model = RandomForestRegressor(**model_params, n_jobs=-1)
        else:
            model_params = {"max_iter": params["n_estimators"], "max_depth": params["max_depth"],
                            "learning_rate": params["learning_rate"], "random_state": 42}
            model = HistGradientBoostingRegressor(**model_params)
        log.info("Модель: %s %s", params["model"], model_params)

        started = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - started

        preds = model.predict(X_test)
        metrics = {
            "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "fit_seconds": round(fit_seconds, 2),
        }
        log.info("Метрики на тесте: rmse=%.3f mae=%.3f r2=%.4f (обучение %.1f c)",
                 metrics["rmse"], metrics["mae"], metrics["r2"], fit_seconds)

        # --- 3. Логирование в MLflow + регистрация модели -------------------
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        run_name = f"{params['model']}_n{params['n_estimators']}_d{params['max_depth']}"
        with mlflow.start_run(run_name=run_name) as run:
            # Логируем параметры запуска как есть — одинаковый набор у всех
            # прогонов, чтобы их было удобно сравнивать в MLflow (Compare).
            # learning_rate у random_forest не используется.
            mlflow.log_params(dict(params))
            mlflow.log_metrics(metrics)
            # Эталон для мониторинга в проде: как выглядели признаки и предсказания
            # на обучении. Инференс сравнит с ним новый батч — без истинных ответов.
            train_preds = model.predict(X_train)
            mlflow.log_dict({
                "features": {c: {"mean": float(X_train[c].mean()), "std": float(X_train[c].std())}
                             for c in X_train.columns},
                "prediction": {"mean": float(train_preds.mean()), "std": float(train_preds.std())},
            }, "reference_stats.json")
            # Теги — удобно фильтровать прогоны в MLflow UI.
            mlflow.set_tags({"stage": "training", "dataset": "synthetic", "model": MODEL_NAME})

            signature = infer_signature(X_train, model.predict(X_train.iloc[:100]))
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
