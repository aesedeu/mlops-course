"""
DAG: inference — батч-инференс «боевой» модели из MLflow Model Registry.

Что делает один запуск:
  1. Тянет модель по alias:  models:/synth-regressor@PROD  из MLflow (и её версию).
  2. Получает новый батч данных (здесь генерируем тем же генератором, что и при
     обучении, с новым seed). Истинных ответов у нас НЕТ — как и в проде:
     они приходят позже или не приходят вовсе. Поэтому RMSE/R² здесь не считаем.
  3. Предсказывает и сохраняет предсказания — результат батча.
  4. Мониторинг без меток: сравнивает батч с эталоном reference_stats.json,
     который train_model сохранил в run PROD-версии:
       * дрейф признаков — насколько сдвинулось среднее каждого признака
         (в единицах std обучающей выборки);
       * дрейф предсказаний — сдвиг среднего и изменение разброса.
  5. Логирует всё это в MLflow — отдельный run в эксперименте "inference".

Параметр shift сдвигает признаки относительно обучения — имитация дрейфа данных:
  airflow dags trigger inference --conf '{"shift": 1.5}'

Автор материалов: Чернов Евгений (НИУ ВШЭ).
Запуск — только по триггеру: schedule=None (см. `make trigger-infer`).
Предусловие: DAG train_model уже отработал и alias PROD проставлен.
Весь трекинг инференса — через self-hosted MLflow.
"""
from __future__ import annotations

import logging
from datetime import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param

log = logging.getLogger(__name__)

MODEL_NAME = "synth-regressor"
PROD_ALIAS = "PROD"
MODEL_URI = f"models:/{MODEL_NAME}@{PROD_ALIAS}"
INFERENCE_EXPERIMENT = "inference"
# Признак считаем «уехавшим», если его среднее сдвинулось больше чем на 0.25 std.
DRIFT_THRESHOLD = 0.25


@dag(
    dag_id="inference",
    description="Батч-инференс synth-regressor@PROD + мониторинг дрейфа",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "inference", "mlflow"],
    params={
        "n_samples": Param(5_000, type="integer", minimum=100, maximum=10_000_000,
                           description="Размер батча для инференса"),
        "shift": Param(0.0, type="number", minimum=-5, maximum=5,
                       description="Сдвиг признаков относительно обучения (0 — без дрейфа)"),
    },
)
def inference():
    @task
    def run_inference(params: dict | None = None) -> dict:
        import os

        import mlflow
        import numpy as np
        import pandas as pd
        from mlflow import MlflowClient

        from mlops_lib.synth_data import make_dataset

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # --- 1. Узнаём версию PROD и загружаем боевую модель по alias -------
        client = MlflowClient(tracking_uri=tracking_uri)
        prod = client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
        log.info("Загружаем модель из MLflow: %s (версия %s)", MODEL_URI, prod.version)
        model = mlflow.pyfunc.load_model(MODEL_URI)
        # Эталон лежит в том же run, где обучали PROD-версию. У старых версий
        # его может не быть — тогда предсказываем, но дрейф не проверяем.
        try:
            ref = mlflow.artifacts.load_dict(f"runs:/{prod.run_id}/reference_stats.json")
        except Exception:
            ref = None
            log.warning("У версии %s нет reference_stats.json — мониторинг дрейфа пропущен. "
                        "Переобучите модель через train_model.", prod.version)

        # --- 2. Новый батч. y генератора выбрасываем: в проде меток нет -----
        X, _ = make_dataset(n_samples=params["n_samples"], shift=params["shift"], seed=None)

        # --- 3. Предсказания -----------------------------------------------
        preds = np.asarray(model.predict(X), dtype=float)

        # --- 4. Мониторинг без меток ---------------------------------------
        metrics = {
            "samples_total": len(preds),
            "pred_mean": float(preds.mean()),
            "pred_std": float(preds.std()),
            "pred_p05": float(np.percentile(preds, 5)),
            "pred_p95": float(np.percentile(preds, 95)),
        }
        drifted = []
        if ref is not None:
            feature_drift = {
                c: abs(X[c].mean() - s["mean"]) / s["std"] for c, s in ref["features"].items()
            }
            drifted = sorted(c for c, d in feature_drift.items() if d > DRIFT_THRESHOLD)
            ref_pred = ref["prediction"]
            metrics.update({
                # сдвиг среднего предсказания в единицах std предсказаний на обучении
                "pred_mean_shift": float(abs(preds.mean() - ref_pred["mean"]) / ref_pred["std"]),
                "pred_std_ratio": float(preds.std() / ref_pred["std"]),
                "feature_drift_max": float(max(feature_drift.values())),
                "feature_drift_mean": float(np.mean(list(feature_drift.values()))),
                "drifted_features": len(drifted),
            })

        log.info("=" * 56)
        log.info("ОТЧЁТ ИНФЕРЕНСА (модель %s, версия %s)", MODEL_URI, prod.version)
        log.info("Батч: %d строк", metrics["samples_total"])
        if ref is None:
            log.info("Предсказания: mean=%.2f, std=%.2f, p05..p95=%.1f..%.1f (эталона нет)",
                     metrics["pred_mean"], metrics["pred_std"], metrics["pred_p05"], metrics["pred_p95"])
        else:
            log.info("Предсказания: mean=%.2f (на обучении %.2f), std=%.2f (на обучении %.2f), p05..p95=%.1f..%.1f",
                     metrics["pred_mean"], ref_pred["mean"], metrics["pred_std"], ref_pred["std"],
                     metrics["pred_p05"], metrics["pred_p95"])
            log.info("Дрейф признаков: max=%.2f std, в среднем %.2f std",
                     metrics["feature_drift_max"], metrics["feature_drift_mean"])
            if drifted:
                log.warning("ДРЕЙФ: %d из %d признаков сдвинулись больше чем на %.2f std: %s",
                            len(drifted), len(feature_drift), DRIFT_THRESHOLD, ", ".join(drifted))
            else:
                log.info("Дрейфа нет: все признаки в пределах %.2f std от обучения", DRIFT_THRESHOLD)
        log.info("=" * 56)

        # --- 5. Логируем прогон в MLflow ------------------------------------
        mlflow.set_experiment(INFERENCE_EXPERIMENT)
        with mlflow.start_run(run_name="batch_inference"):
            mlflow.log_params({
                "model_uri": MODEL_URI,
                "model_version": prod.version,
                "n_samples": params["n_samples"],
                "shift": params["shift"],
            })
            mlflow.set_tags({"stage": "inference", "model": MODEL_NAME, "alias": PROD_ALIAS,
                             "drift": "unknown" if ref is None else ("yes" if drifted else "no")})
            mlflow.log_metrics(metrics)
            # Результат батча — сами предсказания.
            mlflow.log_table(pd.DataFrame({"prediction": preds}), "predictions.json")
        log.info("Предсказания и метрики мониторинга залогированы в MLflow (эксперимент '%s').",
                 INFERENCE_EXPERIMENT)

        return {"model_version": prod.version, **metrics}

    run_inference()


inference()
