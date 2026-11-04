# 03 — Compose-сборка проекта

**Преподаватель:** Евгений Чернов · **Формат:** 1 лекция / 2 семинара

Сборка MLOps-стека одним `docker-compose`: оркестрация, трекинг экспериментов,
реестр моделей, хранилище артефактов и наблюдаемость.

## Стек

Airflow (**CeleryExecutor**) + **MLflow** + **MinIO** + **Prometheus / Grafana**.
Обучение и инференс ML-модели по триггеру через DAG'и, трекинг экспериментов,
Model Registry с alias `PROD`, артефакты в S3-совместимом MinIO.

## Материалы

- Лекция: [lecture/](lecture/)
- Семинары: [seminar_01.md](seminars/seminar_01.md) · [seminar_02.md](seminars/seminar_02.md)

## Запуск

```bash
cp .env.example .env
make up
make init
make ps
```
