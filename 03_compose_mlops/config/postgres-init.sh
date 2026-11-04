#!/bin/bash
# ============================================================================
#  Инициализация Postgres: вторая БД и пользователь для MLflow.
#  Скрипт монтируется в /docker-entrypoint-initdb.d/ и выполняется официальным
#  образом postgres ОДИН раз — при первой инициализации кластера (пустой volume).
#
#  Основная БД `airflow` и пользователь `airflow` создаются самим образом
#  (POSTGRES_USER/POSTGRES_DB). Здесь добавляем БД `mlflow` и пользователя `mlflow`.
# ============================================================================
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    CREATE DATABASE mlflow OWNER mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
    -- прикладная БД для DAG-ов (db_demo): данные пайплайнов не кладём в метаданные Airflow
    CREATE USER app WITH PASSWORD 'app';
    CREATE DATABASE app OWNER app;
EOSQL

echo ">>> postgres-init: базы mlflow и app и их пользователи созданы."
