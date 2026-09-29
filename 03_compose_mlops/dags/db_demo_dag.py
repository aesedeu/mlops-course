"""
DAG: db_demo — работа с Postgres через Airflow connection и хуки.

Что делает один запуск:
  1. create_table  — SQLExecuteQueryOperator создаёт таблицу (если её нет).
  2. load_samples  — PostgresHook вставляет 20 сэмплов Wine, помечая их run_id.
  3. aggregate     — SQLExecuteQueryOperator считает сводку по классам только
                     для текущего запуска; результат запроса уходит в XCom.
  4. report        — забирает строки из XCom и печатает отчёт.

Connection app_db (задан в docker-compose через AIRFLOW_CONN_APP_DB) указывает
на отдельную БД app. Хук сам берёт из него host/login/password — в коде DAG-а
их нет. Метаданные Airflow (БД airflow) не трогаем.

Автор материалов: Чернов Евгений (НИУ ВШЭ).
Запуск — по триггеру: airflow dags trigger db_demo.
"""
from __future__ import annotations

import logging
from datetime import datetime

from airflow.decorators import dag, task
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

log = logging.getLogger(__name__)

CONN_ID = "app_db"
TABLE = "wine_samples"


@dag(
    dag_id="db_demo",
    description="Connection + hook + SQL-оператор: пишем и читаем Postgres",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["demo", "postgres", "hook"],
)
def db_demo():
    create_table = SQLExecuteQueryOperator(
        task_id="create_table",
        conn_id=CONN_ID,
        sql=f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id              SERIAL PRIMARY KEY,
                run_id          TEXT NOT NULL,
                alcohol         DOUBLE PRECISION,
                color_intensity DOUBLE PRECISION,
                target          INTEGER,
                loaded_at       TIMESTAMPTZ DEFAULT now()
            );
        """,
    )

    @task
    def load_samples(run_id: str | None = None) -> int:
        import numpy as np
        from airflow.providers.postgres.hooks.postgres import PostgresHook
        from sklearn.datasets import load_wine

        # Свой генератор на каждый запуск. Глобальный RNG numpy не подходит:
        # Celery-воркер форкает процессы из одного родителя, у всех одинаковое
        # состояние RNG, и sample() без seed выбирал бы одни и те же строки.
        df = load_wine(as_frame=True).frame.sample(n=20, random_state=np.random.default_rng())
        rows = [
            (run_id, float(r.alcohol), float(r.color_intensity), int(r.target))
            for r in df.itertuples()
        ]

        hook = PostgresHook(postgres_conn_id=CONN_ID)
        hook.insert_rows(
            table=TABLE,
            rows=rows,
            target_fields=["run_id", "alcohol", "color_intensity", "target"],
        )
        total = hook.get_first(f"SELECT count(*) FROM {TABLE}")[0]
        log.info("Вставлено строк: %d, всего в таблице: %d", len(rows), total)
        return len(rows)

    aggregate = SQLExecuteQueryOperator(
        task_id="aggregate",
        conn_id=CONN_ID,
        # Значение подставляет драйвер через параметр, а не склейка строк:
        # так нет SQL-инъекций. {{ run_id }} — шаблон Jinja, Airflow
        # подставит id текущего запуска перед выполнением.
        sql=f"""
            SELECT target,
                   count(*)                             AS n,
                   round(avg(alcohol)::numeric, 2)::float AS avg_alcohol
            FROM {TABLE}
            WHERE run_id = %(run_id)s
            GROUP BY target
            ORDER BY target;
        """,
        parameters={"run_id": "{{ run_id }}"},
    )

    @task
    def report(rows: list) -> None:
        log.info("=" * 48)
        log.info("СВОДКА ПО ЗАПУСКУ (класс | сэмплов | средний alcohol)")
        for target, n, avg_alcohol in rows:
            log.info("  класс %s | %2d | %.2f", target, n, avg_alcohol)
        log.info("=" * 48)

    create_table >> load_samples() >> aggregate
    report(aggregate.output)


db_demo()
