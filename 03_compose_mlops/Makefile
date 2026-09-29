# ============================================================================
#  03_compose_mlops — Makefile (единая точка управления стеком)
#  Автор материалов: Чернов Евгений
#
#  Все команды — тонкие обёртки над `docker compose`. Запуск: `make <цель>`.
#  Без аргумента (`make`) печатается справка.
# ============================================================================

.DEFAULT_GOAL := help
.PHONY: help up down logs ps restart init trigger-train trigger-infer

## help: показать список команд
help:
	@echo "MLOps-стек — доступные команды:"
	@echo "  make up             — поднять весь стек в фоне"
	@echo "  make init           — миграции Airflow + admin/admin + бакет MinIO (one-shot)"
	@echo "  make down           — остановить и удалить контейнеры (volume сохраняются)"
	@echo "  make restart        — down + up"
	@echo "  make ps             — статус контейнеров"
	@echo "  make logs           — «хвост» логов всех сервисов (Ctrl+C для выхода)"
	@echo "  make trigger-train  — запустить DAG train_model"
	@echo "  make trigger-infer  — запустить DAG inference"

## init: одноразовая инициализация (БД Airflow, admin/admin, бакет mlflow)
init:
	@echo ">>> Инициализация: миграции Airflow + бакет MinIO. Дождитесь завершения..."
	docker compose up airflow-init minio-init

## up: поднять весь стек в фоновом режиме
up:
	@echo ">>> Поднимаем стек. Первый старт медленный (pip install в Airflow/MLflow)."
	docker compose up -d

## down: остановить и удалить контейнеры (данные в volume остаются)
down:
	@echo ">>> Останавливаем стек. Данные сохраняются (для полной очистки: docker compose down -v)."
	docker compose down

## restart: пересоздать стек
restart: down up

## ps: статус контейнеров
ps:
	docker compose ps

## logs: следить за логами всех сервисов
logs:
	docker compose logs -f --tail=100

## trigger-train: запустить DAG обучения модели
trigger-train:
	@echo ">>> Триггерим DAG train_model..."
	docker compose exec airflow-webserver airflow dags trigger train_model

## trigger-infer: запустить DAG инференса
trigger-infer:
	@echo ">>> Триггерим DAG inference..."
	docker compose exec airflow-webserver airflow dags trigger inference
