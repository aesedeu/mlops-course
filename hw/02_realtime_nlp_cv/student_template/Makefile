.DEFAULT_GOAL := help
.PHONY: help variant venv init token up down logs ps restart smoke clean-venv

# Хостовый python и venv для экспорта ONNX (torch ставится ТОЛЬКО сюда, не в систему).
PY    ?= python3
VENV  ?= .venv
VPY   := $(VENV)/bin/python
STAMP := $(VENV)/.installed

help:  ## Показать это сообщение
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# Создать venv и поставить зависимости экспорта. Пересобирается, только если
# изменился requirements-export.txt. torch/torchvision — CPU-сборка (для разовой
# ONNX-конвертации GPU не нужен и качать CUDA-колёса незачем).
$(STAMP): scripts/requirements-export.txt
	@echo ">> создаю venv $(VENV) и ставлю зависимости для экспорта ONNX..."
	@$(PY) -m venv $(VENV) || { echo "!! не удалось создать venv. Установите: sudo apt install -y python3-venv"; exit 1; }
	@$(VPY) -m pip install --upgrade pip -q
	@$(VPY) -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
	@$(VPY) -m pip install -q onnx
	@touch $(STAMP)
	@echo ">> зависимости установлены в $(VENV)"

variant:  ## Узнать свой вариант (CV-модель) по STUDENT_EMAIL из .env
	@$(PY) scripts/variant.py

venv: $(STAMP)  ## Создать venv и поставить зависимости экспорта (torch/torchvision/onnx, CPU)

init: $(STAMP)  ## Экспортировать модель варианта в ONNX (model_repository/<CV_MODEL>/1/model.onnx)
	@$(PY) scripts/variant.py >/dev/null || { $(PY) scripts/variant.py; exit 1; }
	@MODEL=$$(grep -E '^CV_MODEL=' .env | head -n1 | cut -d= -f2-); \
	 $(VPY) scripts/export_model.py --model "$$MODEL"

token:  ## Сгенерировать JWT (нужен JWT_SECRET из .env). Работает на голом python3.
	@if [ ! -f .env ]; then echo "!! нет .env — сделайте: cp .env.example .env"; exit 1; fi; \
	 SECRET=$$(grep -E '^JWT_SECRET=' .env | head -n1 | cut -d= -f2-); \
	 if [ -z "$$SECRET" ]; then echo "!! JWT_SECRET пуст в .env"; exit 1; fi; \
	 JWT_SECRET="$$SECRET" $(PY) scripts/make_token.py

up:  ## Поднять весь стек (triton+vllm+gateway+observability)
	docker compose up -d --build

down:  ## Остановить стек и убрать контейнеры
	docker compose down

logs:  ## Логи всех сервисов (follow)
	docker compose logs -f

ps:  ## Статус сервисов
	docker compose ps

restart:  ## Перезапустить стек
	docker compose restart

smoke:  ## Локальный тест ручек (health/info/vision/chat)
	bash scripts/smoke.sh

clean-venv:  ## Удалить venv экспорта
	rm -rf $(VENV)
