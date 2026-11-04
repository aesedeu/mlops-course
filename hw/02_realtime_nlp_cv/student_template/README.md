# Шаблон ДЗ «Real-time NLP/CV инференс»

📄 **Начни с [ЗАДАНИЕ.md](ЗАДАНИЕ.md)** — там постановка, критерии и порядок сдачи.
Контракт ручек — [API_CONTRACT.md](API_CONTRACT.md) (человекочитаемо) и
[openapi.yaml](openapi.yaml) (OpenAPI 3.0 — откроется в Swagger UI / editor.swagger.io),
схема стенда — [architecture.svg](architecture.svg).

Это **скелет**: инфраструктура (`Makefile`, скрипты, `.env.example`, `Dockerfile`,
`imagenet_classes.txt`, `prometheus/`) дана, а gateway, `docker-compose.yml` и
`config.pbtxt` вы пишете сами по контракту. CV-модель у каждого своя — см. `make variant`.

## Быстрый старт

```bash
cp .env.example .env    # задайте JWT_SECRET и данные для /info (настоящий email!)
make variant            # ваша CV-модель → впишите CV_MODEL=... в .env
make init               # venv + экспорт модели → model_repository/<CV_MODEL>/1/model.onnx
make up                 # поднять стек
make token              # Bearer-токен для грейдера
make smoke              # прогнать свои ручки (для /vision положите sample.jpg)
```

Дальше — [ЗАДАНИЕ.md](ЗАДАНИЕ.md).
