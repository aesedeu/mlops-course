# Семинар 1. Собираем MLOps-стек одним `docker compose`

> Практический воркшоп к блоку «Компоуз-сборка MLOps-проекта».
> Автор материалов: Чернов Евгений (НИУ ВШЭ, магистратура «Искусственный интеллект»).
> Длительность: 2 академических часа.

Сегодня мы поднимаем **весь MLOps-стенд** одной командой и прогоняем через него
первый цикл обучения модели. К концу занятия у вас на ноутбуке будет работать:
оркестратор пайплайнов (**Airflow** на **CeleryExecutor**), трекинг экспериментов
и Model Registry (**self-hosted MLflow**), объектное хранилище артефактов
(**MinIO**) и заготовка мониторинга (**Prometheus/Grafana**). А главное — вы своими
руками обучите модель в DAG-е, увидите её параметры и метрики в MLflow, и сохраните
артефакты в S3-бакет с алиасом `PROD`.

Идея занятия простая: **MLOps — это не один инструмент, а связка сервисов**,
каждый из которых отвечает за свою часть жизненного цикла модели. `docker compose`
даёт нам собрать эту связку локально, чтобы потрогать её целиком.

---

## Что понадобится

- **Docker** и **Docker Compose v2** (команда `docker compose`, через пробел, а не
  старый `docker-compose`). Проверить:
  ```bash
  docker --version
  docker compose version
  ```
  Ожидаемый вывод — что-то вроде `Docker version 24.x` и `Docker Compose version v2.x`.
- Свободные **4+ ГБ RAM** под Docker (в стеке ~13 контейнеров). На Docker Desktop:
  Settings → Resources → Memory ≥ 6 GB.
- Свободные порты на хосте: `5432, 6379, 5000, 8080, 5555, 9000, 9001, 9090, 3000,
  9100, 9187, 9102`. Если какой-то занят — увидим это в шаге проверки.
  На macOS порт `5000` почти всегда занят системным AirPlay Receiver — см. «Частые ошибки».
- `make` (на macOS ставится с Command Line Tools, на Linux — пакет `make`).
- Рабочая директория курса: `03_compose_mlops` со всеми созданными файлами.

> **На что обратить внимание.** Весь experiment tracking, метрики и Model Registry
> в этом курсе — через **self-hosted MLflow**, который мы поднимаем прямо в нашем
> `docker compose`. Никаких внешних SaaS-трекеров и облачных аккаунтов не требуется:
> и метаданные экспериментов, и артефакты остаются у нас на стенде (Postgres +
> MinIO).

---

## Шаг 0. Осматриваемся в директории

Прежде чем что-то запускать — посмотрим, из чего состоит проект. В MLOps «прочитать
compose-файл» это то же, что для инженера «прочитать схему установки».

```bash
cd 03_compose_mlops
ls -1A
```
Ожидаемый вывод:
```
.env.example
Makefile
README.md
celery_demo
config
dags
data
docker-compose.yml
grafana
lecture
logs
plugins
prometheus
seminars
```

Разложим по полочкам, что за что отвечает:

- `docker-compose.yml` — описание всех сервисов стека (сердце проекта).
- `.env.example` — шаблон переменных окружения (пароли, ключи, UID).
- `Makefile` — короткие команды-обёртки (`make up`, `make init`, ...).
- `dags/` — пайплайны Airflow: `train_dag.py`, `inference_dag.py` (сегодня) и
  демо `db_demo_dag.py`, `sensor_demo_dag.py` (Семинар 2).
- `data/` — пример входного файла для `sensor_demo`.
- `celery_demo/` — игрушечный Celery отдельно от Airflow (см. его README).
- `config/` — init-скрипт Postgres и mapping для statsd-exporter.
- `prometheus/` — конфиг сбора метрик.
- `grafana/` — provisioning (datasource + dashboards) и сами JSON-дашборды.
- `logs/`, `plugins/` — тома, которые Airflow монтирует внутрь себя.

---

## Шаг 1. Разбираем `docker-compose.yml` — сервис за сервисом

Открываем `docker-compose.yml`. Не пугайтесь объёма: 90% строк — это повторяющиеся
шаблоны. Разберём по группам, отвечая на главный вопрос: **зачем этот сервис нужен**.

### 1.1. Общий блок Airflow — YAML anchor

В начале файла — не сервис, а **шаблон**:
```yaml
x-airflow-common: &airflow-common
  image: apache/airflow:2.9.3-python3.11
  environment: &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    ...
  volumes:
    - ./dags:/opt/airflow/dags
    ...
```
У Airflow пять контейнеров (webserver, scheduler, worker, flower, init), и всем им
нужны **одинаковые** env и тома. Копировать это пять раз — путь к ошибкам. YAML
даёт механизм **anchor** (`&airflow-common`) и **alias** (`*airflow-common`): один
раз описали — многократно подмешали через `<<: *airflow-common`.

> **На что обратить внимание.** Строка `_PIP_ADDITIONAL_REQUIREMENTS` доставляет
> `mlflow`, `scikit-learn`, `boto3`, `pandas` внутрь контейнера **при старте**. Это
> удобно для семинара, но означает медленный первый запуск (идёт `pip install`).
> В проде так не делают — там пекут собственный образ заранее.

### 1.2. Хранилища состояния: `postgres` и `redis`

```yaml
postgres:
  image: postgres:15
  environment:
    POSTGRES_USER: airflow
    POSTGRES_PASSWORD: airflow
    POSTGRES_DB: airflow
  volumes:
    - postgres-data:/var/lib/postgresql/data
    - ./config/postgres-init.sh:/docker-entrypoint-initdb.d/10-init-mlflow-db.sh:ro
```
- **Postgres** — общий сервер баз данных: метаданные Airflow (БД `airflow`),
  backend-store MLflow (БД `mlflow`) и прикладная БД `app` для демо-DAG-а `db_demo`.
  Базы `mlflow`, `app` и их пользователи создаются init-скриптом `config/postgres-init.sh`, который монтируется в
  `/docker-entrypoint-initdb.d/` — официальный образ Postgres прогоняет всё оттуда
  при **первой** инициализации (пустой volume).
- **Redis** — брокер сообщений для Celery: scheduler кладёт задачи в очередь, worker
  их забирает. Он же result backend по спецификации через `db+postgresql`.

Обратите внимание на `healthcheck` у обоих: Docker проверяет `pg_isready` и
`redis-cli ping`. Это позволяет другим сервисам ждать не «пока контейнер
запустился», а «пока сервис реально готов принимать соединения» (`condition:
service_healthy`).

### 1.3. Хранилище артефактов: `minio` и `minio-init`

```yaml
minio:
  image: minio/minio:...
  command: server /data --console-address ":9001"
  ports:
    - "9000:9000"   # S3 API
    - "9001:9001"   # Web-консоль
```
**MinIO** — это локальный S3. MLflow и Airflow ходят в него по протоколу Amazon S3
(отсюда переменные `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` = `minioadmin`).
Порт `9000` — API (по нему говорят программы), `9001` — веб-консоль (в неё смотрим
мы). `minio-init` — одноразовый контейнер: утилита `mc` создаёт бакеты `mlflow`
(сюда MLflow складывает модели), `airflow-logs` (логи задач Airflow) и `incoming`
(входящие файлы для демо-сенсора на Семинаре 2).

### 1.4. Трекинг-сервер: `mlflow`

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.14.1
  command: >
    ... mlflow server --host 0.0.0.0 --port 5000
    --backend-store-uri postgresql://mlflow:mlflow@postgres/mlflow
    --artifacts-destination s3://mlflow --serve-artifacts
```
MLflow разделяет две вещи:
- **backend-store** (метаданные экспериментов, параметры, метрики) — в Postgres.
- **artifact store** (файлы модели, графики) — в MinIO (`s3://mlflow`).

Флаг `--serve-artifacts` заставляет сам MLflow проксировать доступ к артефактам —
клиентам не нужно напрямую знать про MinIO. Официальный образ не содержит
`psycopg2` и `boto3`, поэтому в `command` мы их доустанавливаем перед запуском.

### 1.5. Airflow: пять контейнеров

- `airflow-init` — **one-shot**: прогоняет `airflow db migrate` и создаёт
  пользователя `admin/admin`. Отработал — завершился.
- `airflow-webserver` — UI на `:8080` (логин `admin`/`admin`).
- `airflow-scheduler` — «мозг»: решает, какие задачи и когда запускать.
- `airflow-worker` — Celery-воркер, реально выполняет код задач.
- `airflow-flower` — мониторинг Celery на `:5555`.

Все они подмешивают `*airflow-common` и ждут готовности Postgres/Redis, а рабочие
сервисы ещё и ждут завершения `airflow-init` через
`condition: service_completed_successfully`.

### 1.6. Мониторинг: `prometheus`, `grafana` и экспортёры

- `prometheus` (`:9090`) — собирает метрики, опрашивая экспортёры.
- `grafana` (`:3000`, `admin`/`admin`) — рисует дашборды.
- `node-exporter` (`:9100`) — метрики хоста (CPU/RAM/диск).
- `postgres-exporter` (`:9187`) — метрики Postgres.
- `statsd-exporter` (`:9102` метрики / `:9125/udp` приём) — принимает statsd-метрики
  Airflow и отдаёт их Prometheus.

Мониторинг мы подробно настраиваем на Семинаре 2 — сегодня просто поднимаем его
вместе со стеком.

> **На что обратить внимание.** В самом конце файла — блок `volumes:` с
> `postgres-data`, `minio-data`, `grafana-data`, `prometheus-data`. Это
> **именованные тома**: они переживают `make down` и хранят ваши данные. Полностью
> стирает их только `docker compose down -v`.

---

## Шаг 2. Готовим `.env` и поднимаем стек

Compose читает переменные из файла `.env`. Создадим его из шаблона:
```bash
cp .env.example .env
```
Загляните внутрь `.env`. Пока можно ничего не менять: `minioadmin/minioadmin`,
`admin/admin` — учебные значения для локального стенда.

> **На что обратить внимание (Linux).** Переменная `AIRFLOW_UID` должна совпадать с
> вашим UID, иначе файлы в `./logs` будут принадлежать root. Узнать своё значение:
> `id -u`, и вписать в `.env`. На macOS/Windows можно оставить `50000`.

Теперь **инициализация** — одноразовая: миграции БД Airflow, создание админа и
бакета MinIO.
```bash
make init
```
Что происходит внутри (`make init` = `docker compose up airflow-init minio-init`):
скачиваются образы, Postgres создаёт БД `airflow`, `mlflow` и `app`, `airflow-init`
накатывает схему и создаёт `admin/admin`, `minio-init` создаёт бакеты.

Ожидаемый вывод (хвост):
```
mlops-airflow-init  | User "admin" created with role "Admin"
mlops-airflow-init exited with code 0
mlops-minio-init    | >>> buckets mlflow, airflow-logs, incoming ready
mlops-minio-init exited with code 0
```
Оба контейнера завершились с кодом 0 — инициализация прошла. Это нормально, что они
«погасли»: это одноразовые задачи.

Поднимаем весь стек в фоне:
```bash
make up
```
Ожидаемый вывод — список создаваемых контейнеров со статусом `Started`/`Healthy`.
Первый запуск идёт минуты: Airflow и MLflow ставят pip-зависимости.

> **На что обратить внимание.** Не пугайтесь, если `airflow-worker` какое-то время
> в статусе `health: starting`. Ему нужно доустановить `mlflow`, `scikit-learn`,
> `boto3` — мы дали ему `start_period: 90s`.

---

## Шаг 3. Проверяем здоровье стека

Смотрим статус:
```bash
make ps
```
Ожидаемый вывод (сокращённо) — все долгоживущие сервисы в состоянии `Up` /
`healthy`:
```
NAME                        STATUS
mlops-postgres              Up (healthy)
mlops-redis                 Up (healthy)
mlops-minio                 Up (healthy)
mlops-mlflow                Up (healthy)
mlops-airflow-webserver     Up (healthy)
mlops-airflow-scheduler     Up (healthy)
mlops-airflow-worker        Up (healthy)
mlops-airflow-flower        Up (healthy)
mlops-prometheus            Up
mlops-grafana               Up
...
```
`airflow-init` и `minio-init` в списке `Exited (0)` — так и должно быть.

Теперь пройдёмся по UI (откройте в браузере):

| Сервис        | URL                     | Логин         |
|---------------|-------------------------|---------------|
| Airflow       | http://localhost:8080   | admin / admin |
| MLflow        | http://localhost:5000   | —             |
| MinIO консоль | http://localhost:9001   | minioadmin / minioadmin |
| Flower        | http://localhost:5555   | —             |
| Grafana       | http://localhost:3000   | admin / admin |
| Prometheus    | http://localhost:9090   | —             |

**Что вы увидите:**
- **Airflow (8080):** список DAG-ов: `train_model`, `inference`, `db_demo`,
  `sensor_demo` (все на паузе — переключатель слева выключен). Сегодня работаем с
  `train_model`, демо-DAG-и разберём на Семинаре 2. Если списка нет — подождите, пока
  scheduler их подхватит (10–30 сек), и обновите страницу.
- **MLflow (5000):** пустой список экспериментов (пока ничего не обучали).
- **MinIO (9001):** в разделе Buckets видны бакеты `mlflow`, `airflow-logs` и
  `incoming`, пока пустые.

Если какой-то UI не открывается — смотрим логи конкретного сервиса:
```bash
docker compose logs -f airflow-webserver
```
`Ctrl+C` — выйти из просмотра (сервис при этом не останавливается).

### Проверяем CeleryExecutor через Flower

Открываем **http://localhost:5555**. Flower — это дашборд Celery. Что смотрим:
- Вкладка **Workers**: должен быть виден один воркер (имя вида `celery@<hostname>`)
  в статусе **Online**.
- Столбцы **Active / Processed**: пока нули — задач не было.

> **На что обратить внимание.** Именно связка scheduler → Redis (брокер) → worker и
> есть CeleryExecutor. Убедиться, что executor действительно Celery, можно так:
> ```bash
> docker compose exec airflow-webserver airflow config get-value core executor
> ```
> Ожидаемый вывод: `CeleryExecutor`. Если увидели `SequentialExecutor` —
> переменная `AIRFLOW__CORE__EXECUTOR` не применилась (проверьте `.env` и
> перезапустите `make restart`).

---

## Шаг 4. Первый DAG — запускаем `train_model`

Открываем `dags/train_dag.py`. Это TaskFlow-DAG (декораторы `@dag`/`@task`,
современный стиль Airflow 2.9). Разберём его логику по шагам — она ровно повторяет
жизненный цикл обучения:

1. Грузим датасет **Wine** из scikit-learn, делаем train/test split.
2. Обучаем **RandomForestClassifier**, считаем `accuracy` и `f1_macro`.
3. Логируем params/metrics/теги/модель в **MLflow** (с signature и input_example).
4. Регистрируем модель в **MLflow Model Registry** под именем `wine-clf` и ставим
   alias **`PROD`**. Файлы модели физически уходят в MinIO.

> **На что обратить внимание.** DAG объявлен с `schedule=None` — он не крутится по
> расписанию, а запускается только руками/по триггеру. Для обучения это правильно:
> мы не хотим, чтобы модель переобучалась «каждые 5 минут» без причины.

Прежде чем триггерить — снимаем DAG с паузы. Либо переключателем в UI Airflow
слева от имени `train_model`, либо командой:
```bash
docker compose exec airflow-webserver airflow dags unpause train_model
```

Теперь запускаем:
```bash
make trigger-train
```
Ожидаемый вывод — таблица с новым запуском в состоянии `queued`:
```
>>> Триггерим DAG train_model...
conf | dag_id      | dag_run_id                | ... | state
=====+=============+===========================+=====+=======
{}   | train_model | manual__2026-...+00:00    | ... | queued
```

**Что вы увидите в UI Airflow (8080):** зайдите в DAG `train_model` → вкладка
**Grid**. Появится новый запуск: сначала жёлтый (`running`), затем зелёный
(`success`). Кликните на квадрат задачи → **Logs**, чтобы читать лог выполнения.

Ожидаемые строки в логе задачи:
```
Wine dataset: train=133, test=45, признаков=13
Метрики на тесте: accuracy=1.0000 f1_macro=1.0000
MLflow run_id=... залогирован в эксперимент 'wine-training'
alias 'PROD' установлен на wine-clf версии 1
```

> **На что обратить внимание.** Строка `alias 'PROD' установлен на wine-clf версии 1`
> — это и есть автоматическая «выкатка»: DAG зарегистрировал модель и сразу пометил
> свежую версию алиасом `PROD`. По этому алиасу её потом заберёт DAG инференса.

Пока задача выполняется, вернитесь во **Flower (5555)**: на вкладке Workers у
воркера вырастут счётчики Active/Processed — вы видите, как CeleryExecutor реально
исполняет задачу.

---

## Шаг 5. Смотрим эксперимент в MLflow и артефакты в MinIO

Открываем **MLflow (http://localhost:5000)**:
- В списке слева появился эксперимент **`wine-training`**. Заходим — видим один run
  `train_rf`.
- Внутри run-а: **Parameters** (`n_estimators=200`, `max_depth=6`, ...),
  **Metrics** (`accuracy`, `f1_macro`), а во вкладке **Artifacts** — папка `model`
  с файлами (`MLmodel`, `model.pkl`, `conda.yaml`, ...).
- Вкладка **Models** (вверху): там модель **`wine-clf`**, версия 1, с алиасом
  **`PROD`**.

Теперь проверим, что артефакты **физически** легли в S3. Открываем **MinIO консоль
(http://localhost:9001)** → Buckets → `mlflow`. Внутри увидите структуру вида
`<experiment_id>/<run_id>/artifacts/model/...` (например `1/a8d1.../artifacts/model/model.pkl`)
— это и есть ваша модель в объектном хранилище.

> **На что обратить внимание.** Разделение «метаданные в Postgres / файлы в S3» —
> ключевая идея MLflow. В Registry лежит только *ссылка* на артефакт; сам файл — в
> MinIO. Так же устроены облачные трекинг-серверы, только вместо MinIO там реальный
> S3/GCS.

---

## Шаг 6. Разбираем эксперимент в MLflow UI детально

MLflow — наш единственный трекер и Registry. Разберём его UI подробнее, чтобы вы
понимали, где что искать (пригодится на Семинаре 2 и в ДЗ).

Открываем **http://localhost:5000** → эксперимент **`wine-training`** → run
**`train_rf`**. Пройдёмся по вкладкам:

- **Overview / Parameters** — гиперпараметры прогона: `n_estimators=200`,
  `max_depth=6`, `random_state=42`. Их залогировал `mlflow.log_params(params)`.
- **Metrics** — `accuracy` и `f1_macro`. Их залогировал `mlflow.log_metrics(...)`.
  Если сделать несколько прогонов, MLflow даст сравнить их метрики бок о бок
  (выделите run-ы галочками → **Compare**).
- **Tags** — теги `stage=training`, `dataset=wine`, `model=wine-clf`, которые мы
  проставили через `mlflow.set_tags(...)`. По ним удобно фильтровать список
  прогонов (поле поиска: `tags.stage = "training"`).
- **Artifacts** — папка `model` с файлами (`MLmodel`, `model.pkl`, `conda.yaml`,
  `requirements.txt`, `input_example.json`). Это и есть сериализованная модель
  вместе с её сигнатурой и примером входа.

> **На что обратить внимание.** Файл `MLmodel` внутри артефактов — это «паспорт»
> модели: в нём записаны flavor (`sklearn`), сигнатура (какие колонки на входе,
> какой тип на выходе) и версия окружения. Именно по нему `mlflow.pyfunc.load_model`
> на Семинаре 2 поймёт, как загрузить модель обратно.

Теперь откройте вкладку **Models** (вверху) → **`wine-clf`**. Здесь виден список
версий и напротив нужной — бейдж алиаса **`PROD`**. Кликнув по версии, вы попадёте
на её карточку со ссылкой на исходный run — так Registry и эксперименты связаны.

> **На что обратить внимание.** Весь этот трекинг — локальный и самодостаточный:
> метаданные (params/metrics/tags) лежат в нашем Postgres, артефакты — в нашем
> MinIO. Никакие данные не уходят во внешние сервисы. Это и есть «self-hosted
> MLflow».

---

## Шаг 7. Итог: что мы собрали и что дальше

Мы подняли полный MLOps-стенд и прогнали через него первый цикл:
`make init` → `make up` → снятие DAG с паузы → `make trigger-train` → params/metrics/
tags в MLflow → модель `wine-clf@PROD` в Registry, артефакты в MinIO.

**Оставляем стек запущенным** — на Семинаре 2 мы будем работать с этой же моделью
(инференс из Registry) и настраивать Prometheus/Grafana. Если нужно освободить
ресурсы: `make down` (данные в томах сохранятся, при следующем `make up` всё на
месте).

---

## Частые ошибки и заблуждения

- **«Порт занят» при `make up`** (`Bind for 0.0.0.0:8080 failed`). На хосте уже
  что-то слушает этот порт. Найдите: `lsof -i :8080` (или нужный порт), освободите
  либо поменяйте левую часть маппинга в `docker-compose.yml` (например
  `"18080:8080"`). Правую часть (внутренний порт) не трогаем.
- **macOS: `address already in use` на порту 5000.** Его держит системный AirPlay
  Receiver (`lsof -i :5000` покажет `ControlCe`). Либо выключите его (System Settings →
  General → AirDrop & Handoff → AirPlay Receiver), либо опубликуйте MLflow на другом
  порту через `docker-compose.override.yml` (`ports: !override ["5050:5000"]`) и
  открывайте http://localhost:5050. Внутри docker-сети MLflow остаётся на `:5000`.
- **macOS: `path / is mounted on / but it is not a shared or slave mount`.** Так
  падает `node-exporter` на Docker Desktop: монтирование `/:/host:ro,rslave` работает
  только на Linux. На маке поднимайте стек без него:
  `docker compose up -d --scale node-exporter=0` (в Prometheus его target будет DOWN).
- **DAG-и не появляются в UI.** Проверьте логи scheduler:
  `docker compose logs airflow-scheduler`. Частая причина — синтаксическая ошибка в
  `dags/*.py` (появится import error вверху списка DAG-ов). Второй вариант —
  scheduler ещё не просканировал папку, подождите ~30 сек.
- **`make trigger-train` пишет `dag_id could not be found`.** Либо DAG ещё не
  подхватился, либо опечатка в имени. Проверьте: `docker compose exec
  airflow-webserver airflow dags list`.
- **Задача в статусе `queued` и не идёт в `running`.** Не поднят/не здоров
  `airflow-worker`, либо executor не Celery. Смотрим `make ps` и Flower (5555): есть
  ли online-воркер. Проверяем executor (см. шаг 3).
- **`airflow-worker` бесконечно `unhealthy`.** Обычно ещё идёт `pip install`
  (первый старт). Посмотрите `docker compose logs airflow-worker` — если видите
  `Successfully installed mlflow...`, просто подождите healthcheck.
- **MLflow не пишет артефакты / ошибка S3.** Проверьте, что `minio-init` отработал
  (бакет `mlflow` есть в консоли 9001) и что `AWS_ACCESS_KEY_ID/SECRET` совпадают с
  `MINIO_ROOT_USER/PASSWORD`. В нашем стеке это одно значение `minioadmin`.
- **В логе нет строки про регистрацию модели.** Проверьте, что MLflow здоров
  (`make ps` → `mlops-mlflow` healthy) и доступен по `http://localhost:5000`. DAG
  ходит в него по `MLFLOW_TRACKING_URI=http://mlflow:5000` (имя сервиса в docker-сети).
- **Изменил `.env`, но ничего не поменялось.** Переменные читаются при старте
  контейнера. После правки `.env` нужен `make restart`.
- **Хочу начать «с чистого листа».** `docker compose down -v` удалит именованные
  тома (БД, бакет, дашборды). После этого снова `make init && make up`.

---

## Проверьте себя

После этого занятия вы умеете и понимаете:

- [ ] объяснить, за что отвечает каждый сервис стека (Airflow/MLflow/MinIO/
      Redis/Postgres/Prometheus/Grafana)
- [ ] прочитать `docker-compose.yml`: что даёт YAML anchor `x-airflow-common`,
      зачем `healthcheck` и `depends_on: condition: service_healthy`
- [ ] отличить именованный том (данные переживают `down`) от `down -v`
- [ ] выполнить `make init` и `make up`, проверить здоровье через `make ps` и UI
- [ ] убедиться, что executor — Celery, и увидеть online-воркер во Flower (:5555)
- [ ] снять DAG с паузы и запустить его через `make trigger-train`
- [ ] найти эксперимент, метрики, params и теги в MLflow UI, а артефакты — в MinIO
- [ ] увидеть модель `wine-clf` с алиасом `PROD` в MLflow Model Registry
- [ ] прочитать файл `MLmodel` в артефактах и понять, что такое сигнатура/flavor
- [ ] объяснить разделение MLflow: метаданные в Postgres, артефакты в S3(MinIO)

---

## Связь со следующим семинаром

Сегодня мы **обучили** модель и положили её в Registry с алиасом `PROD`. На
Семинаре 2 мы замкнём цикл: DAG `inference` возьмёт `models:/wine-clf@PROD` из
Registry, прогонит батч-инференс и залогирует результаты. А затем настроим
наблюдаемость — подключим Prometheus и импортируем дашборды Grafana, чтобы видеть
здоровье стека и метрики Airflow в реальном времени.
