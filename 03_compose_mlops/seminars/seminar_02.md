# Семинар 2. Model Registry, инференс и наблюдаемость стека

> Практический воркшоп к блоку «Компоуз-сборка MLOps-проекта».
> Автор материалов: Чернов Евгений (НИУ ВШЭ, магистратура «Искусственный интеллект»).
> Длительность: 2 академических часа.

На Семинаре 1 мы обучили модель и положили её в MLflow Model Registry с алиасом
`PROD`. Сегодня замыкаем MLOps-цикл и добавляем к нему **наблюдаемость**:

1. Разбираемся с **MLflow Model Registry**: версии, алиасы `PROD`/`STAGING`, зачем
   это нужно.
2. Запускаем **второй DAG — `inference`**: он берёт модель из Registry по алиасу
   `PROD`, делает предсказания на новом батче и пишет отчёт.
3. Разбираемся, **что можно мониторить в проде без правильных ответов**: дрейф
   входных данных и распределение предсказаний (отчёт — в логи Airflow,
   метрики — в MLflow, отдельный эксперимент `inference`).
4. Подключаем **Prometheus + Grafana**: разбираем `prometheus.yml`, экспортёры
   (Airflow StatsD, postgres-exporter, node-exporter, честно про mlflow),
   импортируем/смотрим дашборды.
5. Разбираем штатную работу Airflow с внешними системами: **connection, hook,
   сенсоры** — на демо-DAG-ах `db_demo` (Postgres) и `sensor_demo` (файл в S3).

Главная мысль дня: **модель в Registry — это управляемый артефакт с версиями и
«ярлыками», а не файл на диске**. И **работающий стек нужно уметь наблюдать**, а не
угадывать его состояние.

---

## Что понадобится

- Поднятый и здоровый стек с Семинара 1. Проверьте:
  ```bash
  cd 03_compose_mlops
  make ps
  ```
  Если стек погашен — `make up` и дождитесь `healthy`. Если БД чистая (делали
  `make prune`) — сначала `make init`, затем `make up`, затем повторите обучение с
  Семинара 1 (`make trigger-train`), иначе в Registry не будет модели.
- Модель `synth-regressor` с алиасом `PROD` в MLflow (Models → должна быть версия с
  бейджем `@PROD`). Если её нет — `make trigger-train`.
- Браузер для UI: Airflow (8080), MLflow (5000), Grafana (3000), Prometheus (9090).

---

## Часть A. MLflow Model Registry: версии и алиасы

### Шаг 1. Что такое Registry и зачем алиасы

Открываем **MLflow (http://localhost:5000)** → вкладка **Models** → `synth-regressor`.

Registry — это «каталог моделей» поверх экспериментов. Ключевые понятия:
- **Version** (версия) — каждая регистрация модели создаёт новую версию (1, 2, 3…).
  Версии неизменяемы: перееобучили — получили новую.
- **Alias** (алиас) — подвижный «ярлык», указывающий на конкретную версию. У нас
  это `PROD` — «версия, которая сейчас в бою». Завтра переобучим — переведём `PROD`
  на новую версию, и весь код, читающий `models:/synth-regressor@PROD`, поедет на неё
  **без изменений**.

> **На что обратить внимание.** Раньше в MLflow были «стадии» (Staging/Production).
> В современных версиях (2.x) их заменили на **алиасы** — они гибче: можно завести
> `PROD`, `STAGING`, `champion`, `challenger` и указывать ими на любые версии.
> Именно поэтому в DAG-е обучения мы вызывали `client.set_registered_model_alias`.

### Шаг 2. Делаем вторую версию и играем алиасами

Чтобы увидеть версионирование вживую, запустим обучение ещё раз:
```bash
make trigger-train
```
Дождитесь `success` в Airflow. Теперь в MLflow → Models → `synth-regressor` на одну
версию больше. Обратите внимание: наш DAG всегда переводит `PROD` на **самую свежую**
версию — значит, бейдж `@PROD` переехал на только что созданную.

Заведём вручную второй алиас — **`STAGING`** — на одну из старых версий (в примере
версия 1), чтобы почувствовать идею «две среды, две метки»:
```bash
docker compose exec airflow-webserver python - <<'PY'
from mlflow import MlflowClient
c = MlflowClient(tracking_uri="http://mlflow:5000")
c.set_registered_model_alias("synth-regressor", "STAGING", "1")
print("STAGING -> version 1 установлен")
PY
```
> Номер версии передаём **строкой** (`"1"`): в MLflow 2.14 целое число даёт
> `TypeError: bad argument type for built-in operation`.
Ожидаемый вывод:
```
STAGING -> version 1 установлен
```
Обновите страницу модели в MLflow: свежая версия помечена `@PROD`, версия 1 — `@STAGING`.

Так же вручную переводят и `PROD` — например, чтобы вернуть «боевую» метку на лучшую
модель после серии экспериментов. Найдите в эксперименте `synth-training` run с лучшим
`r2`, посмотрите номер его версии в Models и подставьте:
```bash
docker compose exec airflow-webserver python - <<'PY'
from mlflow import MlflowClient
c = MlflowClient(tracking_uri="http://mlflow:5000")
c.set_registered_model_alias("synth-regressor", "PROD", "2")   # номер лучшей версии
print("PROD ->", c.get_model_version_by_alias("synth-regressor", "PROD").version)
PY
```

> **На что обратить внимание.** Это ровно тот механизм, которым в реальной жизни
> делают «выкатку»: новая версия сначала висит на `STAGING` (тестируется), а после
> проверки алиас `PROD` переводят на неё. Код-потребитель обращается по алиасу и не
> знает про номера версий.

---

## Часть B. DAG инференса: модель из Registry в дело

### Шаг 3. Разбираем `inference_dag.py`

Открываем `dags/inference_dag.py`. Логика:
1. Узнаёт версию, на которую указывает алиас `PROD`
   (`client.get_model_version_by_alias`), и берёт модель по `models:/synth-regressor@PROD`.
2. Получает **новый батч** данных — 5 000 строк, которых модель не видела. У нас их
   генерирует тот же генератор, что и при обучении, но с новым случайным seed.
3. Делает предсказания и сохраняет их как результат батча.
4. **Мониторинг без правильных ответов** (об этом ниже) и отчёт в лог Airflow.
5. Открывает отдельный run в эксперименте **`inference`** и логирует туда параметры
   прогона, метрики мониторинга и сами предсказания.

Ключевая строка:
```python
MODEL_URI = f"models:/{MODEL_NAME}@{PROD_ALIAS}"   # models:/synth-regressor@PROD
model = mlflow.pyfunc.load_model(MODEL_URI)
```
Мы **не** указываем номер версии и путь к файлу — только имя и алиас. MLflow сам
разрешает алиас в версию, а версию — в артефакт в MinIO, скачивает и отдаёт готовую
модель. Это и есть «управляемый артефакт».

### Шаг 4. Что мониторить, если правильных ответов нет

На обучении мы считали `rmse` и `r2`, потому что знали правильные ответы для тестовой
выборки. **В проде их в момент инференса нет**: модель предсказывает цену, спрос или
отток, а реальное значение станет известно позже — через день, месяц или никогда.
Поэтому в инференсе мы **не считаем метрики качества**. Генератор мог бы выдать нам
и `y`, но мы его сознательно выбрасываем — как будто его нет.

Что можно проверить без правильных ответов:
- **Дрейф входных данных.** Похожи ли новые данные на те, на которых модель
  обучалась? Если признаки «уехали», модель работает вне знакомой ей области, и её
  предсказаниям нельзя доверять так, как на тесте.
- **Распределение предсказаний.** Если средний прогноз или его разброс резко
  изменились — что-то поменялось либо в данных, либо в модели.

Для сравнения нужен **эталон**. Его сохраняет `train_model` в run обученной модели —
файл `reference_stats.json` (среднее и std каждого признака и предсказаний на
обучающей выборке). Инференс берёт эталон **той версии, что сейчас в `PROD`**:
```python
prod = client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
ref = mlflow.artifacts.load_dict(f"runs:/{prod.run_id}/reference_stats.json")
```
и считает для каждого признака сдвиг среднего в единицах std обучающей выборки:
`|mean_batch − mean_train| / std_train`. Признак с дрейфом больше `0.25` считается
«уехавшим». Это простейшая проверка; в проде используют и более тонкие (PSI, тест
Колмогорова–Смирнова), но идея та же — сравнить новые данные с эталоном.

А метрики качества считают отдельно и позже, когда правильные ответы приходят
(«отложенные метки»): сопоставляют сохранённые предсказания с фактом.

### Шаг 5. Запускаем инференс

Снимаем DAG с паузы и триггерим:
```bash
docker compose exec airflow-webserver airflow dags unpause inference
make trigger-infer
```
**Что вы увидите в UI Airflow (8080):** DAG `inference` → Grid → новый зелёный
запуск. Кликните задачу → **Logs**. Ожидаемый отчёт:
```
========================================================
ОТЧЁТ ИНФЕРЕНСА (модель models:/synth-regressor@PROD, версия 6)
Батч: 5000 строк
Предсказания: mean=2.06 (на обучении 1.98), std=5.67 (на обучении 5.70), p05..p95=-6.8..11.5
Дрейф признаков: max=0.03 std, в среднем 0.01 std
Дрейфа нет: все признаки в пределах 0.25 std от обучения
========================================================
```
(Числа немного отличаются от запуска к запуску — батч каждый раз новый.)

Теперь **сымитируем дрейф**: параметр `shift` сдвигает все признаки относительно
обучающих данных.
```bash
docker compose exec airflow-webserver airflow dags trigger inference --conf '{"shift": 1.5}'
```
Отчёт:
```
Предсказания: mean=10.02 (на обучении 1.98), std=6.52 (на обучении 5.70), p05..p95=-0.2..21.4
Дрейф признаков: max=1.52 std, в среднем 1.50 std
ДРЕЙФ: 30 из 30 признаков сдвинулись больше чем на 0.25 std: f00, f01, f02, ...
```
Модель отработала без ошибок и выдала числа — но средний прогноз вырос в пять раз, а
все признаки ушли на полторы std от того, что модель видела при обучении. Без
мониторинга такой батч молча ушёл бы потребителям. Попробуйте `{"shift": 0.5}` —
дрейф поменьше, но порог он тоже превысит.

> **На что обратить внимание.** Шум в целевой переменной так поймать нельзя: он
> меняет только правильные ответы, а их в проде нет. Поэтому у инференса и нет
> параметра `noise`. Мониторинг без меток ловит изменения во **входах** и
> **выходах** модели, но не саму ошибку — для неё нужны метки.

> **На что обратить внимание.** Если задача упала с ошибкой вида
> `RESOURCE_DOES_NOT_EXIST: ... alias PROD` — значит модель `synth-regressor@PROD`
> ещё не зарегистрирована. Вернитесь к `make trigger-train` (Семинар 1 / Часть A) и
> убедитесь, что алиас `PROD` стоит.

### Шаг 6. Результаты инференса в MLflow (эксперимент `inference`)

В самом конце лога задачи будет строка:
```
Предсказания и метрики мониторинга залогированы в MLflow (эксперимент 'inference').
```
**Что вы увидите в MLflow (http://localhost:5000):** в списке экспериментов слева
появился новый — **`inference`** (отдельно от `synth-training`). Зайдите в него →
run-ы `batch_inference`:
- **Parameters** — `model_uri` (`models:/synth-regressor@PROD`), `model_version`
  (номер версии PROD на момент прогона), `n_samples`, `shift`.
- **Metrics** — метрики мониторинга: `feature_drift_max`, `feature_drift_mean`,
  `drifted_features` (сколько признаков превысили порог), `pred_mean`, `pred_std`,
  `pred_p05`/`pred_p95`, `pred_mean_shift`, `pred_std_ratio` (сравнение с обучением).
- **Tags** — `stage=inference`, `model=synth-regressor`, `alias=PROD` и
  `drift=yes/no`. По тегу удобно найти все проблемные батчи: `tags.drift = "yes"`.
- **Artifacts** — `predictions.json`, сами предсказания батча.

Сделайте несколько запусков с разным `shift` и сравните их через **Compare**: на
графике `feature_drift_max` и `pred_mean` видно, как растёт сдвиг.

> **На что обратить внимание.** Мы разнесли два канала результатов осознанно:
> **человекочитаемый отчёт** идёт в логи Airflow (там его смотрит дежурный инженер),
> а **числовые метрики** — в MLflow (там их удобно сравнивать между прогонами).
> Обучение и инференс живут в **разных экспериментах** (`synth-training` и
> `inference`) — так их прогоны не перемешиваются.

---

## Часть C. Наблюдаемость: Prometheus + Grafana

Стек работает — но как понять, что он *здоров*, не заходя в каждый UI? Для этого —
метрики. **Prometheus** их собирает, **Grafana** рисует.

### Шаг 7. Разбираем `prometheus/prometheus.yml`

Открываем `prometheus/prometheus.yml`. Prometheus работает по модели **pull**: сам
периодически (`scrape_interval: 15s`) ходит на `/metrics` каждого таргета. Наши
scrape-jobs:

- **prometheus** (`localhost:9090`) — самомониторинг.
- **node-exporter** (`node-exporter:9100`) — CPU/RAM/диск/сеть хоста.
- **postgres-exporter** (`postgres-exporter:9187`) — соединения, БД, транзакции.
- **airflow-statsd** (`statsd-exporter:9102`) — метрики Airflow.
- **mlflow** — **закомментирован осознанно**: у `mlflow server` нет эндпоинта
  `/metrics` в формате Prometheus, отдельного экспортёра из коробки тоже нет. Честно
  помечено комментарием в конфиге; косвенно за MLflow наблюдаем через node-exporter
  (нагрузка) и postgres-exporter (его backend-store).

> **На что обратить внимание.** Имена таргетов — это **имена сервисов из
> docker-compose** (`node-exporter`, `postgres-exporter`, ...). Внутри docker-сети
> они резолвятся через встроенный DNS. Снаружи (с хоста) те же экспортёры доступны
> по `localhost:<порт>`.

### Шаг 8. Как Airflow попадает в Prometheus (StatsD → exporter)

Airflow не умеет отдавать метрики в формате Prometheus напрямую — он шлёт их по
протоколу **StatsD** (UDP). Цепочка такая:
```
Airflow ──statsd/UDP:9125──▶ statsd-exporter ──HTTP/:9102/metrics──▶ Prometheus
```
За приём отвечают env в `x-airflow-common`:
```yaml
AIRFLOW__METRICS__STATSD_ON: "true"
AIRFLOW__METRICS__STATSD_HOST: statsd-exporter
AIRFLOW__METRICS__STATSD_PORT: "9125"
AIRFLOW__METRICS__STATSD_PREFIX: airflow
```
А за «перевод» statsd-имён в аккуратные prometheus-метрики — файл
`config/statsd-mapping.yml`. Например правило:
```yaml
- match: "airflow.dag.*.*.duration"
  name: "airflow_task_duration"
  labels:
    dag_id: "$1"
    task_id: "$2"
```
превращает `airflow.dag.train_model.train_and_register.duration` в метрику
`airflow_task_duration{dag_id="train_model", task_id="train_and_register"}`. Без
mapping имя «взорвалось» бы: у каждой задачи была бы своя метрика. Хороший пример —
`airflow.ti.finish.<dag>.<task>.<state>`: без правила это 13 отдельных метрик на
**каждую** задачу, с правилом — одна `airflow_ti_finish{dag_id, task_id, state}`.

> **На что обратить внимание.** Airflow шлёт часть метрик дважды: общую
> (`dagrun.duration.success`) и по DAG-у (`dagrun.duration.success.<dag_id>`). Если
> обе попадут в одно имя с разным набором labels, statsd-exporter молча выкинет
> вариант с `dag_id`. Поэтому в начале mapping общие агрегаты отбрасываются
> (`action: drop`).

Проверим, что exporter реально получает данные от Airflow:
```bash
curl -s http://localhost:9102/metrics | grep -i airflow_ | head
```
Ожидаемый вывод — строки вида:
```
airflow_scheduler_heartbeat 77
airflow_ti_successes 3
airflow_dagrun_duration_success{dag_id="train_model",quantile="0.5"} 5.03
airflow_task_duration{dag_id="train_model",task_id="train_and_register",quantile="0.5"} 3.81
```
Если пусто — прогоните пару раз `make trigger-train`/`make trigger-infer`, чтобы
Airflow сгенерировал события, и повторите curl.

### Шаг 9. Проверяем таргеты в Prometheus

Открываем **Prometheus (http://localhost:9090)** → **Status → Targets**.

**Что вы увидите:** список job-ов. У `prometheus`, `node-exporter`,
`postgres-exporter`, `airflow-statsd` состояние **UP** (зелёное). Job `mlflow` в
списке нет — он закомментирован (см. шаг 7). На macOS, если стек поднят без
`node-exporter` (см. Семинар 1, «Частые ошибки»), его target будет **DOWN** — это
ожидаемо, и панели дашборда «Node Exporter» там останутся пустыми.

Попробуем первый запрос. В строке **Graph** введите:
```
airflow_ti_successes
```
и нажмите Execute. Увидите значение счётчика успешных task instance. Ещё пример —
загрузка CPU:
```
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

> **На что обратить внимание.** `Target DOWN` у экспортёра почти всегда означает,
> что либо контейнер не поднялся (`make ps`), либо опечатка в имени/порте таргета в
> `prometheus.yml`. Prometheus прямо в UI показывает текст ошибки соединения —
> читайте его.

### Шаг 10. Grafana: datasource и дашборды приезжают сами

Открываем **Grafana (http://localhost:3000)**, логин `admin`/`admin` (при первом
входе Grafana предложит сменить пароль — для учебного стенда можно пропустить).

Мы настроили **provisioning** — Grafana сама, без ручных кликов, подхватывает
конфигурацию из смонтированных файлов:
- `grafana/provisioning/datasources/datasource.yml` — добавляет источник
  **Prometheus** (`http://prometheus:9090`, отмечен как default).
- `grafana/provisioning/dashboards/dashboards.yml` — говорит читать дашборды из
  `/var/lib/grafana/dashboards` (туда смонтирована папка `grafana/dashboards`).
- `grafana/dashboards/*.json` — сами дашборды.

**Что вы увидите:** слева **Dashboards** → папка **MLOps** с двумя дашбордами:
- **Node Exporter — Overview**: CPU (%), RAM (%), заполнение корневого диска
  (gauge), сетевой трафик.
- **Airflow & Postgres — Overview**: `pg_up`, частота heartbeat планировщика
  (ударов в минуту; `0` — планировщик мёртв), success/failure rate задач, активные
  соединения Postgres, очередь Celery. Линия failure rate пуста, пока ни одна задача
  не упала: счётчик `airflow_ti_failures` появляется только после первого падения.

Откройте оба. При первом открытии дашборд спросит datasource — выберите
**Prometheus** (он подставится сам, т.к. default). Панели заполнятся данными.

> **На что обратить внимание.** Дашборды используют переменную `${DS_PROMETHEUS}` —
> так JSON не привязан к конкретному ID источника и переносится между стендами. Это
> стандартная практика для «шаринга» дашбордов.

Чтобы панели Airflow ожили, сгенерируйте активность:
```bash
make trigger-train
make trigger-infer
```
Через 15–30 сек (интервал scrape) на дашборде «Airflow & Postgres» пойдут данные по
success/failure rate и очереди Celery.

### Шаг 11. (Опционально) Импорт готового дашборда из Grafana.com

Наши дашборды — компактные учебные. В реальной жизни часто берут готовые. Покажем
механику: Grafana → **Dashboards → New → Import** → введите ID `1860` (популярный
«Node Exporter Full») → Load → выберите datasource **Prometheus** → Import.

> **На что обратить внимание.** Готовые дашборды рассчитаны на «полный» набор
> метрик node-exporter. Часть панелей может быть пустой, если у вашего экспортёра
> отключены нужные коллекторы — это нормально, не ошибка конфигурации.

---

## Часть D. Connections, хуки и сенсоры

Наши `train_model` и `inference` ходят в MLflow и MinIO «в обход» Airflow: адреса и
ключи берутся из переменных окружения, клиенты создаются вручную. Для учебного стенда
это нормально, но в Airflow есть штатный механизм работы с внешними системами. Его и
разберём на двух демо-DAG-ах: `db_demo` (Postgres) и `sensor_demo` (ожидание файла в S3).

### Шаг 12. Connection, Hook, Operator — кто за что отвечает

- **Connection — данные: куда и с какими учётными данными подключаться.** Запись с
  полями `conn_id`, `conn_type`, `host`, `port`, `login`, `password`, `schema`,
  `extra`. Сама ничего не умеет.
- **Hook — код: как работать с системой.** Класс, который по `conn_id` достаёт
  connection, создаёт клиента и даёт удобные методы: `PostgresHook.get_records(...)`,
  `S3Hook.read_key(...)`.
- **Operator / Sensor — задача DAG-а**, которая обычно использует хук внутри:
  `SQLExecuteQueryOperator(conn_id=...)` создаёт `PostgresHook`, `S3KeySensor` —
  `S3Hook`.

```
Connection (куда, логин/пароль) ◀── по conn_id ── Hook (клиент + методы) ◀── Operator / Sensor
```

Зачем так: паролей нет в коде DAG-ов, один и тот же DAG работает на dev и prod с
разными connection-ами под тем же `conn_id`, секреты можно держать в Vault/Secrets Manager.

В нашем стеке connection-ы заданы **переменными окружения** `AIRFLOW_CONN_<ID>` в
`x-airflow-common` (`docker-compose.yml`):
```yaml
AIRFLOW_CONN_MINIO_S3: >-
  {"conn_type": "aws", "login": "minioadmin", "password": "minioadmin",
   "extra": {"endpoint_url": "http://minio:9000"}}
AIRFLOW_CONN_APP_DB: >-
  {"conn_type": "postgres", "host": "postgres", "port": 5432,
   "login": "app", "password": "app", "schema": "app"}
```
Проверим, что Airflow их видит:
```bash
docker compose exec airflow-webserver airflow connections get app_db
docker compose exec airflow-webserver airflow connections get minio_s3
```

> **На что обратить внимание.** Connection-ы из переменных окружения **не видны** в
> UI (Admin → Connections): там показываются только записи из БД Airflow. Это не
> ошибка — Airflow ищет connection сначала в env и хранилище секретов, потом в БД.
> Команда `airflow connections test` в Airflow 2.7+ по умолчанию выключена
> (`test_connection = Disabled`), поэтому проверяем через `connections get` и запуск DAG-а.

**Если стек поднимали раньше.** Для `db_demo` нужна отдельная БД `app`. На свежем
стеке её создаёт `config/postgres-init.sh`, но этот скрипт выполняется только при
**первой** инициализации тома Postgres. Если том уже есть (вы делали Семинар 1),
создайте БД один раз вручную:
```bash
docker compose exec postgres psql -U airflow \
  -c "CREATE USER app WITH PASSWORD 'app';" -c "CREATE DATABASE app OWNER app;"
```
Ожидаемый вывод: `CREATE ROLE` и `CREATE DATABASE` (или `already exists` — значит,
БД уже есть, всё в порядке).

### Шаг 13. `db_demo`: пишем и читаем Postgres через connection

Открываем `dags/db_demo_dag.py`. Четыре задачи:
```
create_table ──▶ load_samples ──▶ aggregate ──▶ report
```
1. **`create_table`** — `SQLExecuteQueryOperator` с `conn_id="app_db"` выполняет
   `CREATE TABLE IF NOT EXISTS wine_samples (...)`. Ни одного пароля в коде.
2. **`load_samples`** — `@task` с `PostgresHook(postgres_conn_id="app_db")`: берёт 20
   случайных сэмплов Wine и вставляет через `hook.insert_rows(...)`, помечая каждую
   строку `run_id` текущего запуска.
3. **`aggregate`** — снова `SQLExecuteQueryOperator`: сводка по классам **только для
   текущего запуска**. Результат `SELECT` автоматически уходит в **XCom**.
4. **`report`** — `@task` получает строки из XCom (`aggregate.output`) и печатает отчёт.

Обратите внимание, как передаётся `run_id` в запрос:
```python
sql="... WHERE run_id = %(run_id)s ...",
parameters={"run_id": "{{ run_id }}"},
```
`{{ run_id }}` — шаблон Jinja: Airflow подставит id запуска перед выполнением. А
`%(run_id)s` — параметр драйвера: значение передаётся отдельно от текста SQL, поэтому
нет SQL-инъекций. Склеивать SQL из строк (`f"... = '{run_id}'"`) не нужно.

Запускаем:
```bash
docker compose exec airflow-webserver airflow dags unpause db_demo
docker compose exec airflow-webserver airflow dags trigger db_demo
```
**Что вы увидите в UI Airflow (8080):** DAG `db_demo` → Grid → четыре зелёные задачи.
В логе `load_samples`:
```
Вставлено строк: 20, всего в таблице: 20
```
В логе `report`:
```
СВОДКА ПО ЗАПУСКУ (класс | сэмплов | средний alcohol)
  класс 0 |  8 | 13.59
  класс 1 |  9 | 12.34
  класс 2 |  3 | 13.16
```
(числа у вас будут другими — сэмплы случайные). На задаче `aggregate` откройте
вкладку **XCom** — там строки результата `SELECT`, которые забрал `report`.

Проверим данные прямо в БД:
```bash
docker compose exec postgres psql -U app -d app \
  -c "SELECT run_id, count(*) FROM wine_samples GROUP BY run_id;"
```
Запустите DAG ещё раз: в таблице станет 40 строк, а сводка `report` по-прежнему по 20
строкам — только своего запуска.

> **На что обратить внимание.** В `load_samples` сэмплы берутся с явным генератором
> `random_state=np.random.default_rng()`. Без него все запуски вставляли бы **одни и
> те же** строки: Celery-воркер форкает процессы из одного родителя, у всех одинаковое
> состояние глобального RNG numpy, и `df.sample()` без seed каждый раз выбирает одно и
> то же. Классическая ловушка случайности в воркерах.

### Шаг 14. `sensor_demo`: ждём файл в S3

**Сенсор** — задача, которая ничего не делает, а **ждёт условия**: появления файла,
строки в таблице, завершения внешнего job-а. Открываем `dags/sensor_demo_dag.py`:
```
wait_for_file ──▶ process_files ──▶ archive_files
```
1. **`wait_for_file`** — `S3KeySensor` через connection `minio_s3` раз в 10 с
   проверяет, есть ли в бакете `incoming` файлы по маске `new/*.csv`.
2. **`process_files`** — `S3Hook` читает найденные CSV и печатает сводку (число строк,
   средние по колонкам). Список файлов уходит в XCom.
3. **`archive_files`** — переносит файлы в `processed/<run_id>/`, чтобы следующий
   запуск снова ждал **новый** файл, а не срабатывал на старый.

Ключевые параметры сенсора:
```python
mode="reschedule",              # между проверками слот воркера свободен
poke_interval=10,               # проверять каждые 10 с
timeout=timedelta(minutes=15),  # не дождались — задача падает
```
Режимов два:
- **`poke`** (по умолчанию) — задача всё время ожидания **занимает слот воркера** и
  спит между проверками. Подходит для коротких ожиданий.
- **`reschedule`** — после каждой проверки задача **освобождает слот** и ставится в
  статус `up_for_reschedule`, а через `poke_interval` планировщик запускает её снова.
  Для ожиданий в минуты и часы — только так, иначе десяток ждущих сенсоров займут
  всех воркеров.

**Запускаем без файла:**
```bash
docker compose exec airflow-webserver airflow dags unpause sensor_demo
docker compose exec airflow-webserver airflow dags trigger sensor_demo
```
**Что вы увидите в UI Airflow:** задача `wait_for_file` в статусе
**`up_for_reschedule`** (бирюзовый квадрат), `process_files` и `archive_files` ждут.
В логе сенсора:
```
Poking for key : s3://incoming/new/*.csv
Rescheduling task, marking task as UP_FOR_RESCHEDULE
```
Загляните во **Flower (5555)**: активных задач у воркера нет — сенсор не держит слот.

**Кладём файл.** Пример данных лежит в `data/wine_batch.csv`. Удобнее всего через
MinIO Client `mc` (macOS: `brew install minio/stable/mc`, Linux — бинарник с
min.io/download):
```bash
mc alias set mlops http://localhost:9000 minioadmin minioadmin
mc cp data/wine_batch.csv mlops/incoming/new/wine_batch.csv
```
Без `mc` — через консоль MinIO (http://localhost:9001): Buckets → `incoming` →
Upload, предварительно создав папку `new`.

В течение ~10 с сенсор увидит файл, и DAG доедет до конца. В логе `process_files`:
```
Найдено файлов: 1 -> ['new/wine_batch.csv']
Файл s3://incoming/new/wine_batch.csv: строк=15, колонки=['alcohol', 'malic_acid', 'color_intensity', 'proline', 'target']
```
Проверим, что файл переехал в архив:
```bash
mc ls -r mlops/incoming
```
```
... processed/manual__2026-...+00:00/wine_batch.csv
```
Запустите `sensor_demo` ещё раз — сенсор снова будет ждать: старый файл уже в
`processed/`, а он смотрит только в `new/`.

> **На что обратить внимание.** Сенсор ждёт не бесконечно: через `timeout` (15 мин)
> задача упадёт. В проде так и задумано — лучше явная ошибка «данные не пришли», чем
> пайплайн, который молча висит сутками.

### Шаг 15. Логи задач лежат в S3

Заодно посмотрим, куда Airflow складывает логи. В `x-airflow-common` включено
**remote logging**:
```yaml
AIRFLOW__LOGGING__REMOTE_LOGGING: "true"
AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER: s3://airflow-logs
AIRFLOW__LOGGING__REMOTE_LOG_CONN_ID: minio_s3
```
Воркер пишет лог локально в `./logs`, а когда задача завершается — выгружает его в
бакет `airflow-logs` через тот же connection `minio_s3`:
```bash
mc ls -r mlops/airflow-logs/dag_id=sensor_demo/
```
В UI в начале лога задачи будет строка `*** Found logs in s3:` — webserver читает лог
из S3, а не с диска воркера. Это важно, когда воркеров много или они временные (как
поды в Kubernetes): воркер исчез, а лог остался.

---

## Шаг 16. Итог

Мы замкнули MLOps-цикл и накрыли его наблюдаемостью:
- Разобрались с Registry: версии + алиасы `PROD`/`STAGING`, «выкатка» через
  перевод алиаса.
- DAG `inference` берёт `models:/synth-regressor@PROD`, предсказывает на новом батче
  и без правильных ответов проверяет дрейф признаков и предсказаний относительно
  эталона из обучения: отчёт — в логи Airflow, метрики — в MLflow (эксперимент
  `inference`).
- Prometheus собирает метрики (node/postgres/airflow-statsd), Grafana рисует их на
  provisioning-дашбордах — без ручной настройки.
- Разобрались с connection/hook/operator: `db_demo` пишет и читает Postgres через
  `app_db`, `sensor_demo` ждёт файл в MinIO через `minio_s3` в режиме `reschedule`.

Если стенд больше не нужен — `make down` (данные в томах останутся). Полная очистка
— `make prune` (удалит и volume).

---

## Частые ошибки и заблуждения

- **`inference` падает: `alias PROD not found`.** Модель `synth-regressor` ещё не
  зарегистрирована или алиас не проставлен. Запустите `make trigger-train` и
  проверьте в MLflow → Models бейдж `@PROD`.
- **«Алиас — это то же, что версия».** Нет. Версия неизменяема и растёт (1,2,3…),
  алиас — подвижный ярлык, который вы переводите между версиями. Код читает по
  алиасу и не меняется при переезде.
- **Target `airflow-statsd` UP, но метрик `airflow_*` нет.** Airflow ещё не
  сгенерировал события. Триггерните DAG-и и проверьте `curl
  localhost:9102/metrics`.
- **В Prometheus нет job `mlflow`.** Так и задумано: у MLflow нет Prometheus-эндпоинта,
  job закомментирован осознанно (шаг 7). Это не пропущенная настройка.
- **Grafana: «No data» на панелях Airflow.** Либо ещё не прошёл scrape-интервал
  (подождите ~30 сек), либо не было активности Airflow (триггерните DAG), либо
  datasource не выбран — проверьте Connections → Data sources → Prometheus.
- **Grafana: дашбордов нет в списке.** Проверьте, что папка `grafana/dashboards`
  смонтирована и JSON валиден. Логи: `docker compose logs grafana | grep -i
  provisioning`. Битый JSON Grafana пропустит с ошибкой в логе.
- **`postgres-exporter` DOWN.** Проверьте, что Postgres здоров (`make ps`) и строка
  `DATA_SOURCE_NAME` в compose указывает на `postgres:5432/airflow` с верным
  логином/паролем (`airflow/airflow`).
- **`db_demo` падает: `database "app" does not exist` / `password authentication
  failed for user "app"`.** Том Postgres создан до появления БД `app`, и init-скрипт
  не отработал. Создайте БД командой из шага 12.
- **`db_demo`: connection `app_db` не найден (`The conn_id app_db isn't defined`).**
  Контейнеры Airflow подняты со старым compose без `AIRFLOW_CONN_APP_DB`. Выполните
  `make up` — Compose пересоздаст их с новыми переменными.
- **`sensor_demo` не просыпается, хотя файл загружен.** Проверьте путь: сенсор ждёт
  `incoming/new/*.csv`. Файл в корне бакета (`incoming/wine_batch.csv`) или с другим
  расширением он не увидит. `mc ls -r mlops/incoming` покажет, куда файл реально лёг.
- **`sensor_demo` падает с `NoSuchBucket`.** Нет бакета `incoming`: `minio-init`
  отработал со старым compose. `make up` перезапустит его и создаст бакет.
- **«Сенсор завис».** Статус `up_for_reschedule` — это нормальное ожидание в режиме
  `reschedule`, а не зависание. Задача перезапускается раз в `poke_interval` и сама
  упадёт по `timeout`.
- **Сменил дашборд в UI — после `restart` изменения пропали.** Папка дашбордов
  смонтирована **read-only** из файлов репозитория (provisioning — источник
  истины). Правьте JSON в `grafana/dashboards/`, а не в UI.

---

## Проверьте себя

После этого занятия вы умеете и понимаете:

- [ ] объяснить разницу «версия vs алиас» в MLflow Model Registry
- [ ] завести/перевести алиасы `PROD` и `STAGING` на нужные версии модели
- [ ] прочитать `inference_dag.py` и понять строку `models:/synth-regressor@PROD`
- [ ] запустить DAG `inference` (`make trigger-infer`) и прочитать отчёт в логах
- [ ] объяснить, почему на инференсе не считают rmse/r2 и что можно мониторить без меток
- [ ] запустить `inference` со сдвигом (`--conf '{"shift": 1.5}'`) и найти дрейф в отчёте и в MLflow
- [ ] найти метрики мониторинга в MLflow (эксперимент `inference`, run `batch_inference`, тег `drift`)
- [ ] объяснить pull-модель Prometheus и назначение каждого scrape-job
- [ ] проследить цепочку метрик Airflow: StatsD → statsd-exporter → Prometheus
- [ ] объяснить роль `config/statsd-mapping.yml` (statsd-имена → метрики с labels)
- [ ] проверить таргеты в Prometheus (Status → Targets) и написать простой запрос
- [ ] объяснить, почему нет job `mlflow`, и как за MLflow наблюдать косвенно
- [ ] открыть provisioning-дашборды Grafana и понять, откуда они взялись
- [ ] объяснить разницу connection / hook / operator
- [ ] задать connection через переменную `AIRFLOW_CONN_<ID>` и проверить `airflow connections get`
- [ ] прочитать `db_demo` и объяснить, как `{{ run_id }}` попадает в SQL-параметр и зачем параметр, а не склейка строк
- [ ] найти результат `SELECT` во вкладке XCom и объяснить, как его получает следующая задача
- [ ] объяснить разницу режимов сенсора `poke` и `reschedule` и увидеть `up_for_reschedule` в UI
- [ ] «разбудить» `sensor_demo`, положив файл в MinIO через `mc` или консоль
- [ ] найти лог задачи в бакете `airflow-logs`

---

## Что дальше

Вы прошли полный локальный MLOps-цикл: обучение → Registry с алиасами → инференс →
наблюдаемость. Это фундамент для домашних заданий и итогового проекта, где вы
будете собирать похожие связки под свою задачу. Следующий крупный блок курса —
**LLM-гейтвей и observability для LLM** (LiteLLM + Langfuse + RAG): те же принципы
«всё в compose, управление через make, метрики в Prometheus/Grafana», но уже вокруг
больших языковых моделей.
