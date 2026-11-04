# Клиент (в Airflow эту роль играет scheduler): отправляет задачи в очередь
# и забирает результаты. Сам клиент задачи НЕ исполняет.
import time

from tasks import add, fail, slow_square

# 1. Одна задача. .delay() только кладёт сообщение в Redis и сразу возвращается.
res = add.delay(2, 3)
print(f"отправили add(2, 3), id={res.id}, state={res.state}")
print(f"результат: {res.get(timeout=10)}, state={res.state}\n")

# 2. Шесть медленных задач сразу. Каждая спит 3 c, но воркеры берут их
#    параллельно, поэтому всё вместе занимает заметно меньше 18 c.
t0 = time.time()
jobs = [slow_square.delay(i) for i in range(6)]
print("отправили 6 x slow_square, состояния:", [j.state for j in jobs])
for j in jobs:
    print("  ", j.get(timeout=60))
print(f"6 задач по 3 c выполнены за {time.time() - t0:.1f} c\n")

# 3. Ошибка в задаче не роняет воркер: исключение возвращается клиенту.
res = fail.delay()
try:
    res.get(timeout=10)
except Exception as e:
    print(f"fail(): state={res.state}, ошибка={e!r}")
