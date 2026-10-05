"""
Синтетическая регрессия: 30 признаков, нелинейная зависимость, шум.

Используется и обучением, и инференсом, поэтому данные генерируются одинаково:
  * коэффициенты зависимости фиксированы (seed=0) — «закон природы» не меняется;
  * признаки и шум зависят от seed — так можно получить новую выборку.

Целевая переменная зависит только от f00..f11, признаки f12..f29 — чистый шум.
Шум ограничивает качество сверху: даже идеальная модель не даст R² = 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_FEATURES = 30
N_INFORMATIVE = 12
FEATURES = [f"f{i:02d}" for i in range(N_FEATURES)]

_COEF = np.random.default_rng(0).uniform(-1, 1, 6)


def make_dataset(n_samples: int = 100_000, noise: float = 3.0, shift: float = 0.0,
                 seed: int | None = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Возвращает (X, y).

    noise — стандартное отклонение шума в целевой переменной;
    shift — сдвиг среднего всех признаков (имитация дрейфа входных данных);
    seed  — None даёт новую случайную выборку при каждом вызове.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=shift, size=(n_samples, N_FEATURES))
    y = (3 * X[:, 0] - 2 * X[:, 1]
         + 1.5 * X[:, 2] * X[:, 3]
         + 4 * np.sin(X[:, 4])
         + 2 * X[:, 5] ** 2
         + X[:, 6:N_INFORMATIVE] @ _COEF
         + rng.normal(scale=noise, size=n_samples))
    return pd.DataFrame(X, columns=FEATURES), pd.Series(y, name="target")
