# model_repository

Сюда Triton смотрит за моделями. Структура для вашего варианта (`make variant`):

```
model_repository/
└── <CV_MODEL>/              # напр. efficientnet_b0 — имя модели в Triton
    ├── config.pbtxt         # пишете сами
    └── 1/
        └── model.onnx       # кладёт make init
```

В `config.pbtxt` задайте: `name` (= CV_MODEL), `platform` (`onnxruntime_onnx`),
`max_batch_size`, вход `INPUT__0` и выход `OUTPUT__0` (имена и размерности — по
ONNX-графу и API_CONTRACT.md), `instance_group` (GPU) и `dynamic_batching`.
Параметры батчинга подберите сами: грейдер проверяет по метрикам Triton, что под
нагрузкой модель реально обрабатывает несколько картинок за запуск.

Документация: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md
