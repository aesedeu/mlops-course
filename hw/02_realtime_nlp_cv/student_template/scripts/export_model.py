#!/usr/bin/env python3
"""Экспорт CV-модели вашего варианта (torchvision, веса ImageNet-1k) → ONNX для Triton.

    python scripts/export_model.py --model efficientnet_b0

Какая модель ваша — печатает `make variant` (зависит от email в .env).
Результат: model_repository/<model>/1/model.onnx (opset 17, динамическая ось batch,
вход INPUT__0 [N,3,224,224] FP32, выход OUTPUT__0 [N,1000] — логиты, без softmax).

Скрипт ТОЛЬКО экспортирует веса. Препроцессинг картинки (размер ресайза, интерполяция,
кроп, нормализация) у моделей разный — его вы реализуете в gateway сами; официальные
параметры смотрите в документации torchvision для своих весов (weights.transforms()).
"""
from __future__ import annotations

import argparse
import inspect
import os

import torch
import torchvision.models as models

# Варианты ДЗ: имя → (конструктор, веса). Порядок классов у всех — стандартный
# ImageNet-1k (тот же imagenet_classes.txt, class_id 207 == 'golden retriever').
VARIANTS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(VARIANTS))
    args = parser.parse_args()

    ctor, weights = VARIANTS[args.model]
    out_dir = os.path.abspath(os.path.join(HERE, "..", "model_repository", args.model, "1"))
    out_path = os.path.join(out_dir, "model.onnx")
    os.makedirs(out_dir, exist_ok=True)

    model = ctor(weights=weights).eval()
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    export_kwargs = dict(
        input_names=["INPUT__0"],
        output_names=["OUTPUT__0"],
        opset_version=17,
        dynamic_axes={"INPUT__0": {0: "batch"}, "OUTPUT__0": {0: "batch"}},
        do_constant_folding=True,
    )
    # torch >= 2.9 по умолчанию экспортирует через torch.export (нужен onnxscript,
    # веса уходят во внешний model.onnx.data). Для Triton удобнее один переносимый
    # model.onnx — поэтому явно выбираем классический экспортёр, где он доступен.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, out_path, **export_kwargs)

    print(f"[ok] {args.model} ({weights}) экспортирован в: {out_path}")
    print("[ok] вход INPUT__0 [N,3,224,224] FP32 → выход OUTPUT__0 [N,1000] (логиты)")
    print("[!!] препроцессинг у моделей разный — сверьтесь с weights.transforms() своих весов")


if __name__ == "__main__":
    main()
