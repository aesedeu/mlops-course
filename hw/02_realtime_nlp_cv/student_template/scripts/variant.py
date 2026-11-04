#!/usr/bin/env python3
"""Ваш вариант ДЗ: какую CV-модель вы обслуживаете. Зависит ТОЛЬКО от email.

    make variant          (email берётся из STUDENT_EMAIL в .env)

Грейдер считает вариант по той же формуле из email, который отдаёт ваш /info,
и сверяет ответы /vision с эталоном именно этой модели. Указывайте свой
настоящий email — его видит преподаватель в отчёте.
"""
from __future__ import annotations

import hashlib
import os
import sys

# Порядок важен — такой же список у грейдера.
VARIANTS = ["resnet50", "efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"]


def variant_for(email: str) -> str:
    digest = hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()
    return VARIANTS[int(digest[:8], 16) % len(VARIANTS)]


def read_env(key: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    try:
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    return ""


def main() -> None:
    email = read_env("STUDENT_EMAIL")
    if "@" not in email:
        print("!! задайте свой STUDENT_EMAIL в .env (cp .env.example .env)", file=sys.stderr)
        sys.exit(1)
    v = variant_for(email)
    print(f"email:   {email}")
    print(f"вариант: {v}")
    current = read_env("CV_MODEL")
    if current != v:
        print(f"\n>> пропишите в .env:  CV_MODEL={v}" + (f"   (сейчас CV_MODEL={current})" if current else ""))
        sys.exit(2)
    print("CV_MODEL в .env совпадает с вариантом ✔")


if __name__ == "__main__":
    main()
