#!/usr/bin/env python3
"""Генерация JWT (HS256) для передачи грейдеру как Bearer-токен.

Работает на ЧИСТОЙ стандартной библиотеке (hmac/hashlib/base64) — никаких
зависимостей, достаточно системного python3. Секрет берётся из окружения
JWT_SECRET (тот же, что валидирует gateway). Токен стандартный — gateway
проверяет его обычным PyJWT.

Использование:
    JWT_SECRET=... python3 scripts/make_token.py [--sub student --role student --days 30]
Печатает строку токена (без слова Bearer) — её передают грейдеру в поле token.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def encode_hs256(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    )
    sig = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return signing_input + "." + _b64url(sig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", default="student", help="subject (кто владелец токена)")
    parser.add_argument("--role", default="student", help="роль в payload")
    parser.add_argument("--days", type=int, default=30, help="срок жизни в днях")
    args = parser.parse_args()

    secret = os.environ.get("JWT_SECRET")
    if not secret:
        print("ERROR: переменная окружения JWT_SECRET не задана (см. .env)", file=sys.stderr)
        sys.exit(1)

    now = int(time.time())
    payload = {
        "sub": args.sub,
        "role": args.role,
        "iat": now,
        "exp": now + args.days * 86400,
    }
    print(encode_hs256(payload, secret))


if __name__ == "__main__":
    main()
