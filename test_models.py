"""Проверяет, что конкретные id моделей отвечают на короткий запрос.

    python test_models.py minimaxai/minimax-m2 minimaxai/minimax-m1-80k
"""
import sys

import httpx

from _keyhelper import API_BASE, headers

models = sys.argv[1:]
if not models:
    sys.exit("Укажите id моделей: python test_models.py <model_id> [<model_id> ...]")

for model in models:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10,
    }
    r = httpx.post(f"{API_BASE}/chat/completions", headers=headers(), json=payload, timeout=60)
    print(f"{model}: {r.status_code} - {r.text[:200]}")
