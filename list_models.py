"""Список моделей, доступных ключу на NVIDIA API.

    python list_models.py           # модели с 'minimax' в id
    python list_models.py deepseek  # фильтр по подстроке
    python list_models.py --all     # весь каталог
"""
import sys

import httpx

from _keyhelper import API_BASE, headers

arg = sys.argv[1] if len(sys.argv) > 1 else "minimax"

r = httpx.get(f"{API_BASE}/models", headers=headers(), timeout=30)
if r.status_code != 200:
    sys.exit(f"Ошибка {r.status_code}: {r.text[:300]}")

ids = sorted(m["id"] for m in r.json()["data"])
print(f"Всего моделей доступно: {len(ids)}\n")

if arg == "--all":
    for model_id in ids:
        print(model_id)
else:
    hits = [m for m in ids if arg.lower() in m.lower()]
    if hits:
        print(f"Совпадения по '{arg}':")
        for model_id in hits:
            print(f"  {model_id}")
    else:
        print(f"По '{arg}' совпадений нет. Запустите с --all, чтобы увидеть весь каталог.")
