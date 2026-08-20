"""Проверяет доступность эндпоинтов NVIDIA API текущим ключом."""
import httpx

from _keyhelper import API_BASE, headers

urls = [
    f"{API_BASE}/models",
    f"{API_BASE}/chat/completions",
    f"{API_BASE}/completions",
]

for url in urls:
    r = httpx.get(url, headers=headers(), timeout=30)
    print(f"{url}: {r.status_code} - {r.text[:300]}")
