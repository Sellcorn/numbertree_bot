"""Общий помощник для тест-скриптов: ключ NVIDIA берётся из окружения/.env."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")


def api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        sys.exit("NVIDIA_API_KEY не задан. Положите ключ в .env рядом с ботом.")
    return key


def headers() -> dict:
    return {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }
