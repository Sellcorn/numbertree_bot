"""Проверяет, что бот идёт в интернет только когда это нужно.

Регрессия: описание инструмента требовало искать «любые факты, которые могли
измениться после отсечки знаний». Под это подходит что угодно незнакомое, и
бот лез в поиск даже на бессмысленное слово «хайблс» и на чисто теоретический
вопрос про nil map.

Тест бьёт по живому API, поэтому нужен NVIDIA_API_KEY и он тратит запросы.
Запуск: python test_search_trigger.py
"""
import asyncio
import logging
import sys
from datetime import datetime

logging.disable(logging.ERROR)

import main
import research

# Те же формулировки, что бот подставляет в handle_message
PREFIX = (
    f"Сегодня {datetime.now().strftime('%d.%m.%Y')}. На вопросы по теории, коду, математике "
    "и общим знаниям отвечай сам, без поиска. В интернет иди только если ответ зависит от "
    "текущего момента — новости, цены, последние версии, даты; тогда опирайся на найденное, "
    "а не на память. ТЫ ОБЯЗАН ОТВЕЧАТЬ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. ВОПРОС: "
)

CASES = [
    # (вопрос, должен ли искать)
    ("хайблс", False),
    ("асдфгх", False),
    ("Привет! Как дела?", False),
    ("Что такое горутина в Go?", False),
    ("Напиши функцию сортировки пузырьком на Go", False),
    ("Сколько будет 17 * 23?", False),
    ("Почему падает тест с nil map?", False),
    ("Объясни, как работает сборщик мусора", False),
    ("Как перевести 'hello world' на французский?", False),
    ("В чём разница между каналом и мьютексом?", False),
    ("Что нового вышло в Go в 2026 году?", True),
    ("Какая сейчас последняя версия Go?", True),
    ("Сколько стоит биткоин сегодня?", True),
    ("Найди в интернете, когда выйдет Go 1.28", True),
]


async def searches(provider_key: str, model_id: str, question: str):
    """True, если модель решила вызвать web_search. None — если API не ответил."""
    try:
        message = await main.call_provider_api_once(
            provider_key, model_id,
            [{"role": "user", "content": PREFIX + question}],
            tools=research.TOOLS,
        )
    except Exception as e:
        print(f"   API не ответил на {question!r}: {type(e).__name__}")
        return None
    return bool(message.get("tool_calls"))


async def main_():
    provider_key, model_id = main.get_current_model(0)
    print(f"модель: {model_id}\n")

    results = await asyncio.gather(*(searches(provider_key, model_id, q) for q, _ in CASES))

    failures, skipped = [], 0
    for (question, expected), got in zip(CASES, results):
        if got is None:
            skipped += 1
            continue
        if got != expected:
            failures.append((question, expected, got))
        mark = "ok" if got == expected else "!!"
        print(f"{mark} {question[:44]:46}"
              f"{'нужен поиск' if expected else 'без поиска':>13} -> "
              f"{'искал' if got else 'не искал'}")

    print()
    if skipped:
        print(f"пропущено из-за ошибок API: {skipped}")
    if failures:
        print(f"ПРОВАЛЕНО {len(failures)}:")
        for question, expected, got in failures:
            print(f"  {question!r}: ждали {'поиск' if expected else 'без поиска'}, "
                  f"получили {'поиск' if got else 'без поиска'}")
        return 1
    print(f"OK: все {len(CASES) - skipped} случаев прошли")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_()))
