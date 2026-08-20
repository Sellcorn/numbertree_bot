"""Маршрутизация текста во время интервью.

Регрессия: пока висела сессия техсобеседования, перехватчик съедал весь чат.
Ключевые слова искались подстрокой, поэтому «удали» ловилось как «дал»
(следующий вопрос), «проследи» — как «след», «ответь» — как «ответ».
А всё, что не совпало, уходило на оценку как письменный ответ.

Запуск: python test_interview_routing.py
"""
import sys

from main import _parse_interview_command

# Команды: распознаются только целиком, регистр и пунктуация не мешают
COMMANDS = [
    ("дальше", "next"),
    ("Дальше!", "next"),
    ("  следующий  ", "next"),
    ("следующий вопрос", "next"),
    ("объясни", "explain"),
    ("Разбор.", "explain"),
    ("покажи ответ", "explain"),
    ("завершить", "finish"),
    ("стоп", "finish"),
    ("конец", "finish"),
]

# Обычные реплики: командой быть не должны ни при каких обстоятельствах
NOT_COMMANDS = [
    "удали лишние импорты",          # содержит «дал»
    "проследи, где утечка памяти",   # содержит «след»
    "исследуй проблему с дедлоком",  # содержит «след»
    "ответь кратко: что такое канал",  # содержит «ответ»
    "почему падает тест?",           # «почему» было слишком общим словом
    "напиши функцию сортировки на Go",
    "объясни, как работает GC",      # это просьба к чату, а не команда интервью
    "дай пример кода",
    "закрой соединение в defer",
    "как остановить горутину?",
    "привет",
    "",
]


def main():
    failures = []

    for text, expected in COMMANDS:
        got = _parse_interview_command(text)
        if got != expected:
            failures.append(f"  {text!r}: ожидал {expected!r}, получил {got!r}")

    for text in NOT_COMMANDS:
        got = _parse_interview_command(text)
        if got is not None:
            failures.append(f"  {text!r}: должно быть None, получил {got!r}")

    total = len(COMMANDS) + len(NOT_COMMANDS)
    if failures:
        print(f"ПРОВАЛЕНО {len(failures)} из {total}:")
        print("\n".join(failures))
        return 1

    print(f"OK: все {total} случаев прошли")
    return 0


if __name__ == "__main__":
    sys.exit(main())
