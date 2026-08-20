"""Веб-поиск для бота: Tavily + цикл tool calling.

Модуль намеренно ничего не знает о провайдерах LLM — вызов модели приходит
снаружи колбэком `llm_call`, поэтому research.py не импортирует main.py
(иначе получился бы циклический импорт).

Схема работы:
    1. Модели отдают описание инструмента `web_search` (TOOLS).
    2. Если она решает поискать — возвращает tool_calls вместо текста.
    3. Мы выполняем поиск, кладём результат в диалог и спрашиваем снова.
    4. Как только модель отвечает текстом, а не вызовом — цикл закончен.

Наружу отдаются обогащённые messages: их скармливают обычному стриминговому
вызову уже без tools, чтобы модель просто написала ответ.
"""
import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

MAX_ROUNDS = 3               # кругов «подумал → поискал»
MAX_RESULTS = 5              # источников на один запрос
MAX_CHARS_PER_SOURCE = 1500  # обрезка выжимки одного источника
MAX_CHARS_TOTAL = 8000       # потолок текста поиска на один круг
SEARCH_TIMEOUT = 30.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Ищет актуальную информацию в интернете. ОБЯЗАТЕЛЬНО используй для вопросов "
                "о новостях, версиях, релизах, ценах, датах и любых фактах, которые могли "
                "измениться после твоей отсечки знаний. Можно вызывать несколько раз подряд, "
                "уточняя формулировку, если первых результатов не хватило."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Поисковый запрос. Для технических тем формулируй по-английски — "
                            "выдача будет полнее."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def is_enabled() -> bool:
    """Поиск доступен только если задан ключ. Без него бот работает как раньше."""
    return bool(os.getenv("TAVILY_API_KEY"))


async def search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Ищет через Tavily. Возвращает [{title, url, content}, ...].

    Берём поле `content` — это выжимка по теме, уже очищенная от вёрстки.
    `raw_content` (сырая страница целиком, десятки килобайт) не запрашиваем:
    в контекст модели он всё равно не поместится.
    """
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY не задан")

    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
        "include_answer": False,
    }
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        response = await client.post(
            TAVILY_URL, headers={"Authorization": f"Bearer {key}"}, json=payload
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for item in data.get("results", []):
        content = (item.get("content") or "").strip()
        results.append({
            "title": (item.get("title") or "").strip(),
            "url": (item.get("url") or "").strip(),
            "content": content[:MAX_CHARS_PER_SOURCE],
        })
    return results


def format_results(query: str, results: list[dict]) -> str:
    """Складывает результаты в текст для модели, соблюдая потолок символов."""
    if not results:
        return f"По запросу «{query}» ничего не найдено."

    parts = [f"Результаты поиска по запросу «{query}»:"]
    total = 0
    for i, item in enumerate(results, 1):
        block = f"\n[{i}] {item['title']}\nURL: {item['url']}\n{item['content']}"
        if total + len(block) > MAX_CHARS_TOTAL:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def _extract_query(call: dict) -> str:
    """Достаёт query из вызова функции. Модель может прислать битый JSON."""
    arguments = (call.get("function") or {}).get("arguments") or "{}"
    if isinstance(arguments, dict):
        return str(arguments.get("query") or "").strip()
    try:
        return str(json.loads(arguments).get("query") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        logger.warning(f"research: не разобрал аргументы вызова: {arguments[:200]}")
        return ""


async def run_tool_loop(llm_call, messages: list[dict], *, max_rounds: int = MAX_ROUNDS,
                        on_progress=None) -> tuple[list[dict], list[dict]]:
    """Гоняет модель по кругу «решает → ищем → отдаём результат».

    llm_call(messages, tools) -> dict — нестриминговый вызов модели, возвращает
    message целиком (в нём может быть tool_calls).
    on_progress(text) — необязательный async-колбэк для показа прогресса.

    Возвращает (messages, sources): диалог, дополненный результатами поиска,
    и список использованных источников [{title, url}] без дублей.
    """
    messages = list(messages)
    sources: list[dict] = []
    seen_urls: set[str] = set()

    async def progress(text: str):
        if on_progress:
            try:
                await on_progress(text)
            except Exception:
                pass  # прогресс — украшение, из-за него ронять поиск нельзя

    for round_no in range(1, max_rounds + 1):
        message = await llm_call(messages, TOOLS)
        calls = (message or {}).get("tool_calls") or []

        if not calls:
            # Модель больше не хочет искать — она готова отвечать.
            break

        # Ответ модели с вызовами обязан попасть в историю до результатов,
        # иначе API не сопоставит tool_call_id.
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        })

        for call in calls:
            query = _extract_query(call)
            if not query:
                result_text = "Запрос пустой или не разобран. Сформулируй query заново."
            else:
                await progress(f"Ищу: {query}")
                try:
                    found = await search(query)
                    result_text = format_results(query, found)
                    for item in found:
                        if item["url"] and item["url"] not in seen_urls:
                            seen_urls.add(item["url"])
                            sources.append({"title": item["title"], "url": item["url"]})
                except Exception as e:
                    # Отдаём ошибку модели текстом: пусть попробует иначе
                    # или ответит по памяти, но не роняем весь диалог.
                    logger.error(f"research: поиск не удался ({query}): {e}")
                    result_text = f"Поиск по запросу «{query}» не удался: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": result_text,
            })

        if round_no == max_rounds:
            logger.info(f"research: достигнут потолок в {max_rounds} круга поиска")

    return messages, sources
