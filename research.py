"""Веб-поиск для бота: Tavily + цикл tool calling.

Модуль намеренно ничего не знает о провайдерах LLM — вызов модели приходит
снаружи колбэком `llm_call`, поэтому research.py не импортирует main.py
(иначе получился бы циклический импорт).

Схема работы:
    1. Модели отдают описание инструмента `web_search` (TOOLS).
    2. Если она решает поискать — возвращает tool_calls вместо текста.
    3. За один вызов модель присылает НЕСКОЛЬКО похожих формулировок запроса;
       они уходят в Tavily параллельно, результаты сливаются с дедупликацией
       по URL (близкие запросы дают сильно пересекающуюся выдачу).
    4. Как только модель отвечает текстом, а не вызовом — цикл закончен.

Наружу отдаются обогащённые messages: их скармливают обычному стриминговому
вызову уже без tools, чтобы модель просто написала ответ.
"""
import asyncio
import json
import logging
import os
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

MAX_ROUNDS = 3               # кругов «подумал → поискал»
MAX_QUERIES = 4              # формулировок за один вызов инструмента
MAX_RESULTS = 5              # источников на одну формулировку
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
                "измениться после твоей отсечки знаний. Передавай сразу НЕСКОЛЬКО разных "
                "формулировок одного и того же вопроса — они ищутся параллельно и дают "
                "более полную картину, чем одна. Можно вызывать повторно, если после "
                "первых результатов остались пробелы."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": MAX_QUERIES,
                        "description": (
                            f"От 2 до {MAX_QUERIES} разных формулировок одного вопроса: синонимы, "
                            "уточнения, узкие и широкие варианты. Для технических тем формулируй "
                            "по-английски — выдача будет полнее. Пример: "
                            '["Go 1.27 release notes", "Go 1.27 new features", '
                            '"golang 1.27 changelog"]'
                        ),
                    }
                },
                "required": ["queries"],
            },
        },
    }
]


def is_enabled() -> bool:
    """Поиск доступен только если задан ключ. Без него бот работает как раньше."""
    return bool(os.getenv("TAVILY_API_KEY"))


def domain_of(url: str) -> str:
    """Домен без www — для компактного показа в прогрессе."""
    try:
        host = urlparse(url).netloc
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return url[:40]


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


def format_results(queries: list[str], results: list[dict]) -> str:
    """Складывает результаты в текст для модели, соблюдая потолок символов."""
    shown = ", ".join(f"«{q}»" for q in queries)
    if not results:
        return f"По запросам {shown} ничего не найдено."

    parts = [f"Результаты поиска по запросам {shown}:"]
    total = 0
    for i, item in enumerate(results, 1):
        block = f"\n[{i}] {item['title']}\nURL: {item['url']}\n{item['content']}"
        if total + len(block) > MAX_CHARS_TOTAL:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def _extract_queries(call: dict) -> list[str]:
    """Достаёт список запросов из вызова функции.

    Модель может прислать битый JSON, строку вместо массива или старое поле
    `query` — принимаем всё это, лишь бы не терять круг поиска впустую.
    """
    arguments = (call.get("function") or {}).get("arguments") or "{}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            logger.warning(f"research: не разобрал аргументы вызова: {arguments[:200]}")
            return []
    if not isinstance(arguments, dict):
        return []

    raw = arguments.get("queries") or arguments.get("query") or []
    if isinstance(raw, str):
        raw = [raw]

    queries, seen = [], set()
    for item in raw:
        text = str(item).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            queries.append(text)
    return queries[:MAX_QUERIES]


async def _search_all(queries: list[str]) -> tuple[list[dict], list[str]]:
    """Ищет по всем формулировкам параллельно, сливает выдачу без дублей.

    Возвращает (уникальные результаты, тексты ошибок по неудавшимся запросам).
    """
    batches = await asyncio.gather(
        *(search(q) for q in queries), return_exceptions=True
    )

    merged, errors, seen = [], [], set()
    for query, batch in zip(queries, batches):
        if isinstance(batch, Exception):
            logger.error(f"research: поиск не удался ({query}): {batch}")
            errors.append(f"Запрос «{query}» не удался: {batch}")
            continue
        for item in batch:
            url = item.get("url")
            if url and url not in seen:
                seen.add(url)
                merged.append(item)
    return merged, errors


async def run_tool_loop(llm_call, messages: list[dict], *, max_rounds: int = MAX_ROUNDS,
                        on_progress=None) -> tuple[list[dict], list[dict]]:
    """Гоняет модель по кругу «решает → ищем → отдаём результат».

    llm_call(messages, tools) -> dict — нестриминговый вызов модели, возвращает
    message целиком (в нём может быть tool_calls).
    on_progress(line) — необязательный async-колбэк: строка прогресса для показа
    пользователю (запрос или найденный сайт).

    Возвращает (messages, sources): диалог, дополненный результатами поиска,
    и список использованных источников [{title, url}] без дублей.
    """
    messages = list(messages)
    sources: list[dict] = []
    seen_urls: set[str] = set()
    searched: set[str] = set()  # формулировки, уже отправленные в поиск

    async def progress(line: str):
        if on_progress:
            try:
                await on_progress(line)
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
            queries = _extract_queries(call)
            # Между кругами модель охотно повторяет удачную формулировку —
            # второй раз её искать смысла нет, только квота и время.
            fresh = [q for q in queries if q.lower() not in searched]

            if not queries:
                result_text = "Запросы пустые или не разобраны. Сформулируй queries заново."
            elif not fresh:
                result_text = (
                    "Эти запросы уже искали: "
                    + ", ".join(f"«{q}»" for q in queries)
                    + ". Сформулируй принципиально иначе или отвечай по собранным результатам."
                )
            else:
                searched.update(q.lower() for q in fresh)
                for query in fresh:
                    await progress(f"🔍 {query}")

                found, errors = await _search_all(fresh)

                for item in found:
                    if item["url"] and item["url"] not in seen_urls:
                        seen_urls.add(item["url"])
                        sources.append({"title": item["title"], "url": item["url"]})
                        await progress(f"📄 {domain_of(item['url'])} — {item['title'][:60]}")

                result_text = format_results(fresh, found)
                if errors:
                    # Ошибки отдаём модели текстом: пусть учтёт или переформулирует,
                    # но весь круг из-за одного упавшего запроса не теряем.
                    result_text += "\n\n" + "\n".join(errors)

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": result_text,
            })

        if round_no == max_rounds:
            logger.info(f"research: достигнут потолок в {max_rounds} круга поиска")

    return messages, sources
