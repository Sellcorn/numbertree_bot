"""Публикация длинных ответов на telegra.ph.

Telegram не пропускает сообщения длиннее 4096 символов, а страница Telegraph
открывается прямо в клиенте через Instant View — получается «одно сообщение со
ссылкой, которая разворачивается в окно с текстом».

API публичный и без ключей: аккаунт создаётся на лету и кэшируется на время
работы процесса (можно зафиксировать свой через TELEGRAPH_TOKEN).

Контент принимается ТОЛЬКО массивом узлов — на сырой HTML API отвечает
CONTENT_FORMAT_INVALID, поэтому здесь свой конвертер markdown → узлы.
Список поддерживаемых тегов у Telegraph короткий: таблиц в нём нет, поэтому
markdown-таблицы, как и в самом боте, уезжают в <pre>.
"""
import json
import logging
import os
import re

import httpx

import tables

logger = logging.getLogger(__name__)

API = "https://api.telegra.ph"
TIMEOUT = 30.0

_token_cache: str | None = None

INLINE_RE = re.compile(
    r"\*\*(?P<b>.+?)\*\*"
    r"|__(?P<b2>.+?)__"
    r"|(?<!\*)\*(?P<i>[^*\n]+?)\*(?!\*)"
    r"|`(?P<code>[^`\n]+?)`"
    r"|\[(?P<text>[^\]]+)\]\((?P<url>[^)\s]+)\)",
    re.S,
)


def _inline(text: str) -> list:
    """Разбирает подчёркивания/звёздочки/ссылки внутри строки в узлы Telegraph."""
    nodes, pos = [], 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            nodes.append(text[pos:m.start()])
        if m.group("b") or m.group("b2"):
            nodes.append({"tag": "b", "children": [m.group("b") or m.group("b2")]})
        elif m.group("i"):
            nodes.append({"tag": "i", "children": [m.group("i")]})
        elif m.group("code"):
            nodes.append({"tag": "code", "children": [m.group("code")]})
        else:
            nodes.append({
                "tag": "a",
                "attrs": {"href": m.group("url")},
                "children": [m.group("text")],
            })
        pos = m.end()
    if pos < len(text):
        nodes.append(text[pos:])
    return nodes or [text]


def markdown_to_nodes(md: str) -> list:
    """Переводит markdown в массив узлов Telegraph."""
    lines = (md or "").replace("\r\n", "\n").split("\n")
    nodes: list = []
    para: list[str] = []
    items: list[str] = []
    list_tag: str | None = None

    def flush_para():
        nonlocal para
        if para:
            nodes.append({"tag": "p", "children": _inline(" ".join(para).strip())})
            para = []

    def flush_list():
        nonlocal items, list_tag
        if items:
            nodes.append({
                "tag": list_tag or "ul",
                "children": [{"tag": "li", "children": _inline(x)} for x in items],
            })
            items, list_tag = [], None

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("```"):
            flush_para(); flush_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            nodes.append({"tag": "pre", "children": ["\n".join(buf)]})
            continue

        if stripped.startswith("|") and stripped.count("|") >= 2:
            # Таблиц у Telegraph нет — сохраняем как есть моноширинным блоком
            flush_para(); flush_list()
            buf = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i].strip())
                i += 1
            rendered = tables.render_markdown_table(buf, tables.TELEGRAPH_WIDTH)
            if rendered:
                nodes.append({"tag": "pre", "children": [rendered]})
            continue

        if not stripped:
            flush_para(); flush_list()
            i += 1
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush_para(); flush_list()
            nodes.append({"tag": "hr"})
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_para(); flush_list()
            tag = "h3" if len(heading.group(1)) <= 2 else "h4"
            nodes.append({"tag": tag, "children": _inline(heading.group(2))})
            i += 1
            continue

        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        if bullet:
            flush_para()
            if list_tag != "ul":
                flush_list()
                list_tag = "ul"
            items.append(bullet.group(1))
            i += 1
            continue

        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if numbered:
            flush_para()
            if list_tag != "ol":
                flush_list()
                list_tag = "ol"
            items.append(numbered.group(1))
            i += 1
            continue

        flush_list()
        para.append(stripped)
        i += 1

    flush_para()
    flush_list()
    return nodes


async def _get_token() -> str | None:
    """Токен аккаунта: из окружения либо разовое создание на процесс."""
    global _token_cache
    if _token_cache:
        return _token_cache
    env = os.getenv("TELEGRAPH_TOKEN")
    if env:
        _token_cache = env
        return _token_cache
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{API}/createAccount",
                data={"short_name": "numbertree", "author_name": "numbertree_bot"},
            )
            data = response.json()
        if data.get("ok"):
            _token_cache = data["result"]["access_token"]
            return _token_cache
        logger.error(f"telegraph: не удалось создать аккаунт: {data}")
    except Exception as e:
        logger.error(f"telegraph: createAccount упал: {e}")
    return None


async def publish(title: str, markdown_text: str,
                  author_name: str = "numbertree_bot") -> str | None:
    """Публикует markdown страницей и возвращает URL. None — если не вышло.

    Публикация не критична: вызывающий код должен уметь обойтись без ссылки.
    """
    token = await _get_token()
    if not token:
        return None

    nodes = markdown_to_nodes(markdown_text)
    if not nodes:
        return None

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(f"{API}/createPage", data={
                "access_token": token,
                "title": (title or "Ответ")[:256],
                "author_name": author_name,
                "content": json.dumps(nodes, ensure_ascii=False),
                "return_content": "false",
            })
            data = response.json()
        if data.get("ok"):
            return data["result"]["url"]
        logger.error(f"telegraph: createPage отказал: {data.get('error')}")
    except Exception as e:
        logger.error(f"telegraph: createPage упал: {e}")
    return None
