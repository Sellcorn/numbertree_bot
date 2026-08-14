import os
import asyncio
import logging
import re
from typing import AsyncGenerator

import httpx
import markdown
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_USERNAME = None

# ============ CUSTOM EMOJI CONFIG ============
# Получите custom_emoji_id из пака https://t.me/addemoji/GameEmoji
# דרך: отправьте эмодзи боту @userinfobot → скопируйте custom_emoji_id
CUSTOM_EMOJI = {
    "thinking": 5226702984204797593,   # 🔄 — процесс思考
    "spark": 5463071033256848094,      # 🔝 — идея
    "brain": 5226639745106330551,      # 🧠 — ответ/мозг
    "gear": 5463423955014529788,       # 👌 — обработка
    "rocket": 5463345378587849154,     # 🙈 — запуск
    "check": 5462882007451185227,      # 🚫 — готово
    "answer": 5226639745106330551,     # 🧠 — ответ (тот же мозг)
    "code": 5229011542011299168,       # 👑 — код
    "warning": 5463121572137022242,    # 😂 — внимание
    "error": 5465262274031659421,      # 🥰 — ошибка
    "eye": 5228822494730797152,        # 👁 — глаз
    "like": 5465465194056525619,       # 👍 — лайк
}

# Fallback на обычные эмодзи если custom_id не заданы
def emoji(key: str) -> str:
    fallback = {
        "thinking": "🤔",
        "spark": "✨",
        "brain": "🧠",
        "gear": "⚙️",
        "rocket": "🚀",
        "check": "✅",
        "answer": "💬",
        "code": "💻",
        "warning": "⚠️",
        "error": "❌",
    }
    custom_id = CUSTOM_EMOJI.get(key)
    if custom_id:
        return f'<tg-emoji emoji-id="{custom_id}">{fallback[key]}</tg-emoji>'
    return fallback[key]


def md_to_html(text: str) -> str:
    """Конвертирует markdown в HTML для Telegram."""
    if not text:
        return ""
    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "nl2br", "sane_lists"],
        output_format="html",
    )
    html = md.convert(text)
    html = re.sub(r'<pre><code class="language-(\w+)">', r'<pre><code>', html)
    html = html.replace('<code class="language-">', '<code>')
    html = html.replace('<p>', '').replace('</p>', '\n')
    html = html.replace('<br />', '\n').replace('<br>', '\n')
    html = re.sub(r'\n{3,}', '\n\n', html)
    return html.strip()


def should_respond(update: Update) -> bool:
    msg = update.message
    if not msg or not msg.text:
        return False
    if msg.chat.type == "private":
        return True
    if BOT_USERNAME and f"@{BOT_USERNAME}" in msg.text:
        return True
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if msg.reply_to_message.from_user.username == BOT_USERNAME:
            return True
    return False


NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-mini-4b-instruct"

if not NVIDIA_API_KEY:
    raise ValueError("Установите переменную окружения NVIDIA_API_KEY")


async def call_nvidia_api(messages: list[dict], stream: bool = True) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", NVIDIA_API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"{emoji('rocket')} <b>Nemotron Mini 4B — готов к работе</b>\n\n"
        f"{emoji('brain')} Быстрые ответы с рассуждениями\n"
        f"{emoji('gear')} Работаю в личке, по @username и по reply\n"
        f"{emoji('spark')} Просто задайте вопрос",
        parse_mode=ParseMode.HTML
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not should_respond(update):
        return

    # Логируем custom_emoji_id если есть (для получения ID из GameEmoji)
    if update.message.entities:
        for entity in update.message.entities:
            if entity.type == "custom_emoji" and entity.custom_emoji_id:
                logger.info(f"CUSTOM_EMOJI_ID: {entity.custom_emoji_id} (char: '{update.message.text[entity.offset:entity.offset+entity.length]}')")

    user_text = update.message.text
    if BOT_USERNAME:
        user_text = user_text.replace(f"@{BOT_USERNAME}", "").strip()

    # Начальное сообщение с анимированным эмодзи
    thinking_msg = await update.message.reply_text(
        f"{emoji('thinking')} <b>Думаю...</b> ⏱ <i>0с</i>",
        parse_mode=ParseMode.HTML
    )

    user_prompt = f"Answer step by step. Question: {user_text}"
    messages = [{"role": "user", "content": user_prompt}]

    all_parts = []
    last_edit_len = 0
    start_time = asyncio.get_event_loop().time()

    async for token in call_nvidia_api(messages):
        all_parts.append(token)
        full_text = "".join(all_parts)

        # Обновляем каждые ~300 символов + таймер
        elapsed = int(asyncio.get_event_loop().time() - start_time)
        if len(full_text) - last_edit_len >= 300:
            last_edit_len = len(full_text)
            preview = full_text[-2500:]
            try:
                await thinking_msg.edit_text(
                    f"{emoji('thinking')} <b>Обрабатываю...</b> ⏱ <i>{elapsed}с</i>\n\n"
                    f"<blockquote expandable>{preview}</blockquote>",
                    parse_mode=ParseMode.HTML
                )
            except Exception:
                pass

    full_text = "".join(all_parts).strip()
    elapsed = int(asyncio.get_event_loop().time() - start_time)

    # Красивый финальный ответ с мозгом 🧠
    final_text = (
        f"{emoji('brain')} <b>Ответ</b> <i>({elapsed}с)</i>\n\n"
        f"{md_to_html(full_text)}"
    )

    try:
        await thinking_msg.edit_text(final_text, parse_mode=ParseMode.HTML)
    except Exception:
        await thinking_msg.edit_text(final_text)


def main():
    global BOT_USERNAME
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Установите переменную окружения TELEGRAM_BOT_TOKEN")

    app = Application.builder().token(token).build()

    async def post_init(app):
        global BOT_USERNAME
        me = await app.bot.get_me()
        BOT_USERNAME = me.username
        logger.info(f"Бот @{BOT_USERNAME} запущен")

    app.post_init = post_init
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(app.run_polling())


if __name__ == "__main__":
    main()