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

# Хранилище контекста диалогов
CONVERSATION_HISTORY = {}
MAX_HISTORY_PRIVATE = 10   # личка: полная память
MAX_HISTORY_GROUP = 3      # группы: короткая память на пользователя

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
MODEL = "deepseek-ai/deepseek-v4-flash-0731"

if not NVIDIA_API_KEY:
    raise ValueError("Установите переменную окружения NVIDIA_API_KEY")


async def call_nvidia_api(messages: list[dict], stream: bool = True) -> AsyncGenerator[tuple[str, dict], None]:
    """Yields (content_token, usage_dict). usage_dict is empty until final chunk."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "temperature": 0.6,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", NVIDIA_API_URL, headers=headers, json=payload) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"NVIDIA API error {response.status_code}: {error_text}")
                raise httpx.HTTPStatusError(f"NVIDIA API error: {response.status_code}", request=response.request, response=response)
            
            usage = {}
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
                        # usage приходит в последнем чанке
                        if "usage" in chunk:
                            usage = chunk["usage"]
                        if content:
                            yield content, usage
                    except json.JSONDecodeError:
                        continue


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"{emoji('rocket')} <b>DeepSeek V4 Flash — готов к работе</b>\n\n"
        f"{emoji('brain')} Быстрые ответы с рассуждениями\n"
        f"{emoji('gear')} Работаю в личке, по @username и по reply\n"
        f"{emoji('spark')} Помню контекст диалога (до {MAX_HISTORY} сообщений)\n"
        f"{emoji('code')} /clear — сбросить память",
        parse_mode=ParseMode.HTML
    )


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    is_private = update.message.chat.type == "private"
    history_key = chat_id if is_private else f"{chat_id}:{user_id}"

    if history_key in CONVERSATION_HISTORY:
        del CONVERSATION_HISTORY[history_key]
    scope = "чата" if is_private else "вашей истории в этом чате"
    await update.message.reply_text(
        f"{emoji('check')} <b>История {scope} очищена</b>",
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

    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    is_private = update.message.chat.type == "private"

    # Ключ истории: личка = chat_id, группа = chat_id:user_id
    history_key = chat_id if is_private else f"{chat_id}:{user_id}"
    max_history = MAX_HISTORY_PRIVATE if is_private else MAX_HISTORY_GROUP

    # Получаем историю диалога
    history = CONVERSATION_HISTORY.get(history_key, [])

    # Начальное сообщение с анимированным эмодзи
    thinking_msg = await update.message.reply_text(
        f"{emoji('thinking')} <b>Думаю...</b> ⏱ <i>0с</i>",
        parse_mode=ParseMode.HTML
    )

    # Строим сообщения с историей
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": f"Answer step by step. Question: {user_text}"})

    all_parts = []
    last_edit_len = 0
    start_time = asyncio.get_event_loop().time()
    usage = {}

    try:
        async for token, u in call_nvidia_api(messages):
            all_parts.append(token)
            full_text = "".join(all_parts)
            if u:
                usage = u

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
    except Exception as e:
        logger.error(f"NVIDIA API call failed: {e}")
        await thinking_msg.edit_text(
            f"{emoji('error')} <b>Ошибка API</b>\n\n"
            f"<code>{str(e)[:500]}</code>\n\n"
            f"Попробуйте позже или задайте другой вопрос.",
            parse_mode=ParseMode.HTML
        )
        return

    full_text = "".join(all_parts).strip()
    elapsed = int(asyncio.get_event_loop().time() - start_time)

    # Сохраняем в историю
    history.append({"user": user_text, "assistant": full_text})
    if len(history) > max_history:
        history.pop(0)
    CONVERSATION_HISTORY[history_key] = history

    # Информация о токенах
    token_info = ""
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        token_info = f"\n\n{emoji('code')} <b>Токены:</b> {prompt_tokens} + {completion_tokens} = {total_tokens}"

    # Красивый финальный ответ с мозгом 🧠
    final_text = (
        f"{emoji('brain')} <b>Ответ</b> <i>({elapsed}с)</i>\n\n"
        f"{md_to_html(full_text)}"
        f"{token_info}"
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
    app.add_handler(CommandHandler("clear", clear_history))
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