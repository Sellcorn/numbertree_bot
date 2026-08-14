import os
import asyncio
import logging
import re
from typing import AsyncGenerator

import httpx
import markdown
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_USERNAME = None  # Заполнится при старте


def md_to_tg_html(text: str) -> str:
    """Конвертирует markdown в HTML, понятный Telegram."""
    if not text:
        return ""
    # Создаём новый экземпляр каждый раз (Markdown stateful)
    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "nl2br",
            "sane_lists",
        ],
        extension_configs={
            "codehilite": {
                "use_pygments": False,
                "css_class": "code",
            }
        },
        output_format="html",
    )
    html = md.convert(text)
    html = re.sub(r"<pre><code class=\"language-(\w+)\">", r"<pre><code>", html)
    html = html.replace("<code class=\"language-\">", "<code>")
    html = re.sub(r"</code></pre>", "</code></pre>", html)
    html = html.replace("<p>", "").replace("</p>", "\n")
    html = html.replace("<br />", "\n").replace("<br>", "\n")
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def should_respond(update: Update) -> bool:
    """Проверяет, должен ли бот ответить на сообщение."""
    msg = update.message
    if not msg or not msg.text:
        return False

    # В личке всегда отвечаем
    if msg.chat.type == "private":
        return True

    # В группах: упоминание @username или ответ на сообщение бота
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


async def call_nvidia_api(messages: list[dict], stream: bool = True) -> AsyncGenerator[str, None]:
    """Yields content tokens from DeepSeek V4 Flash."""
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
        "���� Привет! Я бот на базе DeepSeek-R1 (NVIDIA).\n"
        "Задавай любой вопрос — я покажу процесс рассуждений, а потом выдам ответ."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not should_respond(update):
        return

    user_text = update.message.text
    chat_id = update.message.chat_id

    # Убираем @username из текста если есть
    if BOT_USERNAME:
        user_text = user_text.replace(f"@{BOT_USERNAME}", "").strip()

    thinking_msg = await update.message.reply_text("���� Думаю...")

    # DeepSeek V4 Flash: просим показать рассуждения в ответе
    user_prompt = (
        "Ты полезный AI-ассистент. "
        "Сначала напиши свои рассуждения под заголовком '## Рассуждения', "
        "а потом итоговый ответ под заголовком '## Ответ'.\n\n"
        f"Вопрос: {user_text}"
    )
    messages = [{"role": "user", "content": user_prompt}]

    all_parts = []
    in_reasoning = True

    async for token in call_nvidia_api(messages):
        all_parts.append(token)
        full_text = "".join(all_parts)

        # Проверяем переход к ответу
        if in_reasoning and "## Ответ" in full_text:
            in_reasoning = False
            await thinking_msg.edit_text("���� Рассуждаю...\n\n" + md_to_tg_html(full_text.split("## Ответ")[0]))
            await asyncio.sleep(0.3)
            continue

        if len(full_text) % 150 == 0:
            preview = md_to_tg_html(full_text[-2000:])
            prefix = "���� Думаю...\n\n" if in_reasoning else "���� Рассуждения и ответ:\n\n"
            await thinking_msg.edit_text(prefix + preview)

    full_text = "".join(all_parts).strip()

    # Разделяем на рассуждения и ответ
    if "## Ответ" in full_text:
        reasoning_part, answer_part = full_text.split("## Ответ", 1)
        reasoning_part = reasoning_part.replace("## Рассуждения", "").strip()
        answer_part = answer_part.strip()
    else:
        reasoning_part = ""
        answer_part = full_text

    reasoning_html = md_to_tg_html(reasoning_part) if reasoning_part else ""
    answer_html = md_to_tg_html(answer_part)

    if reasoning_html:
        await thinking_msg.edit_text(
            f"���� <b>Процесс рассуждений:</b>\n{reasoning_html}\n\n"
            f"��� <b>Ответ:</b>\n{answer_html}",
            parse_mode="HTML"
        )
    else:
        await thinking_msg.edit_text(f"��� <b>Ответ:</b>\n{answer_html}", parse_mode="HTML")


def main():
    global BOT_USERNAME
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Установите переменную окружения TELEGRAM_BOT_TOKEN")

    app = Application.builder().token(token).build()

    # Получаем username бота
    async def post_init(app):
        global BOT_USERNAME
        me = await app.bot.get_me()
        BOT_USERNAME = me.username
        logger.info(f"Бот @{BOT_USERNAME} запущен")

    app.post_init = post_init
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()