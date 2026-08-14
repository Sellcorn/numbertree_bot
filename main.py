import os
import asyncio
import logging
import re
from typing import AsyncGenerator

import httpx
import markdown
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
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

# Настройки провайдеров и моделей
PROVIDERS = {
    "nvidia": {
        "name": "NVIDIA",
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "api_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "models": {
            "deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash-0731",
            "nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",
            "llama-8b": "meta/llama-3.1-8b-instruct",
        },
        "default": "deepseek-v4-flash",
    },
    "groq": {
        "name": "Groq",
        "api_key": os.getenv("GROQ_API_KEY"),
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "models": {
            "llama-3.3-70b": "llama-3.3-70b-versatile",
            "llama-3.1-8b": "llama-3.1-8b-instant",
            "mixtral-8x7b": "mixtral-8x7b-32768",
            "gemma2-9b": "gemma2-9b-it",
        },
        "default": "llama-3.3-70b",
    },
}

# Пользовательские настройки: chat_id -> {provider, model}
USER_SETTINGS = {}
DEFAULT_PROVIDER = "nvidia"

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
    """Конвертирует markdown в HTML для Telegram. Также чистит лишний HTML."""
    if not text:
        return ""
    # Сначала экранируем HTML-сущности, потом конвертим markdown
    import html as html_module
    # Если текст уже содержит HTML-теги — вытащим текст
    if re.search(r'<[^>]+>', text):
        # Убираем HTML теги, оставляем содержимое
        text = re.sub(r'<[^>]+>', '', text)
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


def get_user_settings(chat_id: int) -> dict:
    """Получает настройки пользователя (провайдер, модель)."""
    return USER_SETTINGS.get(chat_id, {"provider": DEFAULT_PROVIDER})


def set_user_provider(chat_id: int, provider: str):
    if chat_id not in USER_SETTINGS:
        USER_SETTINGS[chat_id] = {"provider": DEFAULT_PROVIDER}
    USER_SETTINGS[chat_id]["provider"] = provider
    # Сбрасываем модель на дефолт для нового провайдера
    USER_SETTINGS[chat_id]["model"] = PROVIDERS[provider]["default"]


def set_user_model(chat_id: int, model_key: str):
    if chat_id not in USER_SETTINGS:
        USER_SETTINGS[chat_id] = {"provider": DEFAULT_PROVIDER}
    USER_SETTINGS[chat_id]["model"] = model_key


def get_current_model(chat_id: int) -> tuple[str, str]:
    """Возвращает (provider, model_id) для чата."""
    settings = get_user_settings(chat_id)
    provider = settings.get("provider", DEFAULT_PROVIDER)
    model_key = settings.get("model", PROVIDERS[provider]["default"])
    model_id = PROVIDERS[provider]["models"].get(model_key, PROVIDERS[provider]["default"])
    return provider, model_id


# Inline клавиатуры для меню
def build_main_menu(chat_id: int):
    """Главное меню: выбор провайдера."""
    settings = get_user_settings(chat_id)
    current = settings.get("provider", DEFAULT_PROVIDER)
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = []
    for key, prov in PROVIDERS.items():
        label = f"{'✅ ' if key == current else ''}{prov['name']}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"provider:{key}")])
    buttons.append([InlineKeyboardButton("🔧 Модели текущего провайдера", callback_data="models_menu")])
    return InlineKeyboardMarkup(buttons)


def build_models_menu(chat_id: int):
    """Меню выбора модели для текущего провайдера."""
    settings = get_user_settings(chat_id)
    provider_key = settings.get("provider", DEFAULT_PROVIDER)
    provider = PROVIDERS[provider_key]
    current_model = settings.get("model", provider["default"])
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = []
    for model_key, model_id in provider["models"].items():
        label = f"{'✅ ' if model_key == current_model else ''}{model_key}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"model:{model_key}")])
    buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


async def call_provider_api(provider_key: str, model_id: str, messages: list[dict], stream: bool = True) -> AsyncGenerator[tuple[str, dict], None]:
    """Универсальный вызов API для NVIDIA и Groq."""
    provider = PROVIDERS[provider_key]
    api_key = provider["api_key"]
    api_url = provider["api_url"]
    
    if not api_key:
        raise ValueError(f"API key not set for {provider_key}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": stream,
        "temperature": 0.6,
        "max_tokens": 4096,
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", api_url, headers=headers, json=payload) as response:
            if response.status_code in (429, 529, 503):
                raise httpx.HTTPStatusError(f"Rate limited", request=response.request, response=response)
            
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"{provider['name']} API error {response.status_code}: {error_text}")
                raise httpx.HTTPStatusError(f"API error: {response.status_code}", request=response.request, response=response)
            
            usage = {}
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if "usage" in chunk:
                            usage = chunk["usage"]
                        if content:
                            yield content, usage
                    except json.JSONDecodeError:
                        continue


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"{emoji('rocket')} <b>Бот готов к работе</b>\n\n"
        f"{emoji('brain')} Отвечаю на русском с рассуждениями\n"
        f"{emoji('gear')} Работаю в личке, по @username и по reply\n"
        f"{emoji('spark')} Помню контекст диалога\n"
        f"{emoji('code')} /menu — выбор модели и провайдера\n"
        f"{emoji('code')} /clear — сбросить память",
        parse_mode=ParseMode.HTML,
        reply_markup=build_main_menu(update.message.chat_id)
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


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает главное меню выбора провайдера/модели."""
    chat_id = update.message.chat_id
    settings = get_user_settings(chat_id)
    provider_key = settings.get("provider", DEFAULT_PROVIDER)
    model_key = settings.get("model", PROVIDERS[provider_key]["default"])
    provider_name = PROVIDERS[provider_key]["name"]
    model_name = model_key
    
    text = (
        f"{emoji('gear')} <b>Настройки модели</b>\n\n"
        f"Текущий провайдер: <b>{provider_name}</b>\n"
        f"Текущая модель: <b>{model_name}</b>\n\n"
        f"Выберите действие:"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=build_main_menu(chat_id))


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на инлайн-кнопки."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data = query.data
    
    if data == "main_menu":
        await query.edit_message_text(
            f"{emoji('gear')} <b>Настройки модели</b>\n\nВыберите провайдера:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_main_menu(chat_id)
        )
    elif data == "models_menu":
        await query.edit_message_text(
            f"{emoji('gear')} <b>Выбор модели</b>\n\nВыберите модель:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_models_menu(chat_id)
        )
    elif data.startswith("provider:"):
        provider_key = data.split(":")[1]
        if provider_key in PROVIDERS:
            set_user_provider(chat_id, provider_key)
            provider_name = PROVIDERS[provider_key]["name"]
            await query.edit_message_text(
                f"{emoji('check')} Провайдер изменён на <b>{provider_name}</b>\n\nВыберите модель:",
                parse_mode=ParseMode.HTML,
                reply_markup=build_models_menu(chat_id)
            )
    elif data.startswith("model:"):
        model_key = data.split(":")[1]
        settings = get_user_settings(chat_id)
        provider_key = settings.get("provider", DEFAULT_PROVIDER)
        if model_key in PROVIDERS[provider_key]["models"]:
            set_user_model(chat_id, model_key)
            await query.edit_message_text(
                f"{emoji('check')} Модель изменена на <b>{model_key}</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=build_models_menu(chat_id)
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

    # Получаем текущую модель пользователя
    provider_key, model_id = get_current_model(chat_id)
    provider_name = PROVIDERS[provider_key]["name"]

    # Строим сообщения с историей
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": f"Отвечай на русском. Рассуждай шаг за шагом. Используй markdown-форматирование (блоки кода, жирный, списки). Без HTML. Вопрос: {user_text}"})

    all_parts = []
    last_edit_len = 0
    start_time = asyncio.get_event_loop().time()
    usage = {}

    try:
        async for token, u in call_provider_api(provider_key, model_id, messages):
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
                        f"{emoji('thinking')} <b>Обрабатываю...</b> ⏱ <i>{elapsed}с</i> ({provider_name})\n\n"
                        f"<blockquote expandable>{preview}</blockquote>",
                        parse_mode=ParseMode.HTML
                    )
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"API call failed: {e}")
        await thinking_msg.edit_text(
            f"{emoji('error')} <b>Ошибка API</b>\n\n"
            f"<code>{str(e)[:500]}</code>\n\n"
            f"Попробуйте позже или смените модель через /menu.",
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
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CallbackQueryHandler(callback_handler))
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