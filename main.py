import os
import asyncio
import logging
import re
from datetime import datetime
from typing import AsyncGenerator

import httpx
import markdown
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes, ChatMemberHandler
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

# Хранилище сообщений чатов для /summary, /judge, /context
# chat_id -> список сообщений [{user, text, time, user_name}]
CHAT_MESSAGES = {}
MAX_CHAT_MESSAGES = 100  # максимум сообщений на чат

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
    """Конвертирует markdown в HTML для Telegram. Агрессивно удаляет HTML."""
    if not text:
        return ""
    # СНАЧАЛА удаляем ВСЕ HTML-теги (модели часто выдают HTML вместо markdown)
    text = re.sub(r'<[^>]+>', '', text)
    # Затем конвертируем markdown в HTML
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
    # fallback на model_id, а не на ключ
    default_model_id = PROVIDERS[provider]["models"][PROVIDERS[provider]["default"]]
    model_id = PROVIDERS[provider]["models"].get(model_key, default_model_id)
    return provider, model_id


# Inline клавиатуры для меню
def build_main_menu(chat_id: int):
    """Главное меню: выбор провайдера и инструментов."""
    settings = get_user_settings(chat_id)
    current = settings.get("provider", DEFAULT_PROVIDER)
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = []
    for key, prov in PROVIDERS.items():
        label = f"{'✅ ' if key == current else ''}{prov['name']}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"provider:{key}")])
    buttons.append([InlineKeyboardButton("🔧 Модели текущего провайдера", callback_data="models_menu")])
    buttons.append([
        InlineKeyboardButton("🧠 Викторина", callback_data="quiz_menu"),
        InlineKeyboardButton("📊 Опрос", callback_data="poll_menu"),
    ])
    buttons.append([
        InlineKeyboardButton("💻 Помощь с кодом", callback_data="code_help"),
        InlineKeyboardButton("🎯 Задачи по коду", callback_data="code_tasks"),
    ])
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


def build_quiz_menu(chat_id: int):
    """Меню викторин."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = [
        [InlineKeyboardButton("🧠 Общие знания", callback_data="quiz:general")],
        [InlineKeyboardButton("💻 Программирование", callback_data="quiz:programming")],
        [InlineKeyboardButton("🌍 География", callback_data="quiz:geography")],
        [InlineKeyboardButton("🔬 Наука", callback_data="quiz:science")],
        [InlineKeyboardButton("🎲 Случайная", callback_data="quiz:random")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_poll_menu(chat_id: int):
    """Меню опросов."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = [
        [InlineKeyboardButton("💭 Мнение по теме", callback_data="poll:opinion")],
        [InlineKeyboardButton("📊 Сравнение вариантов", callback_data="poll:compare")],
        [InlineKeyboardButton("🎯 Приоритеты", callback_data="poll:priority")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_code_help_menu(chat_id: int):
    """Меню помощи с кодом."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = [
        [InlineKeyboardButton("🔍 Code Review", callback_data="code:review")],
        [InlineKeyboardButton("📖 Объяснить код", callback_data="code:explain")],
        [InlineKeyboardButton("🛠 Исправить баг", callback_data="code:fix")],
        [InlineKeyboardButton("⚡ Оптимизировать", callback_data="code:optimize")],
        [InlineKeyboardButton("📝 Написать с нуля", callback_data="code:write")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_code_tasks_menu(chat_id: int):
    """Меню задач по программированию."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = [
        [InlineKeyboardButton("🟢 Junior (Easy)", callback_data="task:easy")],
        [InlineKeyboardButton("🟡 Middle (Medium)", callback_data="task:medium")],
        [InlineKeyboardButton("🔴 Senior (Hard)", callback_data="task:hard")],
        [InlineKeyboardButton("🎲 Случайная", callback_data="task:random")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")],
    ]
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
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
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


async def welcome_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Приветствует новых участников в группе с кнопками меню."""
    if update.message and update.message.new_chat_members:
        for member in update.message.new_chat_members:
            if member.id == BOT_USERNAME:
                continue  # Не приветствуем самого бота
            await update.message.reply_text(
                f"{emoji('rocket')} <b>Привет, {member.mention_html()}!</b>\n\n"
                f"{emoji('brain')} Я ИИ-ассистент с выбором моделей (NVIDIA / Groq).\n"
                f"{emoji('gear')} Отвечаю по @username или reply.\n\n"
                f"Выберите модель и провайдера:",
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


async def context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает последние сообщения чата: /context [N]"""
    if update.message.chat.type not in ("group", "supergroup"):
        await update.message.reply_text(f"{emoji('warning')} Команда работает только в группах")
        return
    
    chat_id = update.message.chat_id
    messages = CHAT_MESSAGES.get(chat_id, [])
    if not messages:
        await update.message.reply_text(f"{emoji('warning')} Нет сохранённых сообщений")
        return
    
    n = 20
    if context.args:
        try:
            n = max(1, min(50, int(context.args[0])))
        except ValueError:
            pass
    
    recent = messages[-n:]
    lines = [f"{emoji('code')} <b>Последние {len(recent)} сообщений:</b>\n"]
    for msg in recent:
        time_str = msg["time"][11:16]
        lines.append(f"<code>{time_str}</code> <b>{msg['user_name']}</b>: {msg['text'][:200]}")
    
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Резюмирует последние сообщения: /summary [N]"""
    if update.message.chat.type not in ("group", "supergroup"):
        await update.message.reply_text(f"{emoji('warning')} Команда работает только в группах")
        return
    
    chat_id = update.message.chat_id
    messages = CHAT_MESSAGES.get(chat_id, [])
    if not messages:
        await update.message.reply_text(f"{emoji('warning')} Нет сохранённых сообщений для резюме")
        return
    
    n = 30
    if context.args:
        try:
            n = max(1, min(50, int(context.args[0])))
        except ValueError:
            pass
    
    recent = messages[-n:]
    dialog = "\n".join([f"{msg['user_name']}: {msg['text']}" for msg in recent])
    
    provider_key, model_id = get_current_model(update.message.chat_id)
    
    prompt = (
        f"Сделай краткое резюме диалога на русском языке. "
        f"Выдели главные темы, споры, решения. Формат: кратко, по пунктам, без воды.\n\n"
        f"Диалог:\n{dialog}"
    )
    
    thinking = await update.message.reply_text(f"{emoji('thinking')} <b>Анализирую чат...</b>", parse_mode=ParseMode.HTML)
    
    try:
        summary_parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": prompt}]):
            summary_parts.append(token)
        summary = "".join(summary_parts).strip()
        
        await thinking.edit_text(
            f"{emoji('brain')} <b>Резюме последних {len(recent)} сообщений:</b>\n\n"
            f"{md_to_html(summary)}",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await thinking.edit_text(f"{emoji('error')} Ошибка: {e}")


async def judge_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Даёт мнение по спору: /judge [вопрос] или реплай на спор"""
    if update.message.chat.type not in ("group", "supergroup"):
        await update.message.reply_text(f"{emoji('warning')} Команда работает только в группах")
        return
    
    chat_id = update.message.chat_id
    messages = CHAT_MESSAGES.get(chat_id, [])
    if not messages:
        await update.message.reply_text(f"{emoji('warning')} Нет сохранённых сообщений")
        return
    
    question = " ".join(context.args) if context.args else "Кто прав в этом споре? Дай объективное мнение."
    
    recent = messages[-20:]
    dialog = "\n".join([f"{msg['user_name']}: {msg['text']}" for msg in recent])
    
    provider_key, model_id = get_current_model(chat_id)
    
    prompt = (
        f"Проанализируй диалог и дай объективное мнение по вопросу: {question}\n"
        f"Будь беспристрастным, опирайся только на факты из чата. Ответ на русском, кратко.\n\n"
        f"Контекст:\n{dialog}"
    )
    
    thinking = await update.message.reply_text(f"{emoji('brain')} <b>Анализирую спор...</b>", parse_mode=ParseMode.HTML)
    
    try:
        opinion_parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": prompt}]):
            opinion_parts.append(token)
        opinion = "".join(opinion_parts).strip()
        
        await thinking.edit_text(
            f"{emoji('brain')} <b>Мнение по спору:</b>\n\n"
            f"{md_to_html(opinion)}",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await thinking.edit_text(f"{emoji('error')} Ошибка: {e}")


async def quiz_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, topic: str):
    """Запускает викторину на выбранную тему."""
    provider_key, model_id = get_current_model(update.message.chat_id)
    
    topics = {
        "general": "общие знания",
        "programming": "программирование",
        "geography": "география",
        "science": "наука",
        "random": "случайная тема",
    }
    topic_name = topics.get(topic, topic)
    
    prompt = (
        f"Создай 1 вопрос викторины на тему: {topic_name}. "
        f"Формат: вопрос, 4 варианта ответа (A, B, C, D), правильный ответ. "
        f"На русском языке."
    )
    
    msg = await update.message.reply_text(f"{emoji('brain')} <b>Генерирую викторину...</b>", parse_mode=ParseMode.HTML)
    
    try:
        parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": prompt}]):
            parts.append(token)
        text = "".join(parts).strip()
        
        # Парсим вопрос и варианты для создания опроса
        lines = text.split("\n")
        question = ""
        options = []
        correct_option = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not question:
                question = line.replace("Вопрос:", "").replace("Вопрос ", "").strip()
            elif line.startswith(("A)", "B)", "C)", "D)", "А)", "Б)", "В)", "Г)")):
                options.append(line[2:].strip())
                if "правиль" in line.lower() or "✓" in line or "✅" in line:
                    correct_option = len(options) - 1
        
        if len(options) >= 2:
            await msg.delete()
            await context.bot.send_poll(
                chat_id=update.message.chat_id,
                question=question or "Вопрос викторины",
                options=options[:4],
                type="quiz",
                correct_option_id=min(correct_option, len(options) - 1),
                explanation="Правильный ответ будет показан после голосования"
            )
        else:
            await msg.edit_text(f"{emoji('brain')} <b>Викторина:</b>\n\n{md_to_html(text)}", parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"{emoji('error')} Ошибка: {e}")


async def poll_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, poll_type: str):
    """Создаёт умный опрос."""
    provider_key, model_id = get_current_model(update.message.chat_id)
    
    types = {
        "opinion": "создай опрос для сбора мнений по теме, варианты: полностью согласен / скорее согласен / нейтрально / скорее нет / полностью не согласен",
        "compare": "создай опрос для сравнения 2-3 вариантов, укажи плюсы/минусы каждого",
        "priority": "создай опрос для определения приоритетов, варианты: высокий / средний / низкий приоритет",
    }
    
    prompt = (
        f"{types.get(poll_type, types['opinion'])}. "
        f"Верни: вопрос опроса и 3-5 вариантов ответа. На русском."
    )
    
    msg = await update.message.reply_text(f"{emoji('code')} <b>Создаю опрос...</b>", parse_mode=ParseMode.HTML)
    
    try:
        parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": prompt}]):
            parts.append(token)
        text = "".join(parts).strip()
        
        # Парсим для создания опроса
        lines = text.split("\n")
        question = lines[0] if lines else "Опрос"
        options = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "-", "•", "A)", "B)", "C)")):
                opt = line.lstrip("12345.-•ABC) ").strip()
                if opt:
                    options.append(opt)
        
        if len(options) >= 2:
            await msg.delete()
            await context.bot.send_poll(
                chat_id=update.message.chat_id,
                question=question,
                options=options[:10],
                type="regular",
                allows_multiple_answers=False
            )
        else:
            await msg.edit_text(f"{emoji('code')} <b>Опрос:</b>\n\n{md_to_html(text)}", parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"{emoji('error')} Ошибка: {e}")


async def code_help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, action: str):
    """Помощь с кодом."""
    if not update.message.text or len(update.message.text.split()) < 2:
        await update.message.reply_text(
            f"{emoji('warning')} Пришли код после команды или сделай реплай на сообщение с кодом.\n"
            f"Пример: <code>/code_review твой код здесь</code>",
            parse_mode=ParseMode.HTML
        )
        return
    
    code = update.message.text.split(" ", 1)[1]
    # Если это реплай, берём код из реплая
    if update.message.reply_to_message and update.message.reply_to_message.text:
        code = update.message.reply_to_message.text
    
    provider_key, model_id = get_current_model(update.message.chat_id)
    
    actions = {
        "review": "Сделай code review: найди баги, проблемы стиля, проблемы безопасности, предложи улучшения",
        "explain": "Объясни что делает этот код, пошагово, простым языком",
        "fix": "Найди и исправь все баги в коде, верни исправленную версию",
        "optimize": "Оптимизируй код: производительность, память, читаемость",
        "write": "Напиши код по описанию",
    }
    
    prompt = (
        f"{actions.get(action, actions['explain'])}.\n\n"
        f"Код:\n```\n{code}\n```\n\n"
        f"Ответ на русском, используй markdown для кода."
    )
    
    msg = await update.message.reply_text(f"{emoji('code')} <b>Анализирую код...</b>", parse_mode=ParseMode.HTML)
    
    try:
        parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": prompt}]):
            parts.append(token)
        result = "".join(parts).strip()
        
        await msg.edit_text(
            f"{emoji('code')} <b>Результат:</b>\n\n{md_to_html(result)}",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await msg.edit_text(f"{emoji('error')} Ошибка: {e}")


async def task_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, difficulty: str):
    """Генерирует задачу по программированию."""
    provider_key, model_id = get_current_model(update.message.chat_id)
    
    difficulties = {
        "easy": "Junior уровень: базовые алгоритмы, массивы, строки, циклы",
        "medium": "Middle уровень: структуры данных, DP, графы, алгоритмы сортировки",
        "hard": "Senior уровень: сложные алгоритмы, системное проектирование, конкурентность",
        "random": "случайная сложность",
    }
    
    prompt = (
        f"Создай задачу по программированию уровня: {difficulties.get(difficulty, difficulties['medium'])}.\n"
        f"Формат:\n"
        f"1. Название задачи\n"
        f"2. Условие (входные/выходные данные, ограничения)\n"
        f"3. Пример ввода/вывода\n"
        f"4. Подсказка (алгоритм)\n"
        f"5. Решение на Python с комментариями\n\n"
        f"На русском языке."
    )
    
    msg = await update.message.reply_text(f"{emoji('brain')} <b>Генерирую задачу...</b>", parse_mode=ParseMode.HTML)
    
    try:
        parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": prompt}]):
            parts.append(token)
        result = "".join(parts).strip()
        
        await msg.edit_text(
            f"{emoji('brain')} <b>Задача ({difficulty}):</b>\n\n{md_to_html(result)}",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await msg.edit_text(f"{emoji('error')} Ошибка: {e}")


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
        # Проверяем что модель существует у текущего провайдера
        if model_key in PROVIDERS[provider_key]["models"]:
            set_user_model(chat_id, model_key)
            await query.edit_message_text(
                f"{emoji('check')} Модель изменена на <b>{model_key}</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=build_models_menu(chat_id)
            )
        else:
            await query.answer(f"Модель {model_key} недоступна для этого провайдера", show_alert=True)
    elif data == "quiz_menu":
        await query.edit_message_text(
            f"{emoji('brain')} <b>Викторина</b>\n\nВыберите тему:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_quiz_menu(chat_id)
        )
    elif data == "poll_menu":
        await query.edit_message_text(
            f"{emoji('code')} <b>Умные опросы</b>\n\nВыберите тип:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_poll_menu(chat_id)
        )
    elif data == "code_help":
        await query.edit_message_text(
            f"{emoji('code')} <b>Помощь с кодом</b>\n\nВыберите действие:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_code_help_menu(chat_id)
        )
    elif data == "code_tasks":
        await query.edit_message_text(
            f"{emoji('brain')} <b>Задачи по программированию</b>\n\nВыберите уровень:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_code_tasks_menu(chat_id)
        )
    elif data.startswith("quiz:"):
        topic = data.split(":")[1]
        await query.answer()
        # Создаём фейковое сообщение для quiz_handler
        class FakeMessage:
            def __init__(self, chat_id, from_user):
                self.chat_id = chat_id
                self.from_user = from_user
                self.chat = type('obj', (object,), {'id': chat_id, 'type': 'private'})()
        fake_update = type('obj', (object,), {
            'message': FakeMessage(chat_id, query.from_user),
            'callback_query': query
        })()
        await quiz_handler(fake_update, context, topic)
    elif data.startswith("poll:"):
        poll_type = data.split(":")[1]
        await query.answer()
        class FakeMessage:
            def __init__(self, chat_id, from_user):
                self.chat_id = chat_id
                self.from_user = from_user
                self.chat = type('obj', (object,), {'id': chat_id, 'type': 'private'})()
        fake_update = type('obj', (object,), {
            'message': FakeMessage(chat_id, query.from_user),
            'callback_query': query
        })()
        await poll_handler(fake_update, context, poll_type)
    elif data.startswith("code:"):
        action = data.split(":")[1]
        await query.answer()
        class FakeMessage:
            def __init__(self, chat_id, from_user):
                self.chat_id = chat_id
                self.from_user = from_user
                self.chat = type('obj', (object,), {'id': chat_id, 'type': 'private'})()
        fake_update = type('obj', (object,), {
            'message': FakeMessage(chat_id, query.from_user),
            'callback_query': query
        })()
        await code_help_handler(fake_update, context, action)
    elif data.startswith("task:"):
        difficulty = data.split(":")[1]
        await query.answer()
        class FakeMessage:
            def __init__(self, chat_id, from_user):
                self.chat_id = chat_id
                self.from_user = from_user
                self.chat = type('obj', (object,), {'id': chat_id, 'type': 'private'})()
        fake_update = type('obj', (object,), {
            'message': FakeMessage(chat_id, query.from_user),
            'callback_query': query
        })()
        await task_handler(fake_update, context, difficulty)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Сохраняем сообщения групп для /summary, /judge, /context
    if update.message and update.message.text and update.message.chat.type in ("group", "supergroup"):
        chat_id = update.message.chat_id
        user = update.message.from_user
        CHAT_MESSAGES.setdefault(chat_id, []).append({
            "user_id": user.id,
            "user_name": user.first_name or user.username or str(user.id),
            "text": update.message.text,
            "time": datetime.now().isoformat()
        })
        # Ограничиваем размер
        if len(CHAT_MESSAGES[chat_id]) > MAX_CHAT_MESSAGES:
            CHAT_MESSAGES[chat_id] = CHAT_MESSAGES[chat_id][-MAX_CHAT_MESSAGES:]

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
    messages.append({"role": "user", "content": f"ТЫ ОБЯЗАН ОТВЕЧАТЬ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ HTML-ТЕГИ (<hr>, <strong>, <b>, <ol>, <ul>, <li>, <h1>, <h2>, <h3>, <p>, <div>, <span> И ДРУГИЕ). ИСПОЛЬЗУЙ ТОЛЬКО MARKDOWN: **жирный**, *курсив*, `код`, ```блоки кода```, - списки, 1. нумерованные списки, > цитаты. РАССУЖДАЙ ШАГ ЗА ШАГОМ. ВОПРОС: {user_text}"})

    all_parts = []
    last_edit_len = 0
    start_time = asyncio.get_event_loop().time()
    usage = {}

    # Фоновая задача для обновления таймера каждые 2 секунды
    stop_timer = asyncio.Event()
    last_timer_text = ""

    async def timer_updater():
        nonlocal last_timer_text
        while not stop_timer.is_set():
            await asyncio.sleep(2)
            if stop_timer.is_set():
                break
            elapsed = int(asyncio.get_event_loop().time() - start_time)
            preview = "".join(all_parts)[-2500:] if all_parts else ""
            timer_text = (
                f"{emoji('thinking')} <b>Обрабатываю...</b> ⏱ <i>{elapsed}с</i> ({provider_name})\n\n"
                f"<blockquote expandable>{preview}</blockquote>"
            )
            if timer_text != last_timer_text:
                try:
                    await thinking_msg.edit_text(timer_text, parse_mode=ParseMode.HTML)
                    last_timer_text = timer_text
                except Exception:
                    pass

    timer_task = asyncio.create_task(timer_updater())

    try:
        async for token, u in call_provider_api(provider_key, model_id, messages):
            all_parts.append(token)
            full_text = "".join(all_parts)
            if u:
                usage = u

            # Обновляем каждые ~300 символов (плюс таймер обновляется отдельно)
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
        stop_timer.set()
        await timer_task
        return
    finally:
        stop_timer.set()
        await timer_task

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

        # Устанавливаем меню бота (кнопки слева от поля ввода)
        from telegram import BotCommand
        await app.bot.set_my_commands([
            BotCommand("menu", "🤖 Выбрать модель и провайдера"),
            BotCommand("clear", "🗑 Очистить историю диалога"),
            BotCommand("start", "🚀 Перезапуск бота"),
        ])

    app.post_init = post_init
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear_history))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("context", context_command))
    app.add_handler(CommandHandler("summary", summary_command))
    app.add_handler(CommandHandler("judge", judge_command))
    # Новые команды для кнопок
    app.add_handler(CommandHandler("quiz", lambda u, c: quiz_handler(u, c, c.args[0] if c.args else "random")))
    app.add_handler(CommandHandler("poll", lambda u, c: poll_handler(u, c, c.args[0] if c.args else "opinion")))
    app.add_handler(CommandHandler("code", lambda u, c: code_help_handler(u, c, c.args[0] if c.args else "explain")))
    app.add_handler(CommandHandler("task", lambda u, c: task_handler(u, c, c.args[0] if c.args else "medium")))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_member))

    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(app.run_polling())


if __name__ == "__main__":
    main()