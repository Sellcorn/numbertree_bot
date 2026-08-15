import os
import asyncio
import json
import logging
import random
import re
from datetime import datetime
from typing import AsyncGenerator

import httpx
import markdown
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, PollAnswerHandler, filters, ContextTypes, ChatMemberHandler
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
    "qwen": {
        "name": "Qwen",
        "api_key": os.getenv("QWEN_API_KEY"),
        "api_url": "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
        "models": {
            "qwen3.6-flash": "qwen3.6-flash",
        },
        "default": "qwen3.6-flash",
    },
}

# Пользовательские настройки: chat_id -> {provider, model}
USER_SETTINGS = {}
DEFAULT_PROVIDER = "qwen"

# ============ ROADMAP (конфигурация уровней и стека) ============
# Читается из roadmap.json рядом с ботом. Содержит:
#   - "languages": каталог языков, у каждого — свой путь (уровни от нуля)
#   - "templates": готовые шаблоны стека (набор языков)
# Пользователь выбирает шаблон или собирает свой стек из языков — это определяет
# активный список уровней (active_levels). Настройки и прогресс хранятся в progress.json.
ROADMAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roadmap.json")
PROGRESS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.json")
DEFAULT_ROADMAP = {
    "title": "Путь разработчика",
    "description": "Базовый конфиг. Отредактируйте roadmap.json.",
    "languages": {},
    "templates": [],
}
ROADMAP = DEFAULT_ROADMAP


def load_roadmap():
    """Загружает roadmap.json в глобальную переменную ROADMAP."""
    global ROADMAP
    try:
        with open(ROADMAP_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if data and isinstance(data, dict) and isinstance(data.get("languages"), dict):
            ROADMAP = data
            return True
    except Exception as e:
        logger.error(f"Не удалось загрузить roadmap.json: {e}")
    return False


load_roadmap()

# ============ НАСТРОЙКИ И ПРОГРЕСС ПОЛЬЗОВАТЕЛЯ (персистентно в progress.json) ============
# Структура: chat_id(str) -> {
#   "template": str|None          — id выбранного готового шаблона
#   "languages": [lang_id, ...]   — языки своего стека (если без шаблона)
#   "level_index": int            — текущий уровень в активном списке
#   "done_topics": [str, ...]     — пройденные темы
# }
# В памяти под дублируется для скорости, persist в progress.json при изменениях.
PROGRESS_LOCK = asyncio.Lock()
USER_CONFIG = {}  # chat_id -> настройки стека (без скорости, хранится в файле)

# Загружается при старте в USER_CONFIG
def _load_user_config():
    global USER_CONFIG
    try:
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            USER_CONFIG = data
    except FileNotFoundError:
        USER_CONFIG = {}
    except Exception as e:
        USER_CONFIG = {}
        logger.error(f"Не удалось загрузить progress.json: {e}")


def _load_user_config_sync():
    _load_user_config()


# Загружаем сохранённые настройки при старте (синхронно, до запуска loop)
_load_user_config()


async def _save_user_config():
    """Сохраняет USER_CONFIG в progress.json (атомарно, под локом)."""
    async with PROGRESS_LOCK:
        tmp = PROGRESS_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(USER_CONFIG, f, ensure_ascii=False, indent=2)
            os.replace(tmp, PROGRESS_PATH)
        except Exception as e:
            logger.error(f"Не удалось сохранить progress.json: {e}")


def _get_user_settings(chat_id: int) -> dict:
    sid = str(chat_id)
    if sid not in USER_CONFIG or not isinstance(USER_CONFIG[sid], dict):
        USER_CONFIG[sid] = {"template": None, "languages": [], "level_index": 0, "done_topics": []}
    s = USER_CONFIG[sid]
    s.setdefault("template", None)
    s.setdefault("languages", [])
    s.setdefault("level_index", 0)
    s.setdefault("done_topics", [])
    return s


def get_progress(chat_id: int) -> dict:
    """Настройки+прогресс пользователя (персистентные)."""
    return _get_user_settings(chat_id)


# ============ АКТИВНЫЙ СПИСОК УРОВНЕЙ (исходя из выбранного шаблона/языков) ============
def _lang_levels(lang_id: str) -> list:
    return list(ROADMAP.get("languages", {}).get(lang_id, {}).get("levels", []))


def _template_levels(tpl_id: str) -> list:
    """Уровни готового шаблона = конкатенация уровней его языков (в порядке из шаблона)."""
    tpl = None
    for t in ROADMAP.get("templates", []):
        if t.get("id") == tpl_id:
            tpl = t
            break
    if not tpl:
        return []
    out = []
    for lang_id in tpl.get("languages", []):
        out.extend(_lang_levels(lang_id))
    return out


def active_levels(chat_id: int) -> list:
    """Список уровней для пользователя: из выбранного шаблона, либо из его языков.
    Если ничего не выбрано — все уровни всех языков каталога (демо-режим)."""
    s = _get_user_settings(chat_id)
    if s.get("template"):
        lvl = _template_levels(s["template"])
        if lvl:
            return lvl
    langs = s.get("languages") or []
    out = []
    for lang_id in langs:
        out.extend(_lang_levels(lang_id))
    if out:
        return out
    # demo: все языки
    demo = []
    for lang_id in ROADMAP.get("languages", {}):
        demo.extend(_lang_levels(lang_id))
    return demo


def get_level(chat_id: int, index: int) -> dict | None:
    levels = active_levels(chat_id)
    if 0 <= index < len(levels):
        return levels[index]
    return None


def normalize_level_index(chat_id: int, index: int) -> int:
    levels = active_levels(chat_id)
    if not levels:
        return 0
    return max(0, min(index, len(levels) - 1))


def level_count(chat_id: int) -> int:
    return len(active_levels(chat_id))


def topics_of(level: dict) -> list[str]:
    focus = level.get("focus", "")
    return [t.strip() for t in focus.split(",") if t.strip()]


def level_topics_done(chat_id: int, level_index: int) -> list[str]:
    """Список пройденных тем пользователя для данного уровня."""
    level = get_level(chat_id, level_index)
    if not level:
        return []
    tops = topics_of(level)
    prog = get_progress(chat_id)
    key = f"lvl{level_index}"
    done_keys = set(prog.get("done_topics", []))
    return [t for t in tops if f"{key}#{tops.index(t)}" in done_keys]


async def set_user_template(chat_id: int, tpl_id: str):
    """Выбирает готовый шаблон стека, сбрасывая свой набор языков."""
    s = _get_user_settings(chat_id)
    s["template"] = tpl_id
    s["languages"] = []
    s["level_index"] = 0
    s["done_topics"] = []
    await _save_user_config()


async def toggle_user_language(chat_id: int, lang_id: str, on: bool):
    """Включает/выключает язык в пользовательском стеке."""
    s = _get_user_settings(chat_id)
    s["languages"] = [l for l in s["languages"] if l != lang_id]
    if on:
        s["languages"].append(lang_id)
        s["template"] = None
    if not s["languages"]:
        s["template"] = None
    s["level_index"] = 0
    s["done_topics"] = []
    await _save_user_config()


async def reset_user_stack(chat_id: int):
    s = _get_user_settings(chat_id)
    s["template"] = None
    s["languages"] = []
    s["level_index"] = 0
    s["done_topics"] = []
    await _save_user_config()


async def set_level_index(chat_id: int, index: int):
    s = _get_user_settings(chat_id)
    s["level_index"] = max(0, min(index, level_count(chat_id) - 1))
    await _save_user_config()


# Времчивое хранилище сгенерированных задач/викторин для экспорта:
# chat_id -> { "tasks": [ {level_title, item} ], "quizzes": [ {level_title, item} ] }
ROADMAP_GENERATED = {}

# ============ INTERVIEW (техсобеседование: база Go-вопросов) ============
# Курируемая база вопросов в interview.json (варианты ответов MC + открытые,
# с объяснением). Активные сессии: chat_id -> {"index": int, "category": str|None,
# "queue": [idx,...], "mode": "mc"|"open"}. "queue" — порядок вопросов.
INTERVIEW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interview.json")
DEFAULT_INTERVIEW = {"title": "Go · техсобеседование", "description": "", "categories": {}, "questions": []}
INTERVIEW = DEFAULT_INTERVIEW
INTERVIEW_SESSIONS = {}
# Сопоставление полл_id -> {"chat_id", "correct", "category"} для интервью-теста (MC)
INTERVIEW_QUIZ = {}


def load_interview():
    global INTERVIEW
    try:
        with open(INTERVIEW_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if data and isinstance(data, dict) and isinstance(data.get("questions"), list):
            INTERVIEW = data
            return True
    except Exception as e:
        logger.error(f"Не удалось загрузить interview.json: {e}")
    return False


load_interview()


def _interview_indexed() -> list:
    return [q for q in INTERVIEW.get("questions", []) if isinstance(q, dict)]


def _interview_pick(chat_id: int, category: str | None) -> dict | None:
    """Возвращает следующий вопрос по категории (или случайно) и индекс.
    Очередь хранит глобальные индексы в базе вопросов."""
    all_qs = _interview_indexed()
    if not all_qs:
        return None
    if category == "all":
        qs = all_qs
    elif category:
        qs = [q for q in all_qs if q.get("category") == category]
        if not qs:
            return None
    else:
        qs = all_qs
    s = INTERVIEW_SESSIONS.setdefault(chat_id, {"last_idx": -1, "queue": [], "score": 0, "total": 0, "category": None, "ready": False})
    if s.get("category") != category:
        # сменилась категория — пересобираем очередь
        s["queue"] = []
        s["ready"] = False
        s["category"] = category
    if not s.get("ready"):
        # сохраняем глобальные индексы один раз на длительность сессии
        s["queue"] = [all_qs.index(q) for q in qs]
        random.shuffle(s["queue"])
        s["ready"] = True
    if not s["queue"]:
        INTERVIEW_SESSIONS.pop(chat_id, None)
        return None
    idx = s["queue"].pop()
    s["last_idx"] = idx
    question = all_qs[idx]
    return question


def _interview_question_text(q: dict, pos: int, total: int) -> str:
    cat_name = INTERVIEW.get("categories", {}).get(q.get("category", ""), q.get("category", ''))
    mode = "📊 Тест (варианты)" if q.get("type") == "mc" else "✍️ Напиши сам"
    return (f"🎙️ <b>Техсобеседование</b> · {cat_name}\n"
            f"{emoji('spark')} Вопрос {pos}/{total} · {mode}")


def _interview_stats(chat_id: int) -> tuple[int, int]:
    sess = INTERVIEW_SESSIONS.get(chat_id)
    if not sess:
        return 0, 0
    return sess.get("score", 0), sess.get("total", 0)


def build_interview_menu(chat_id: int):
    """Меню техсобеса: выбрать категорию, случайный вопрос, все вопросы."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    cat = INTERVIEW.get("categories", {})
    buttons = []
    for cid, cname in cat.items():
        buttons.append([InlineKeyboardButton(cname, callback_data=f"intv:start:{cid}")])
    buttons.append([InlineKeyboardButton("🎲 Случайный вопрос", callback_data="intv:start:")])
    buttons.append([InlineKeyboardButton("🧩 Всё подряд (перемешать)", callback_data="intv:start:all")])
    score, total = _interview_stats(chat_id)
    buttons.append([InlineKeyboardButton("🏁 Закрыть сессию", callback_data="intv:close")])
    buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")])
    header = (
        f"{emoji('microphone')} <b>{INTERVIEW.get('title', 'Техсобеседование')}</b>\n\n"
        f"{INTERVIEW.get('description', '')}\n"
        f"📊 Счёт текущей сессии: <b>{score}/{total}</b>\n"
        f"Выберите категорию или режим:"
    )
    return header, InlineKeyboardMarkup(buttons)


async def interview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Техсобеседование по Go: /interview"""
    chat_id = update.message.chat_id
    text, markup = build_interview_menu(chat_id)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=markup)


async def _interview_start(chat_id: int, category: str):
    """Запускает/продолжает сессию интервью и выдаёт следующий вопрос в нужном формате."""
    from telegram import Bot
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    q = _interview_pick(chat_id, category or None)
    if not q:
        await bot.send_message(chat_id, f"{emoji('warning')} Вопросы кончились. Начните заново (/interview).", parse_mode=ParseMode.HTML)
        return
    total = len(_interview_indexed())
    sess = INTERVIEW_SESSIONS.get(chat_id)
    pos = sess.get("total", 0) + 1
    header = _interview_question_text(q, pos, total)
    qtext = f"<b>{q.get('q', '')}</b>"

    if q.get("type") == "mc":
        options = [str(o) for o in q.get("options", []) if str(o)]
        correct = int(q.get("correct", -1))
        if len(options) >= 2 and 0 <= correct < len(options):
            expl = str(q.get("explanation", "") or "")[:200]
            poll_kwargs = {
                "chat_id": chat_id,
                "question": q.get("q", "")[:255],
                "options": options[:10],
                "type": "quiz",
                "correct_option_id": correct,
                "is_anonymous": False,
                "open_period": 120,
            }
            if expl:
                poll_kwargs["explanation"] = expl
            sent = await bot.send_poll(**poll_kwargs)
            INTERVIEW_QUIZ[sent.poll.id] = {"chat_id": chat_id, "correct": correct}
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            nav = InlineKeyboardMarkup([
                [InlineKeyboardButton("⏭️ Следующий вопрос", callback_data="intv:next")],
                [InlineKeyboardButton("🏁 Завершить", callback_data="intv:finish")],
            ])
            await bot.send_message(
                chat_id,
                f"{header}\n\n<i>Полл выше — ответьте в нём. Счёт придёт после выбора.</i>",
                parse_mode=ParseMode.HTML,
                reply_markup=nav,
            )
            return
        # fallback: MC без валидных вариантов -> открытый
    # открытый вопрос
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    buttons = [
        [InlineKeyboardButton("👁️ Показать разбор", callback_data="intv:answer")],
        [InlineKeyboardButton("⏭️ Следующий вопрос", callback_data="intv:next")],
        [InlineKeyboardButton("🏁 Завершить", callback_data="intv:finish")],
    ]
    await bot.send_message(
        chat_id,
        f"{header}\n\n{qtext}\n\n{emoji('pencil')} Напишите ответ своими словами — как ответили бы на собеседовании. Затем нажмите «Показать разбор».",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def _interview_show_answer(chat_id: int):
    """Показывает разбор текущего (последнего) вопроса."""
    from telegram import Bot
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    sess = INTERVIEW_SESSIONS.get(chat_id)
    idx = sess.get("last_idx", -1) if sess else -1
    qs = _interview_indexed()
    if idx < 0 or idx >= len(qs):
        await bot.send_message(chat_id, f"{emoji('warning')} Сначала задайте вопрос.", parse_mode=ParseMode.HTML)
        return
    q = qs[idx]
    lines = [f"{emoji('brain')} <b>Разбор</b>\n",
             f"{emoji('question')} {q.get('q','')}"]
    if q.get("type") == "mc" and q.get("options"):
        correct = int(q.get("correct", -1))
        lines.append("\n<b>Варианты:</b>")
        for i, o in enumerate(q["options"]):
            mark = "✅" if i == correct else ""
            lines.append(f"{i+1}. {o} {mark}")
    answer = q.get("answer") or next((o for i, o in enumerate(q.get("options", [])) if i == int(q.get("correct", -1))), "")
    if answer:
        lines.append(f"\n<b>Правильный ответ:</b> {answer}")
    if q.get("explanation"):
        lines.append(f"\n<b>Как правильно ответить:</b>\n{q['explanation']}")
    buttons = [
        [InlineKeyboardButton("⏭️ Следующий вопрос", callback_data="intv:next")],
        [InlineKeyboardButton("🏁 Завершить", callback_data="intv:finish")],
    ]
    await bot.send_message(chat_id, "\n".join(lines), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(buttons))


async def _interview_next(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Следующий вопрос той же сессии."""
    sess = INTERVIEW_SESSIONS.get(chat_id)
    category = sess.get("category") if sess else None
    await _interview_start(chat_id, category or "")


def _interview_finish(chat_id: int) -> str:
    sess = INTERVIEW_SESSIONS.get(chat_id)
    score, total = (sess.get("score", 0), sess.get("total", 0)) if sess else (0, 0)
    INTERVIEW_SESSIONS.pop(chat_id, None)
    INTERVIEW_QUIZ.clear()
    return (f"{emoji('microphone')} <b>Техсобеседование завершено!</b>\n\n"
            f"📊 Счёт: <b>{score}/{total}</b>\n"
            f"Начать заново — /interview")


async def _interview_poll_result(update: Update, iinfo: dict):
    """Обрабатывает ответ на MC-полл интервью."""
    pa = update.poll_answer
    chat_id = iinfo["chat_id"]
    correct = iinfo["correct"]
    chosen = pa.option_ids[0] if pa.option_ids else -1
    is_right = (chosen == correct)
    sess = INTERVIEW_SESSIONS.setdefault(chat_id, {"score": 0, "total": 0, "last_idx": -1})
    sess["total"] += 1
    if is_right:
        sess["score"] += 1
        verdict = f"{emoji('check')} <b>Верно!</b>"
    else:
        verdict = f"{emoji('warning')} <b>Неверно.</b> Правильный — вариант {correct+1}."
    from telegram import Bot
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    buttons = [[InlineKeyboardButton("⏭️ Следующий вопрос", callback_data="intv:next")],
               [InlineKeyboardButton("🏁 Завершить", callback_data="intv:finish")]]
    await bot.send_message(
        chat_id,
        f"{verdict}\n📊 Счёт: <b>{sess['score']}/{sess['total']}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(buttons),
    )


def add_generated_task(chat_id: int, level_title: str, item: dict):
    ROADMAP_GENERATED.setdefault(chat_id, {"tasks": [], "quizzes": []})
    ROADMAP_GENERATED[chat_id]["tasks"].append({"level": level_title, "item": item})


def add_generated_quiz(chat_id: int, level_title: str, item: dict):
    ROADMAP_GENERATED.setdefault(chat_id, {"tasks": [], "quizzes": []})
    ROADMAP_GENERATED[chat_id]["quizzes"].append({"level": level_title, "item": item})


# ============ EXPORT (экспорт для Obsidian) ============
def build_export_markdown(chat_id: int) -> str:
    """Собирает весь roadmap + сгенерированные задачи/викторины в один markdown."""
    lines = []
    lines.append(f"# {ROADMAP.get('title', 'Roadmap')}\n")
    lines.append(f"{ROADMAP.get('description', '')}\n")
    prog = get_progress(chat_id)

    for idx, level in enumerate(active_levels(chat_id)):
        title = level.get("title", f"Уровень {idx + 1}")
        lines.append(f"## {idx + 1}. {title}\n")
        lines.append(f"- **Уровень**: {level.get('difficulty', 'medium')}")
        lines.append(f"- **Стек**: {level.get('stack', '-')}")
        lines.append(f"- **Языки**: {', '.join(level.get('languages', []))}")
        lines.append(f"- **Фокус**: {level.get('focus', '')}\n")

        tops = topics_of(level)
        if tops:
            lines.append("### Темы для изучения")
            done = level_topics_done(chat_id, idx)
            done_keys = [t for t in done]
            for t in tops:
                mark = "[x]" if t in done_keys else "[ ]"
                lines.append(f"- {mark} {t}")
            lines.append("")

        authored = level.get("tasks", [])
        if authored:
            lines.append("### Задачи уровня")
            for t in authored:
                lines.append(f"**{t.get('name', 'Задача')}** ({t.get('lang', '-')}): {t.get('desc', '')}\n")
            lines.append("")

    gen = ROADMAP_GENERATED.get(chat_id, {"tasks": [], "quizzes": []})
    if gen["tasks"]:
        lines.append("---\n\n## Задачи, сгенерированные на этой сессии\n")
        for g in gen["tasks"]:
            lines.append(f"### {g['level']}\n")
            lines.append(f"{md_to_md(g['item'].get('problem', g['item'].get('name', '')))}")
            lines.append("")
    if gen["quizzes"]:
        lines.append("---\n\n## Викторины, сгенерированные на этой сессии\n")
        for g in gen["quizzes"]:
            item = g["item"]
            lines.append(f"### {g['level']}\n")
            lines.append(f"**Вопрос:** {item.get('question', '')}\n")
            for i, opt in enumerate(item.get("options", [])):
                marker = "✅" if i == item.get("correct", -1) else " "
                lines.append(f"{i + 1}. {opt} {marker}")
            if item.get("explanation"):
                lines.append(f"*Почему:* {item['explanation']}")
            lines.append("")

    lines.append(f"\n\n_Экспортировано {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    return "\n".join(lines).strip()


def md_to_md(text: str) -> str:
    """Помогает нормализовать LLM-текст: убираем HTML теги, оставляем markdown."""
    return re.sub(r'<[^>]+>', '', text) if text else ''


# Хранилище сообщений чатов для /summary, /judge, /context
# chat_id -> список сообщений [{user, text, time, user_name}]
CHAT_MESSAGES = {}
MAX_CHAT_MESSAGES = 100  # максимум сообщений на чат

# ============ РЕЖИМ ОБУЧЕНИЯ (теория -> тест -> проверка знаний) ============
# Активная сессия обучения: chat_id -> {"level_index": int, "theory": str}
LEARN_SESSIONS = {}
# Сопоставление quiz-poll_id -> {"correct": int, "chat_id": int, "level_index": int}
# нужно, чтобы проверить знание по ответу на полл-викторину.
LEARN_QUIZ = {}

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
        "target": "🎯",
        "microphone": "🎙️",
        "arrow": "➡️",
        "pencil": "✍️",
        "question": "❓",
    }
    custom_id = CUSTOM_EMOJI.get(key)
    if custom_id:
        return f'<tg-emoji emoji-id="{custom_id}">{fallback[key]}</tg-emoji>'
    return fallback[key]


# HTML-теги, которые поддерживает Telegram (см. Bot API, раздел «HTML style»).
# НЕ поддерживает: table, th, td, tr, ul, ol, li, h1-h6, hr, div, span (кроме tg-spoiler).
_TG_TAGS = ("b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
            "a", "code", "pre", "blockquote", "tg-spoiler", "tg-emoji", "br")


_TABLE_ROW_RE = re.compile(r'^\s*\|?(?:\s*[^|]+\s*\|\s*)+\s*\|?\s*$')
_TABLE_SEP_RE = re.compile(r'^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$')


def _convert_table_blocks(text: str) -> str:
    """Находит markdown-таблицы и превращает их в Telegram-совместимый
    monospace-блок <pre> (Telegram не поддерживает <table>, <th>, <td>)."""
    lines = text.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Таблица начинается с строки, содержащей '|', за которой идёт разделитель
        if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            block = [line, lines[i + 1]]
            j = i + 2
            while j < n and "|" in lines[j] and _TABLE_ROW_RE.match(lines[j]):
                block.append(lines[j])
                j += 1
            rows = []
            for bl in block:
                if _TABLE_SEP_RE.match(bl):
                    continue
                cells = [c.strip() for c in bl.strip().strip("|").split("|")]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                out.append("\n".join(["<pre>"] + rows + ["</pre>"]))
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def md_to_html(text: str) -> str:
    """Конвертирует markdown в HTML для Telegram. Гарантирует, что на выходе
    только теги, поддерживаемые Telegram (заголовки→жирный текст, таблицы→<pre>)."""
    if not text:
        return ""
    # СНАЧАЛА удаляем ВСЕ HTML-теги (модели часто выдают HTML вместо markdown)
    text = re.sub(r'<[^>]+>', '', text)
    # Markdown-таблицы --> в <pre> (Telegram не умеет <table>)
    text = _convert_table_blocks(text)
    # Заголовки --> жирный текст (h1-h6 не поддерживаются)
    text = re.sub(r'(?m)^(#{1,6})\s+(.+)$', lambda m: f"<b>{m.group(2)}</b>", text)
    # HR --> разделитель
    text = re.sub(r'(?m)^\s*(---+|\*\*\*+)\s*$', '────────────────', text)
    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "nl2br", "sane_lists"],
        output_format="html",
    )
    html = md.convert(text)
    html = re.sub(r'<pre><code class="language-(\w+)">', r'<pre><code>', html)
    html = html.replace('<code class="language-">', '<code>')
    html = re.sub(r'<p>', '', html)
    html = html.replace('</p>', '\n')
    # ul/ol/li --> списки простым текстом (Telegram не поддерживает эти теги)
    html = re.sub(r'<ul>|<ol>', '\n', html)
    html = re.sub(r'</ul>|</ol>', '\n', html)
    html = re.sub(r'<li>', '• ', html)
    html = re.sub(r'</li>', '\n', html)
    # Остальные теги --> в допустимый вид
    html = re.sub(r'<h[1-6][^>]*>', '<b>', html)
    html = re.sub(r'</h[1-6]>', '</b>', html)
    html = re.sub(r'<hr[^>]*>', '\n────────────────\n', html)
    html = html.replace('<br />', '\n').replace('<br>', '\n')
    # Удаляем любые не-поддерживаемые теги, оставляя их содержимое
    html = re.sub(r'</?(?:table|thead|tbody|tr|td|th|div|span|font|img|figure)\b[^>]*>', '', html)
    html = html.replace('<tg-spoiler>', '<span class="tg-spoiler">').replace('</tg-spoiler>', '</span>')
    # Схлопываем избыточные переносы
    html = re.sub(r' *\n *', '\n', html)
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
    buttons.append([InlineKeyboardButton("🗺️ Roadmap по уровням", callback_data="roadmap_menu")])
    buttons.append([InlineKeyboardButton("🧰 Стек и языки (/stack)", callback_data="stack_menu")])
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


def build_roadmap_menu(chat_id: int):
    """Меню уровней (roadmap): текущий уровень и продвижение по стеку/языкам."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    levels = active_levels(chat_id)
    prog = get_progress(chat_id)
    buttons = []
    cur = prog.get("level_index", 0)
    for idx, level in enumerate(levels):
        tops = topics_of(level)
        done = len(level_topics_done(chat_id, idx))
        total = len(tops)
        mark = "📍" if idx == cur else ("✅" if done >= total and total > 0 else "⬜")
        label = f"{mark} {level.get('title', f'Уровень {idx+1}')} ({done}/{total})" if total else f"{mark} {level.get('title', 'Уровень')}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"road:level:{idx}")])
    buttons.append([InlineKeyboardButton("📤 Экспортировать всё (.md)", callback_data="road:export_all")])
    buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


def build_stack_menu(chat_id: int):
    """Меню выбора стека: готовый шаблон или свой набор языков."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    s = _get_user_settings(chat_id)
    cur_template = s.get("template")
    cur_langs = set(s.get("languages", []))

    lines = []
    lines.append(f"{emoji('rocket')} <b>Стек и языки</b>\n")
    if cur_template:
        for t in ROADMAP.get("templates", []):
            if t.get("id") == cur_template:
                lines.append(f"{emoji('check')} Шаблон: <b>{t.get('icon', '')} {t.get('name', cur_template)}</b>")
                break
    elif cur_langs:
        names = [ROADMAP["languages"][l].get("name", l) for l in cur_langs if l in ROADMAP.get("languages", {})]
        lines.append(f"{emoji('check')} Свой стек: <b>{', '.join(names) or '—'}</b>")
    else:
        lines.append(f"{emoji('gear')} Пока выбран демо-режим (все языки).")
    lines.append(f"{emoji('target')} Уровней в активном пути: <b>{level_count(chat_id)}</b>\n")

    buttons = []
    buttons.append([InlineKeyboardButton("🎲 Готовый шаблон:", callback_data="stack:none")])
    for t in ROADMAP.get("templates", []):
        mark = "✅ " if t.get("id") == cur_template else ""
        buttons.append([InlineKeyboardButton(f"{mark}{t.get('icon', '')} {t.get('name')}", callback_data=f"stack:template:{t.get('id')}")])
    buttons.append([InlineKeyboardButton("⚙️ Собрать свой стек:", callback_data="stack:none")])
    for lang_id, lang in ROADMAP.get("languages", {}).items():
        mark = "✅ " if lang_id in cur_langs else "  "
        buttons.append([InlineKeyboardButton(f"{mark}{lang.get('icon', '')} {lang.get('name')}", callback_data=f"stack:lang:{lang_id}")])
    buttons.append([InlineKeyboardButton("🗑️ Сбросить стек", callback_data="stack:reset")])
    buttons.append([InlineKeyboardButton("🗺️ Перейти к уровням", callback_data="roadmap_menu")])
    buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")])
    text = "\n".join(lines)
    return text, InlineKeyboardMarkup(buttons)


def build_roadmap_level_menu(chat_id: int, index: int):
    """Меню конкретного уровня: задача, викторина, темы, продвижение вперёд."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    index = normalize_level_index(chat_id, index)
    level = get_level(chat_id, index)
    if not level:
        return build_roadmap_menu(chat_id)
    tops = topics_of(level)
    done = level_topics_done(chat_id, index)
    done_keys = set(done)
    # next undone topic
    next_topic = next((t for t in tops if t not in done_keys), None)

    buttons = []
    line = []
    if len(tops) > 0:
        if next_topic:
            line.append(InlineKeyboardButton(f"✅ Отметить тему «{next_topic[:40]}»", callback_data=f"road:done:{index}"))
        buttons.append(line)
    buttons.append([InlineKeyboardButton("🎓 Режим обучения: теория + тест", callback_data=f"road:learn:{index}")])
    buttons.append([InlineKeyboardButton(f"💻 Задача по стеку ({', '.join(level.get('languages', [])[:2])})", callback_data=f"road:task:{index}")])
    buttons.append([InlineKeyboardButton(f"🧠 Викторина по языку", callback_data=f"road:quiz:{index}")])
    buttons.append([InlineKeyboardButton(f"📤 Экспорт уровня (.md)", callback_data=f"road:export:{index}")])
    # следующ. уровень
    next_index = index + 1
    if next_index < level_count(chat_id):
        buttons.append([InlineKeyboardButton("➡️ Следующий уровень", callback_data=f"road:next:{index}")])
    else:
        buttons.append([InlineKeyboardButton("🏁 Дошёл до конца", callback_data="road:finish")])
    buttons.append([InlineKeyboardButton("⬅️ К списку уровней", callback_data="road:menu")])
    return InlineKeyboardMarkup(buttons)


def build_roadmap_level_text(chat_id: int, index: int) -> str:
    """Текст описания уровня с прогрессом по темам."""
    index = normalize_level_index(chat_id, index)
    level = get_level(chat_id, index)
    if not level:
        return f"{emoji('warning')} Уровень не найден."
    tops = topics_of(level)
    done = level_topics_done(chat_id, index)
    done_keys = set(done)
    lines = [f"{emoji('rocket')} <b>{level.get('title', f'Уровень {index+1}')}</b>"]
    lines.append(f"{emoji('gear')} Уровень: <b>{level.get('difficulty', 'medium')}</b>")
    lines.append(f"{emoji('code')} Стек: {level.get('stack', '-')}")
    lines.append(f"{emoji('spark')} Языки: {', '.join(level.get('languages', []))}")
    if level.get('focus'):
        lines.append(f"{emoji('brain')} Фокус: {level['focus']}")
    if tops:
        lines.append(f"\n{emoji('check')} <b>Темы ({len(done)}/{len(tops)}):</b>")
        for i, t in enumerate(tops):
            mark = "✅" if t in done_keys else "⬜"
            lines.append(f"{mark} {t}")
    next_index = index + 1
    if next_index < level_count(chat_id):
        lines.append(f"\nСледующий: <b>{get_level(chat_id, next_index).get('title', '')}</b>")
    else:
        lines.append("\n🏁 Это последний уровень.")
    return "\n".join(lines)


async def call_provider_api(provider_key: str, model_id: str, messages: list[dict], stream: bool = True, temperature: float = 0.6) -> AsyncGenerator[tuple[str, dict], None]:
    """Универсальный вызов API провайдера."""
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
        "temperature": temperature,
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
        f"{emoji('gear')} Работаю в личке, по @numbertree_bot и по reply\n"
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
                f"{emoji('brain')} Я ИИ-ассистент на базе Qwen 3.6 Flash.\n"
                f"{emoji('gear')} Отвечаю по @numbertree_bot или reply.\n\n"
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


async def quiz_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает викторину на выбранную тему: /quiz [topic]"""
    topic = context.args[0] if context.args else "random"
    await _run_quiz(update.message.chat_id, context, topic)


async def poll_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Создаёт умный опрос: /poll [type]"""
    poll_type = context.args[0] if context.args else "opinion"
    await _run_poll(update.message.chat_id, context, poll_type)


async def code_help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Помощь с кодом: /code [action] [code] или реплай на сообщение с кодом"""
    if not update.message.text or len(update.message.text.split()) < 2:
        await update.message.reply_text(
            f"{emoji('warning')} Пришли код после команды или сделай реплай на сообщение с кодом.\n"
            f"Пример: <code>/code review твой код здесь</code>",
            parse_mode=ParseMode.HTML
        )
        return
    
    action = context.args[0] if context.args else "explain"
    
    # Получаем код: из аргументов команды или из реплая
    if len(context.args) > 1:
        code = " ".join(context.args[1:])
    elif update.message.reply_to_message and update.message.reply_to_message.text:
        code = update.message.reply_to_message.text
    else:
        await update.message.reply_text(
            f"{emoji('warning')} Пришли код после команды или сделай реплай на сообщение с кодом.",
            parse_mode=ParseMode.HTML
        )
        return
    
    await _run_code_help(update.message.chat_id, context, action, code)


async def roadmap_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает меню уровней: /roadmap"""
    chat_id = update.message.chat_id
    await update.message.reply_text(
        f"{emoji('rocket')} <b>Roadmap по стеку и языкам</b>\n\n"
        f"{ROADMAP.get('description', '')}",
        parse_mode=ParseMode.HTML,
        reply_markup=build_roadmap_menu(chat_id)
    )


async def stack_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Меню выбора стека и языков: /stack"""
    chat_id = update.message.chat_id
    text, markup = build_stack_menu(chat_id)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=markup)


async def stack_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок меню /stack: шаблон, язык, сброс."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data = query.data
    payload = data.removeprefix("stack:")
    if payload == "reset":
        await reset_user_stack(chat_id)
    elif payload.startswith("template:"):
        await set_user_template(chat_id, payload.split(":", 1)[1])
    elif payload.startswith("lang:"):
        lang_id = payload.split(":", 1)[1]
        cur = set(_get_user_settings(chat_id).get("languages", []))
        await toggle_user_language(chat_id, lang_id, lang_id not in cur)
    elif payload == "none":
        pass
    text, markup = build_stack_menu(chat_id)
    await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=markup)


async def _interview_callback(update: Update):
    """Обработчик кнопок меню техсобеседования."""
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data
    await query.answer()
    if data == "intv:start:":
        await _interview_start(chat_id, "")
    elif data.startswith("intv:start:"):
        cat = data.split(":", 2)[2]
        await _interview_start(chat_id, cat)
    elif data == "intv:next":
        await _interview_next(chat_id, None)
    elif data == "intv:answer":
        await _interview_show_answer(chat_id)
    elif data in ("intv:finish", "intv:close"):
        msg = _interview_finish(chat_id)
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        await query.edit_message_text(msg, parse_mode=ParseMode.HTML,
                                      reply_markup=InlineKeyboardMarkup(
                                          [[InlineKeyboardButton("🎙️ Начать заново", callback_data="intv:start:")],
                                           [InlineKeyboardButton("⬅️ Главное меню", callback_data="main_menu")]]))


async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Режим обучения по уровню roadmap: /learn [index]"""
    index = 0
    chat_id = update.message.chat_id
    if context.args:
        try:
            index = int(context.args[0]) - 1
        except ValueError:
            index = 0
    index = normalize_level_index(chat_id, index)
    await update.message.reply_text(
        building_learn_start(chat_id, index),
        parse_mode=ParseMode.HTML
    )


def building_learn_start(chat_id: int, index: int) -> str:
    """Промежуточное сообщение перед запуском обучения."""
    level = get_level(chat_id, index)
    if not level:
        return f"{emoji('warning')} Уровень не найден."
    return (f"{emoji('spark')} Начинаю обучение по уровню <b>{level.get('title', 'Уровень')}</b>.\n"
            f"{emoji('gear')} Сначала — теория, затем тест.")


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Экспортирует roadmap + сгенерированные задачи/викторины в .md для Obsidian: /export"""
    chat_id = update.message.chat_id
    md = build_export_markdown(chat_id)
    if not md:
        await update.message.reply_text(f"{emoji('warning')} Пока нечего экспортировать.")
        return
    from io import BytesIO
    filename = f"roadmap_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    bio = BytesIO(md.encode("utf-8"))
    bio.name = filename
    await update.message.reply_document(
        document=bio,
        caption=f"{emoji('check')} <b>Экспорт для Obsidian</b>\n{txt_fname(filename)}",
        parse_mode=ParseMode.HTML
    )


def txt_fname(fname: str) -> str:
    return f"Файл: <code>{fname}</code>"


def build_export_with_level(chat_id: int, index: int) -> str:
    """Экспорт одного уровня в markdown."""
    level = get_level(chat_id, index)
    if not level:
        return ""
    lines = []
    lines.append(f"# {index + 1}. {level.get('title', f'Уровень {index+1}')}")
    lines.append(f"- **Уровень**: {level.get('difficulty', 'medium')}")
    lines.append(f"- **Стек**: {level.get('stack', '-')}")
    lines.append(f"- **Языки**: {', '.join(level.get('languages', []))}")
    lines.append(f"- **Фокус**: {level.get('focus', '')}\n")
    tops = topics_of(level)
    if tops:
        lines.append("### Темы")
        done = level_topics_done(chat_id, index)
        done_keys = set(done)
        for t in tops:
            mark = "[x]" if t in done_keys else "[ ]"
            lines.append(f"- {mark} {t}")
    authored = level.get("tasks", [])
    if authored:
        lines.append("\n### Задачи")
        for t in authored:
            lines.append(f"**{t.get('name','Задача')}** ({t.get('lang','-')}): {t.get('desc','')}")
    for g in ROADMAP_GENERATED.get(chat_id, {"tasks": [], "quizzes": []}).get("tasks", []):
        if g.get("level") == level.get("title"):
            lines.append(f"\n**Сгенерировано:**\n{g['item'].get('problem', '')}")
    return "\n".join(lines)


async def roadmap_export_level(chat_id: int, index: int):
    """Отправляет уровень как .md файл."""
    from io import BytesIO
    from telegram import Bot
    md = build_export_with_level(chat_id, index)
    if not md:
        return
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    bio = BytesIO(md.encode("utf-8"))
    bio.name = f"level_{index + 1}.md"
    await bot.send_document(chat_id=chat_id, document=bio, caption=f"{emoji('check')} Экспорт уровня {index + 1}")


async def roadmap_export_all(chat_id: int):
    """Отправляет полный roadmap как .md файл."""
    from io import BytesIO
    from telegram import Bot
    md = build_export_markdown(chat_id)
    if not md:
        return
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    bio = BytesIO(md.encode("utf-8"))
    bio.name = "roadmap.md"
    await bot.send_document(chat_id=chat_id, document=bio, caption=f"{emoji('check')} Экспорт всего roadmap (.md)")


def mark_topic_done(chat_id: int, index: int) -> tuple[str, int]:
    """Отмечает следующую неотмеченную тему уровня. Возвращает (задача, прогресс)."""
    index = normalize_level_index(chat_id, index)
    level = get_level(chat_id, index)
    if not level:
        return "Уровень не найден", 0
    tops = topics_of(level)
    prog = get_progress(chat_id)
    done = set(prog.get("done_topics", []))
    key_prefix = f"lvl{index}#"
    next_i = next((i for i, t in enumerate(tops) if f"{key_prefix}{i}" not in done), None)
    if next_i is None:
        return "Все темы этого уровня уже отмечены. Переходите на следующий.", len(done)
    done.add(f"{key_prefix}{next_i}")
    prog["done_topics"] = sorted(done)
    return f"Отмечено: {tops[next_i]}", len(done)


async def _run_learn(chat_id: int, context: ContextTypes.DEFAULT_TYPE, index: int):
    """Режим обучения: отправляет теорию по уровню roadmap, затем тест (викторину),
    и по ответу на полл оценивает знание."""
    index = normalize_level_index(chat_id, index)
    level = get_level(chat_id, index)
    if not level:
        from telegram import Bot
        bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        await bot.send_message(chat_id, f"{emoji('warning')} Уровень не найден.", parse_mode=ParseMode.HTML)
        return

    provider_key, model_id = get_current_model(chat_id)
    langs = ", ".join(level.get("languages", []))
    stack = level.get("stack", "-")
    focus = level.get("focus", "")

    theory_prompt = (
        f"Ты — преподаватель. Составь КРАТКИЙ и структурированный учебный конспект по уровню «{level.get('title', '')}».\n"
        f"Стек: {stack}. Языки: {langs}. Фокус: {focus}.\n"
        f"Требования:\n"
        f"- Пиши на русском, без воды, без HTML-тегов.\n"
        f"- Используй markdown: **жирный**, списки -, ```код```.\n"
        f"- Разделы: 1) ключевые понятия, 2) примеры кода, 3) частые ошибки.\n"
        f"- Объём ~20-30 строк. Это теория перед тестом — дай самое важное."
    )

    from telegram import Bot
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    msg = await bot.send_message(chat_id, f"{emoji('brain')} <b>Готовлю теорию по уровню...</b>", parse_mode=ParseMode.HTML)

    try:
        parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": theory_prompt}], temperature=0.4):
            parts.append(token)
        theory = "".join(parts).strip()

        LEARN_SESSIONS[chat_id] = {"level_index": index, "theory": theory}
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"{emoji('spark')} <b>Теория · {level.get('title', 'Уровень')}</b>\n\n{md_to_html(theory)}\n\n"
                 f"{emoji('gear')} Дальше — тест по этому уровню. Промежуточный счёт будет приходить после каждого ответа.",
            parse_mode=ParseMode.HTML
        )
        # Затем запускаем тест-викторину по этому уровню
        await _run_quiz(chat_id, context, "programming", level, learn_session=True)
    except Exception as e:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"{emoji('error')} Ошибка обучения: {e}"
        )


async def _poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверяет ответ пользователя на учебную викторину и сообщает результат."""
    pa = update.poll_answer
    # Сопоставление фото poll_id -> данные для интервью-теста
    iinfo = INTERVIEW_QUIZ.get(pa.poll_id)
    if iinfo:
        await _interview_poll_result(update, iinfo)
        return

    info = LEARN_QUIZ.get(pa.poll_id)
    if not info:
        return
    chat_id = info["chat_id"]
    correct = info["correct"]
    chosen = pa.option_ids[0] if pa.option_ids else -1
    is_right = (chosen == correct)
    # накапливаем счёт в сессии обучения
    sess = LEARN_SESSIONS.get(chat_id)
    score = 1 if is_right else 0
    if sess:
        sess.setdefault("score", 0)
        sess.setdefault("total", 0)
        sess["total"] += 1
        sess["score"] += score

    from telegram import Bot
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    if is_right:
        verdict = f"{emoji('check')} <b>Верно!</b>"
    else:
        verdict = f"{emoji('warning')} <b>Неверно.</b> Правильный ответ — вариант {correct + 1}."

    track = ""
    if sess and sess.get("total"):
        track = f"\n\n📊 Счёт сессии: <b>{sess['score']}/{sess['total']}</b>"
    try:
        await bot.send_message(chat_id, f"{verdict}{track}", parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.warning(f"poll answer send failed: {e}")


# ===== Внутренние функции для команд и callback =====

def _extract_json(text: str):
    """Достаёт JSON-объект из ответа LLM (игнорирует markdown-обёртки и лишний текст)."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _parse_quiz(text: str):
    """Парсит JSON викторины от LLM. Возвращает None, если ответ невалидный."""
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None
    question = str(data.get("question", "")).strip()[:300]
    options = [str(o).strip()[:100] for o in data.get("options", []) if str(o).strip()]
    options = options[:10]
    if not question or len(options) < 2:
        return None
    try:
        correct = int(data.get("correct", -1))
    except (TypeError, ValueError):
        return None
    if not 0 <= correct < len(options):
        return None
    explanation = str(data.get("explanation", "") or "").strip()[:200]
    return {"question": question, "options": options, "correct": correct, "explanation": explanation}


def _parse_poll(text: str):
    """Парсит JSON опроса от LLM. Возвращает None, если ответ невалидный."""
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None
    question = str(data.get("question", "")).strip()[:300]
    options = [str(o).strip()[:100] for o in data.get("options", []) if str(o).strip()]
    options = options[:10]
    if not question or len(options) < 2:
        return None
    return {"question": question, "options": options}


async def _ask_for_json(provider_key: str, model_id: str, prompt: str, parser, label: str):
    """Запрашивает у LLM структурированный JSON; при невалидном ответе повторяет ещё раз."""
    for attempt in range(2):
        p = prompt
        if attempt > 0:
            p += ("\n\nВАЖНО: предыдущий ответ не распознан. Верни СТРОГО один валидный JSON-объект — "
                  "без markdown, без блоков кода, без пояснений до и после.")
        parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": p}], temperature=0.9):
            parts.append(token)
        parsed = parser("".join(parts))
        if parsed:
            return parsed
        logger.warning(f"{label}: невалидный JSON от модели (попытка {attempt + 1}): {''.join(parts)[:300]}")
    return None


async def _run_quiz(chat_id: int, context: ContextTypes.DEFAULT_TYPE, topic: str, level: dict = None, learn_session: bool = False):
    """Внутренняя функция для запуска викторины.
    Если передан level (из roadmap), вопрос привязывается к языкам/стеку уровня
    и сохраняется для экспорта в markdown.
    Если learn_session=True — полл-викторина регистрируется для проверки знаний
    в режиме обучения (ответ пользователя проверяется и начисляется счёт)."""
    provider_key, model_id = get_current_model(chat_id)

    topics = {
        "general": "общие знания",
        "programming": "программирование",
        "geography": "география",
        "science": "наука",
        "random": "случайная тема",
    }
    topic_name = topics.get(topic, topic)
    level_title = None

    if level:
        level_title = level.get("title", "Уровень")
        langs = ", ".join(level.get("languages", []))
        topic_name = (f"программирование ({topic_name}), с уклоном в конкретный стек: "
                      f"{level.get('stack', '-')}, языки: {langs}. "
                      f"Вопрос должен проверять знание именно этого стека/языков уровня «{level_title}»")

    prompt = (
        f"Ты — составитель викторин. Придумай ОДИН новый вопрос викторины на тему: {topic_name}.\n"
        f"Номер раунда: {random.randint(1, 999999)} — используй его как зерно случайности: "
        f"каждый раунд обязан отличаться, выбирай новую узкую подтему и нестандартный факт, "
        f"не повторяй заезженные шаблонные вопросы.\n"
        f"Строгие правила:\n"
        f"1. Вопрос однозначный и фактически корректный, на русском языке.\n"
        f"2. Ровно 4 варианта ответа, из них РОВНО ОДИН правильный.\n"
        f"3. Правильный ответ обязан быть объективно верным — перепроверь факт перед отправкой "
        f"(например, язык программирования для веба — JavaScript, а не Python).\n"
        f"4. Расположи правильный ответ на случайной позиции, а не всегда первым.\n"
        f"Верни ТОЛЬКО один валидный JSON-объект, без markdown и пояснений:\n"
        f'{{"question": "текст вопроса", "options": ["вариант 1", "вариант 2", "вариант 3", "вариант 4"], '
        f'"correct": N, "explanation": "почему этот ответ правильный, 1-2 предложения"}}\n'
        f"correct — индекс правильного варианта от 0 до 3."
    )

    from telegram import Bot
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    msg = await bot.send_message(chat_id, f"{emoji('brain')} <b>Генерирую викторину...</b>", parse_mode=ParseMode.HTML)

    try:
        quiz = await _ask_for_json(provider_key, model_id, prompt, _parse_quiz, "quiz")
        if quiz:
            if level and level_title:
                add_generated_quiz(chat_id, level_title, quiz)
            await bot.delete_message(chat_id, msg.message_id)
            poll_kwargs = {
                "chat_id": chat_id,
                "question": quiz["question"],
                "options": quiz["options"],
                "type": "quiz",
                "correct_option_id": quiz["correct"],
            }
            if quiz["explanation"]:
                poll_kwargs["explanation"] = quiz["explanation"]
            sent = await bot.send_poll(**poll_kwargs)
            if learn_session:
                sess = LEARN_SESSIONS.get(chat_id)
                sess.setdefault("score", 0)
                sess.setdefault("total", 0)
                LEARN_QUIZ[sent.poll.id] = {
                    "correct": quiz["correct"],
                    "chat_id": chat_id,
                    "level_index": sess.get("level_index", 0) if sess else 0,
                }
        else:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg.message_id,
                text=f"{emoji('warning')} Модели не удалось составить корректный вопрос. Попробуйте ещё раз."
            )
    except Exception as e:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"{emoji('error')} Ошибка: {e}"
        )


async def _run_poll(chat_id: int, context: ContextTypes.DEFAULT_TYPE, poll_type: str):
    """Внутренняя функция для создания опроса."""
    provider_key, model_id = get_current_model(chat_id)

    types = {
        "opinion": "опрос для сбора мнений по актуальной теме; варианты ответа: Полностью согласен / Скорее согласен / Нейтрально / Скорее нет / Полностью не согласен",
        "compare": "опрос, в котором участники выбирают лучший из 3-5 вариантов (технологии, инструменты, подходы)",
        "priority": "опрос о приоритетах — что важнее; варианты ответа: Высокий приоритет / Средний приоритет / Низкий приоритет",
    }

    prompt = (
        f"Ты — составитель опросов. Придумай ОДИН новый оригинальный опрос.\n"
        f"Номер раунда: {random.randint(1, 999999)} — используй его как зерно случайности: "
        f"каждый опрос обязан отличаться, выбирай новую тему, не повторяйся.\n"
        f"Тип: {types.get(poll_type, types['opinion'])}.\n"
        f"Правила: вопрос короткий и понятный, на русском; 3-6 вариантов ответа без дублей.\n"
        f"Верни ТОЛЬКО один валидный JSON-объект, без markdown и пояснений:\n"
        f'{{"question": "текст вопроса", "options": ["вариант 1", "вариант 2", "вариант 3"]}}'
    )

    from telegram import Bot
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    msg = await bot.send_message(chat_id, f"{emoji('code')} <b>Создаю опрос...</b>", parse_mode=ParseMode.HTML)

    try:
        poll = await _ask_for_json(provider_key, model_id, prompt, _parse_poll, "poll")
        if poll:
            await bot.delete_message(chat_id, msg.message_id)
            await bot.send_poll(
                chat_id=chat_id,
                question=poll["question"],
                options=poll["options"],
                type="regular",
                allows_multiple_answers=False
            )
        else:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg.message_id,
                text=f"{emoji('warning')} Модели не удалось составить опрос. Попробуйте ещё раз."
            )
    except Exception as e:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"{emoji('error')} Ошибка: {e}"
        )


async def _run_code_help(chat_id: int, context: ContextTypes.DEFAULT_TYPE, action: str, code: str = None):
    """Внутренняя функция для помощи с кодом."""
    provider_key, model_id = get_current_model(chat_id)
    
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
    
    from telegram import Bot
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    msg = await bot.send_message(chat_id, f"{emoji('code')} <b>Анализирую код...</b>", parse_mode=ParseMode.HTML)
    
    try:
        parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": prompt}]):
            parts.append(token)
        result = "".join(parts).strip()
        
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"{emoji('code')} <b>Результат:</b>\n\n{md_to_html(result)}",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"{emoji('error')} Ошибка: {e}"
        )


async def _run_task(chat_id: int, context: ContextTypes.DEFAULT_TYPE, difficulty: str, level: dict = None):
    """Внутренняя функция для генерации задачи. Если передан level (из roadmap),
    задача привязывается к конкретному стеку/языкам уровня и сохраняется для экспорта."""
    provider_key, model_id = get_current_model(chat_id)
    
    difficulties = {
        "easy": "Junior уровень: базовые алгоритмы, массивы, строки, циклы",
        "medium": "Middle уровень: структуры данных, DP, графы, алгоритмы сортировки",
        "hard": "Senior уровень: сложные алгоритмы, системное проектирование, конкурентность",
        "random": "случайная сложность",
    }

    lang_note = ""
    stack_note = ""
    level_title = None
    if level:
        level_title = level.get("title", "Уровень")
        langs = level.get("languages", [])
        stack_note = f"\nТребуемый стек: {level.get('stack', '-')}."
        if langs:
            lang_note = (f"\nЗадача обязательно должна решаться на одном из этих языков: {', '.join(langs)}. "
                         f"Выбери конкретный язык из списка и укажи его. Решение пиши именно на этом языке.")
    
    prompt = (
        f"Создай задачу по программированию уровня: {difficulties.get(difficulty, difficulties['medium'])}."
        f"{stack_note}"
        f"{lang_note}\n"
        f"Формат:\n"
        f"1. Название задачи\n"
        f"2. Условие (входные/выходные данные, ограничения)\n"
        f"3. Пример ввода/вывода\n"
        f"4. Подсказка (алгоритм)\n"
        f"5. Решение на выбранном языке с комментариями\n\n"
        f"На русском языке."
    )
    
    from telegram import Bot
    bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    msg = await bot.send_message(chat_id, f"{emoji('brain')} <b>Генерирую задачу...</b>", parse_mode=ParseMode.HTML)
    
    try:
        parts = []
        async for token, _ in call_provider_api(provider_key, model_id, [{"role": "user", "content": prompt}]):
            parts.append(token)
        result = "".join(parts).strip()

        if level and level_title:
            add_generated_task(chat_id, level_title, {"level_label": f"Задача · {level_title}", "problem": result})

        title = f"Задача ({difficulty}) · {level_title}" if level_title else f"Задача ({difficulty})"
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"{emoji('brain')} <b>{title}:</b>\n\n{md_to_html(result)}",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"{emoji('error')} Ошибка: {e}"
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
    chat_id = query.message.chat_id
    data = query.data
    
    # Быстрые переходы меню — отвечаем сразу
    if data in ("main_menu", "models_menu", "quiz_menu", "poll_menu", "code_help", "code_tasks", "roadmap_menu", "stack_menu"):
        await query.answer()
    
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
            await query.answer()
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
            await query.answer()
            await query.edit_message_text(
                f"{emoji('check')} Модель изменена на <b>{model_key}</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=build_models_menu(chat_id)
            )
        else:
            await query.answer(f"Модель {model_key} недоступна для этого провайдера", show_alert=True)
    elif data == "quiz_menu":
        await query.answer()
        await query.edit_message_text(
            f"{emoji('brain')} <b>Викторина</b>\n\nВыберите тему:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_quiz_menu(chat_id)
        )
    elif data == "poll_menu":
        await query.answer()
        await query.edit_message_text(
            f"{emoji('code')} <b>Умные опросы</b>\n\nВыберите тип:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_poll_menu(chat_id)
        )
    elif data == "code_help":
        await query.answer()
        await query.edit_message_text(
            f"{emoji('code')} <b>Помощь с кодом</b>\n\nВыберите действие:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_code_help_menu(chat_id)
        )
    elif data == "code_tasks":
        await query.answer()
        await query.edit_message_text(
            f"{emoji('brain')} <b>Задачи по программированию</b>\n\nВыберите уровень:",
            parse_mode=ParseMode.HTML,
            reply_markup=build_code_tasks_menu(chat_id)
        )
    elif data == "roadmap_menu":
        await query.answer()
        await query.edit_message_text(
            f"{emoji('rocket')} <b>Roadmap по стеку и языкам</b>\n\n"
            f"{ROADMAP.get('description', '')}",
            parse_mode=ParseMode.HTML,
            reply_markup=build_roadmap_menu(chat_id)
        )
    elif data == "stack_menu":
        await query.answer()
        text, markup = build_stack_menu(chat_id)
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=markup)
    elif data.startswith("stack:"):
        await stack_callback(update, context)
        return
    elif data.startswith("intv:"):
        await _interview_callback(update)
        return
    elif data == "road:menu":
        await query.answer()
        await query.edit_message_text(
            f"{emoji('rocket')} <b>Roadmap по стеку и языкам</b>\n\n"
            f"{ROADMAP.get('description', '')}",
            parse_mode=ParseMode.HTML,
            reply_markup=build_roadmap_menu(chat_id)
        )
    elif data == "road:export_all":
        await query.answer(f"{emoji('check')} Отправляю roadmap...")
        await roadmap_export_all(chat_id)
    elif data == "road:finish":
        await query.answer("🏁 Все уровни пройдены!")
        await query.edit_message_text(
            f"{emoji('rocket')} 🏁 <b>Вы прошли весь roadmap!</b>\n\n"
            f"Можно повторить уровни или экспортировать результат в Obsidian.",
            parse_mode=ParseMode.HTML,
            reply_markup=build_roadmap_menu(chat_id)
        )
    elif data.startswith("road:"):
        _, action, raw_index = data.split(":", 2)
        index = normalize_level_index(chat_id, int(raw_index))
        level = get_level(chat_id, index)
        if action == "level":
            await query.answer()
            await query.edit_message_text(
                build_roadmap_level_text(chat_id, index),
                parse_mode=ParseMode.HTML,
                reply_markup=build_roadmap_level_menu(chat_id, index)
            )
        elif action == "learn" and level:
            await query.answer("🎓 Начинаю обучение...")
            await _run_learn(chat_id, context, index)
        elif action == "export":
            await query.answer(f"{emoji('check')} Экспортирую уровень {index + 1}...")
            await roadmap_export_level(chat_id, index)
        elif action == "task" and level:
            await query.answer(f"{emoji('brain')} Генерирую задачу по стеку...")
            await _run_task(chat_id, context, level.get("difficulty", "medium"), level)
        elif action == "quiz" and level:
            await query.answer("🧠 Генерирую викторину по языку...")
            await _run_quiz(chat_id, context, "programming", level)
        elif action == "done":
            msg, ndone = mark_topic_done(chat_id, index)
            await _save_user_config()
            await query.answer(f"{emoji('check')} {msg}")
            await query.edit_message_text(
                build_roadmap_level_text(chat_id, index),
                parse_mode=ParseMode.HTML,
                reply_markup=build_roadmap_level_menu(chat_id, index)
            )
        elif action == "next":
            nxt = normalize_level_index(chat_id, index + 1)
            await set_level_index(chat_id, nxt)
            await query.answer(f"Переходим к уровню {nxt + 1}")
            await query.edit_message_text(
                build_roadmap_level_text(chat_id, nxt),
                parse_mode=ParseMode.HTML,
                reply_markup=build_roadmap_level_menu(chat_id, nxt)
            )
    elif data.startswith("quiz:"):
        topic = data.split(":")[1]
        await _run_quiz(chat_id, context, topic)
    elif data.startswith("poll:"):
        poll_type = data.split(":")[1]
        await _run_poll(chat_id, context, poll_type)
    elif data.startswith("code:"):
        action = data.split(":")[1]
        await query.answer()
        # Для помощи с кодом нужно, чтобы пользователь прислал код после
        await query.edit_message_text(
            f"{emoji('warning')} Пришли код после нажатия кнопки или сделай реплай на сообщение с кодом.\n"
            f"Пример: <code>/code review твой код</code>",
            parse_mode=ParseMode.HTML
        )
    elif data.startswith("task:"):
        difficulty = data.split(":")[1]
        await _run_task(chat_id, context, difficulty)


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
    messages.append({"role": "user", "content": f"ТЫ ОБЯЗАН ОТВЕЧАТЬ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ HTML-ТЕГИ (<hr>, <strong>, <b>, <ol>, <ul>, <li>, <h1>, <h2>, <h3>, <p>, <div>, <span> И ДРУГИЕ). ИСПОЛЬЗУЙ ТОЛЬКО MARKDOWN: **жирный**, *курсив*, `код`, ```блоки кода```, - списки, 1. нумерованные списки, > цитаты. ОТВЕЧАЙ СРАЗУ И ЧЁТКО, БЕЗ РАССУЖДЕНИЙ. СТРУКТУРИРУЙ ОТВЕТ: короткое вступление, затем разделы/списки/код. ЕСЛИ ЗАДАЛИ ВОПРОС (о викторине, коде или чём угодно) — ОТВЕЧАЙ ПРЯМО НА НЕГО. ВОПРОС: {user_text}"})

    all_parts = []
    last_edit_len = 0
    start_time = asyncio.get_event_loop().time()
    usage = {}

    # Фоновая задача для обновления таймера каждые 2 секунды
    stop_timer = asyncio.Event()
    last_timer_text = ""
    edit_lock = asyncio.Lock()  # Защита от race condition при edit_text

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
                async with edit_lock:
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
                async with edit_lock:
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
            BotCommand("roadmap", "🗺️ Roadmap по уровням (стек/языки)"),
            BotCommand("stack", "🧰 Выбрать стек и языки"),
            BotCommand("interview", "🎙️ Техсобеседование по Go"),
            BotCommand("learn", "🎓 Режим обучения: теория + тест"),
            BotCommand("export", "📤 Экспорт tasks/вопросов в .md для Obsidian"),
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
    app.add_handler(CommandHandler("roadmap", roadmap_command))
    app.add_handler(CommandHandler("stack", stack_command))
    app.add_handler(CommandHandler("interview", interview_command))
    app.add_handler(CommandHandler("learn", learn_command))
    app.add_handler(CommandHandler("export", export_command))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(PollAnswerHandler(_poll_answer))
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