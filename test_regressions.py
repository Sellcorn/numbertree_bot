"""Офлайн-регрессии: разметка, счёт собеседования, настройки, регистрация команд.

Живой API не трогает и токенов не тратит — всё проверяется на чистых функциях.
Запуск: python test_regressions.py
"""
import inspect
import os
import sys

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")

import main


failures = []


def check(name, got, expected):
    if got != expected:
        failures.append(f"  {name}\n    ожидалось: {expected!r}\n    получено:  {got!r}")


def check_true(name, condition, note=""):
    if not condition:
        failures.append(f"  {name}{': ' + note if note else ''}")


# ---------- md_to_html ----------
# Регрессия: чистка HTML-тегов шла по сырому тексту и съедала угловые скобки
# внутри кода, а финальная сборка переносов срезала отступы.

html = main.md_to_html("```cpp\nstd::vector<int> v;\n```\n")
check_true("дженерики в блоке кода уцелели", "vector&lt;int&gt;" in html, html)

html = main.md_to_html("```python\ndef f():\n    if True:\n        return 1\n```\n")
check_true("отступы в блоке кода уцелели", "\n    if True:\n        return 1" in html, html)

html = main.md_to_html("Тип `vector<int>` тут.\n")
check_true("угловые скобки в `коде` уцелели", "<code>vector&lt;int&gt;</code>" in html, html)

html = main.md_to_html("```bash\n# коммент\necho hi\n```\n")
check_true("решётка в блоке кода не стала заголовком", "<b>" not in html, html)

# А вот HTML, присланный моделью вместо markdown, по-прежнему вычищается.
check("HTML из прозы вычищен",
      main.md_to_html("<p>Модель прислала <strong>HTML</strong></p>"),
      "Модель прислала HTML")

# Таблицы по-прежнему уезжают в моноширинный <pre>.
table = main.md_to_html("| Версия | Дата |\n|---|---|\n| 1.26 | 2026-02-10 |\n")
check_true("таблица отрисована как <pre>", table.startswith("<pre>") and "│" in table, table)

# Код в ячейке по-прежнему разворачивается в список (tables.split_cell).
cell = main.md_to_html("| Метод | Пример |\n|---|---|\n| Slice | ```go\\ns := 1\\n``` |\n")
check_true("код из ячейки вынесен списком", "▸ Slice" in cell, cell)

check("пустой вход", main.md_to_html(""), "")


# ---------- оценка письменного ответа ----------
# Регрессия: _extract_grade возвращал None, а вызывающий код сравнивал его с 7.
check("балл разобран", main._extract_grade("Оценка: 8/10\nВерно: всё"), 8)
check("балл обрезан сверху", main._extract_grade("Оценка: 42/10"), 10)
check("балла нет — None", main._extract_grade("Ответ неплохой, но без цифр"), None)
check_true("None не роняет сравнение", (main._extract_grade("нет балла") or 0) < 7)


# ---------- эмодзи ----------
# Регрессия: eye/like были в CUSTOM_EMOJI, но не в fallback — KeyError.
for key in set(main.CUSTOM_EMOJI) | set(main.FALLBACK_EMOJI):
    try:
        check_true(f"emoji({key!r}) не пустой", bool(main.emoji(key)))
    except Exception as e:
        failures.append(f"  emoji({key!r}) упал: {type(e).__name__}: {e}")


# ---------- настройки: чтение не создаёт записей ----------
main.USER_CONFIG.clear()
main._peek_user_settings(4242)
main.get_progress(4242)
main.get_user_settings(4242)
check("чтение не завело запись в progress.json", main.USER_CONFIG, {})
main._get_user_settings(4242)
check_true("запись на изменение создана", "4242" in main.USER_CONFIG)
main.USER_CONFIG.clear()


# ---------- дубликаты тем в focus ----------
# Регрессия: tops.index(t) находил первое совпадение, и одинаковые темы
# отмечались пройденными разом.
saved_roadmap = main.ROADMAP
main.ROADMAP = {
    "languages": {"x": {"levels": [{"title": "L1", "focus": "каналы, каналы, слайсы"}]}},
    "templates": [],
}
main.USER_CONFIG.clear()
main.USER_CONFIG["7"] = {"template": None, "languages": ["x"], "level_index": 0,
                         "done_topics": ["lvl0#0"]}
check("отмечена только первая из одинаковых тем",
      main.level_topics_done(7, 0), ["каналы"])
main.ROADMAP = saved_roadmap
main.USER_CONFIG.clear()


# ---------- регистрация команд ----------
# Регрессия: лямбда передавала третий аргумент в хендлер на два параметра,
# а task_handler вообще не был определён.
for name in ("quiz_handler", "poll_handler", "code_help_handler", "task_handler"):
    fn = getattr(main, name, None)
    if fn is None:
        failures.append(f"  {name} не определён")
        continue
    params = list(inspect.signature(fn).parameters)
    check(f"{name} принимает (update, context)", params, ["update", "context"])


# ---------- маппинг поллов интервью ----------
# Регрессия: завершение сессии делало INTERVIEW_QUIZ.clear() и гасило чужие поллы.
main.INTERVIEW_QUIZ.clear()
main.INTERVIEW_QUIZ["p1"] = {"chat_id": 1, "correct": 0}
main.INTERVIEW_QUIZ["p2"] = {"chat_id": 2, "correct": 1}
main.INTERVIEW_SESSIONS[1] = {"score": 0, "total": 0, "written": []}
main._interview_finish(1)
check("чужой полл пережил завершение чужой сессии",
      sorted(main.INTERVIEW_QUIZ), ["p2"])
main.INTERVIEW_QUIZ.clear()
main.INTERVIEW_LAST_FINISH.clear()


# ---------- ограничение числа отслеживаемых чатов ----------
store = {i: [] for i in range(main.MAX_TRACKED_CHATS + 10)}
main._trim_store(store)
check("словарь обрезан до потолка", len(store), main.MAX_TRACKED_CHATS)
check_true("выброшены самые старые ключи", 0 not in store and 509 in store)


# ---------- поиск не запускается на болтовне ----------
# Регрессия: шаг веб-поиска стоит ОТДЕЛЬНОГО обращения к модели («искать или
# нет»), и на «привет» оно тратилось впустую. У рассуждающей модели это второй
# полный проход размышлений, то есть приветствие оплачивалось дважды.
for text in ("привет", "Привет!", "  спасибо  ", "как дела?", "ок", "hello"):
    check_true(f"{text!r} — болтовня", main.is_small_talk(text))
for text in ("Какая сейчас последняя версия Go?", "привет, найди курс биткоина",
             "что такое горутина", "расскажи про привет", ""):
    check_true(f"{text!r} — НЕ болтовня", not main.is_small_talk(text))


# ---------- сила рассуждений ----------
# Muse Glimmer читает её из системного промпта; маршрутному вызову high не нужен.
msgs = [{"role": "user", "content": "x"}]
routed = main.with_reasoning(msgs, "meta/muse-glimmer-30b", "low")
check("маршруту — низкая сила", routed[0], {"role": "system", "content": "Reasoning strength: low"})
check("исходный список не тронут", len(msgs), 1)
check("дважды не накапливается", len(main.with_reasoning(msgs, "meta/muse-glimmer-30b", "low")), 2)
check("чужая модель без system",
      main.with_reasoning(msgs, "nvidia/nemotron-3-super-120b-a12b", "low"), msgs)


# ---------- перехватчик интервью ----------
# Регрессия: _is_interview_active был async и в `if` давал корутину — всегда
# истинную, поэтому «стоп» и «дальше» уходили в несуществующее собеседование.
main.INTERVIEW_SESSIONS.clear()
check("без сессии интервью неактивно", main._is_interview_active(42), False)
main.INTERVIEW_SESSIONS[42] = {"score": 0, "total": 0}
check("с сессией интервью активно", main._is_interview_active(42), True)
main.INTERVIEW_SESSIONS.clear()


total = 26
if failures:
    print(f"ПРОВАЛЕНО {len(failures)}:")
    print("\n".join(failures))
    sys.exit(1)
print(f"OK: все проверки прошли ({total} групп)")
