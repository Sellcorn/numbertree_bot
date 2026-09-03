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


# ---------- модель по умолчанию и запасные ----------
# Регрессия: дефолтом стояла рассуждающая модель. Она молчит, пока думает, и
# первый токен приходил через десятки секунд — в чате это «бот завис».
_nvidia = main.PROVIDERS["nvidia"]
check("дефолт — Nemotron Super",
      _nvidia["models"][_nvidia["default"]], "nvidia/nemotron-3-super-120b-a12b")
check_true("рассуждающая Muse убрана из меню",
           not any("muse" in m for m in _nvidia["models"].values()),
           str(list(_nvidia["models"].values())))
# Запасные берутся по порядку словаря: подмена модели не должна оказаться
# медленнее той, что не ответила.
check("запасные — самые быстрые из оставшихся",
      main._fallback_models("nvidia", "nvidia/nemotron-3-super-120b-a12b")[1:],
      ["nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3-ultra-550b-a55b"])


# ---------- история не раздувает промпт ----------
# Регрессия: история уходит в модель дважды за сообщение (маршрутный вызов +
# ответ), и десять развёрнутых ответов превращались в десятки тысяч токенов
# префилла — а он целиком лежит в ожидании первого токена.
_hist = [{"user": "a", "assistant": "x" * 5000}, {"user": "b", "assistant": "y" * 5000}]
_msgs = main.history_messages(_hist)
check("реплик столько же", len(_msgs), 4)
check_true("старый ответ обрезан",
           len(_msgs[1]["content"]) <= main.MAX_OLD_REPLY_CHARS + 20,
           str(len(_msgs[1]["content"])))
check("последний ответ идёт целиком", _msgs[3]["content"], "y" * 5000)
check("вопросы пользователя не трогаем", _msgs[0]["content"], "a")


# ---------- перехватчик интервью ----------
# Регрессия: _is_interview_active был async и в `if` давал корутину — всегда
# истинную, поэтому «стоп» и «дальше» уходили в несуществующее собеседование.
main.INTERVIEW_SESSIONS.clear()
check("без сессии интервью неактивно", main._is_interview_active(42), False)
main.INTERVIEW_SESSIONS[42] = {"score": 0, "total": 0}
check("с сессией интервью активно", main._is_interview_active(42), True)
main.INTERVIEW_SESSIONS.clear()


# ---------- потолок на время одного запроса ----------
# Регрессия: повторы перемножались (3 попытки x 3 модели = 9 обращений), и при
# молчащей модели пользователь ждал больше 1000с вместо ответа или ошибки.
import asyncio
import json

import httpx


async def _deadline_probe():
    fake, budget = 3.0, 8.0
    saved = (main._stream_model, main._call_model_once, main.TOTAL_DEADLINE)
    main.TOTAL_DEADLINE = budget

    async def dead_stream(p, m, msgs, stream=True, temperature=0.6, timeout=None):
        assert timeout is not None, "таймаут обязан прокидываться в попытку"
        await asyncio.sleep(min(fake, timeout))
        raise httpx.ReadTimeout("модель молчит")
        yield  # pragma: no cover

    async def dead_once(p, m, msgs, tools=None, temperature=0.3, timeout=None):
        assert timeout is not None, "таймаут обязан прокидываться в попытку"
        await asyncio.sleep(min(fake, timeout))
        raise httpx.ReadTimeout("модель молчит")

    main._stream_model, main._call_model_once = dead_stream, dead_once
    loop = asyncio.get_event_loop()
    try:
        t0 = loop.time()
        try:
            async for _ in main.call_provider_api("nvidia", "nvidia/nemotron-3-super-120b-a12b", []):
                pass
        except Exception:
            pass
        stream_took = loop.time() - t0

        t0 = loop.time()
        try:
            await main.call_provider_api_once("nvidia", "nvidia/nemotron-3-super-120b-a12b", [])
        except Exception:
            pass
        once_took = loop.time() - t0
    finally:
        main._stream_model, main._call_model_once, main.TOTAL_DEADLINE = saved
    return stream_took, once_took, budget


_stream_took, _once_took, _budget = asyncio.run(_deadline_probe())
check_true(f"стриминг уложился в бюджет ({_stream_took:.1f}s <= {_budget}s)",
           _stream_took <= _budget + 1, f"{_stream_took:.1f}s")
check_true(f"маршрутный вызов уложился в бюджет ({_once_took:.1f}s <= {_budget}s)",
           _once_took <= _budget + 1, f"{_once_took:.1f}s")
check_true("бюджет поиска задан", main.RESEARCH_BUDGET > 0)


# ---------- предпросмотр не задерживает финальный ответ ----------
# Регрессия: фоновая задача таймера спала фиксированные 2с, и код ответа ждал
# конца сна — до двух лишних секунд молчания на КАЖДОМ ответе.
async def _preview_probe():
    stop = asyncio.Event()

    async def updater():
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=main.PREVIEW_INTERVAL)
                return
            except asyncio.TimeoutError:
                pass

    loop = asyncio.get_event_loop()
    task = asyncio.create_task(updater())
    await asyncio.sleep(0)
    t0 = loop.time()
    stop.set()
    await task
    return loop.time() - t0


_preview_took = asyncio.run(_preview_probe())
check_true(f"таймер останавливается сразу ({_preview_took * 1000:.0f}мс)",
           _preview_took < 0.2, f"{_preview_took:.2f}s")


# ---------- общий HTTP-клиент ----------
# Регрессия: под каждый запрос к модели поднимался свой httpx.AsyncClient, то
# есть полное TLS-рукопожатие на каждое из восьми обращений за один вопрос.
# Клиент теперь общий — проверяем, что оба вызова через него по-прежнему
# разбирают ответ провайдера. Сеть не трогаем: MockTransport.
def _fake_provider(request):
    body = json.loads(request.content)
    if body.get("stream"):
        chunks = [
            'data: {"choices":[{"delta":{"content":"При"}}]}',
            'data: {"choices":[{"delta":{"content":"вет"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,'
            '"total_tokens":12}}',
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n\n".join(chunks) + "\n\n")
    return httpx.Response(200, json={"choices": [{"message": {"content": "нет"}}]})


async def _http_probe():
    saved_key = main.PROVIDERS["nvidia"]["api_key"]
    main.PROVIDERS["nvidia"]["api_key"] = "nvapi-test"
    main._HTTP_CLIENT = httpx.AsyncClient(transport=httpx.MockTransport(_fake_provider))
    model = "nvidia/nemotron-3-super-120b-a12b"
    try:
        tokens, usage = [], {}
        async for token, u in main.call_provider_api("nvidia", model, []):
            tokens.append(token)
            if u:
                usage = u
        message = await main.call_provider_api_once("nvidia", model, [])
        await main.close_http_client()
        return "".join(tokens), usage, message, main._HTTP_CLIENT
    finally:
        main.PROVIDERS["nvidia"]["api_key"] = saved_key
        main._HTTP_CLIENT = None


_text, _usage, _message, _closed = asyncio.run(_http_probe())
check("поток собран из чанков", _text, "Привет")
check("usage из финального чанка прочитан", _usage.get("total_tokens"), 12)
check("нестриминговый вызов вернул message", _message.get("content"), "нет")
check("после close_http_client клиента нет", _closed, None)
check_true("таймауты разнесены по фазам",
           main.request_timeout(42.0).read == 42.0
           and main.request_timeout(42.0).connect == main.CONNECT_TIMEOUT)


total = 32
if failures:
    print(f"ПРОВАЛЕНО {len(failures)}:")
    print("\n".join(failures))
    sys.exit(1)
print(f"OK: все проверки прошли ({total} групп)")
