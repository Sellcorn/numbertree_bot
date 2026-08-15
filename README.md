# Qwen Telegram Bot

Telegram-бот на базе Qwen 3.6 Flash (QwenCloud Token Plan) со стримингом ответов и отображением процесса рассуждений.

## Возможности
- Показывает процесс «думания» поэтапно (таймер + предпросмотр ответа)
- Streaming-режим (по токенам)
- Помнит контекст диалога (полная память в личке, короткая в группах)
- Выбор провайдера и модели через меню (/menu)
- Викторины (/quiz), умные опросы (/poll), помощь с кодом (/code), задачи (/task)
- Групповые команды: /context, /summary, /judge
- Работает в личке, по @упоминанию и по reply в группах

## Установка

```bash
cd nvidia_telegram_bot

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# Отредактируйте .env и вставьте ваши токены
```

## Получение токенов

### Telegram Bot Token
1. Напишите [@BotFather](https://t.me/BotFather)
2. Отправьте `/newbot`
3. Следуйте инструкциям
4. Скопируйте токен в `.env` как `TELEGRAM_BOT_TOKEN`

### Qwen API Key
1. Зарегистрируйтесь на [home.qwencloud.com](https://home.qwencloud.com/)
2. Оформите подписку Token Plan и создайте API Key (начинается с `sk-sp-`)
3. Скопируйте в `.env` как `QWEN_API_KEY`

Endpoint Token Plan (OpenAI-совместимый) уже настроен в коде:
`https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`

## Запуск

```bash
python main.py
```

## Деплой (Docker + Railway)

```bash
docker build -t telegram-bot .
```

Расписание запуска/остановки настроено в `.github/workflows/bot-schedule.yml` (09:00–01:00 MSK).

## Как это работает

1. Пользователь пишет вопрос
2. Бот отправляет сообщение «Думаю...»
3. В процессе генерации бот обновляет сообщение каждые ~300 символов и обновляет таймер каждые 2 секунды
4. В финале показывает ответ с HTML-разметкой и счётчиком токенов

## Структура проекта
```
nvidia_telegram_bot/
├── main.py          # Основной код бота
├── requirements.txt # Зависимости
├── .env.example     # Пример конфигурации
└── .env             # Ваши секреты (не коммитить в git!)
```
