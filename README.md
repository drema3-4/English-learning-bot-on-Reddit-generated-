# Reddit English Learning Bot

Telegram-бот для изучения английского по постам Reddit. Бот принимает ссылку на пост, забирает текст и комментарии, отправляет их в OpenAI-совместимую модель, сохраняет найденные слова, фразы и правила в SQLite, а затем помогает повторять материал в Telegram.

## 1. Что делает бот

- Принимает ссылки на посты Reddit.
- Извлекает заголовок, текст поста и верхние комментарии.
- Создает карточки со словами, фразами и грамматическими правилами.
- Хранит прогресс повторения для каждого пользователя.
- Ограничивает доступ первыми `MAX_USERS` пользователями.

## 2. Требования

- Docker и Docker Compose на сервере.
- Telegram Bot Token.
- Reddit API credentials.
- OpenAI API key или ключ совместимого API.

## 3. Настройка .env

Скопируйте пример и заполните секреты:

```bash
cp .env.example .env
nano .env
```

Основные переменные:

```env
TELEGRAM_BOT_TOKEN=
OPENAI_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
DATABASE_URL=sqlite+aiosqlite:////data/bot.db
```

Для Docker оставьте `DATABASE_URL=sqlite+aiosqlite:////data/bot.db`, чтобы база лежала в volume.

## 4. Получение Telegram Bot Token

1. Откройте Telegram и найдите `@BotFather`.
2. Выполните команду `/newbot`.
3. Задайте имя и username бота.
4. Скопируйте token в `TELEGRAM_BOT_TOKEN`.

## 5. Получение Reddit API credentials

1. Откройте страницу Reddit apps: `https://www.reddit.com/prefs/apps`.
2. Создайте приложение типа `script`.
3. Скопируйте `client_id` в `REDDIT_CLIENT_ID`.
4. Скопируйте `secret` в `REDDIT_CLIENT_SECRET`.
5. Оставьте понятный `REDDIT_USER_AGENT`, например `reddit-english-learning-bot/0.1`.

## 6. Запуск через Docker Compose

Команды для сервера:

```bash
git clone <repo>
cd reddit-english-bot
cp .env.example .env
nano .env
docker compose up --build -d
docker compose logs -f
```

Для запуска в текущей консоли:

```bash
docker compose up --build
```

Контейнер перед стартом бота выполняет миграции:

```bash
poetry run alembic upgrade head
poetry run bot
```

## 7. Команды бота

```text
/start
/help
/review_words
/review_phrases
/review_rules
/status
```

## 8. Где лежит база

SQLite хранится на сервере в:

```text
./data/bot.db
```

Внутри контейнера этот путь смонтирован как:

```text
/data/bot.db
```

Бэкап SQLite:

```bash
cp data/bot.db data/bot.backup.db
```

## 9. Как посмотреть логи

```bash
docker compose logs -f
```

Логи только сервиса бота:

```bash
docker compose logs -f bot
```

## 10. Как остановить бота

```bash
docker compose down
```
