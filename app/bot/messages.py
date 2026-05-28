from __future__ import annotations

from app.services.review import PhraseCard, ReviewCardDTO, RuleCard, WordCard


START_MESSAGE = """Привет! Я бот для изучения английского по Reddit.

Пришли мне ссылку на Reddit-пост, и я извлеку из него:
— важные слова;
— полезные фразы;
— правила и конструкции.

Команды:
/review_words — повторить слова
/review_phrases — повторить фразы
/review_rules — повторить правила
/status — статус обработки"""

HELP_MESSAGE = START_MESSAGE

USER_LIMIT_REACHED = "Доступ ограничен первыми 5 пользователями."
SEND_REDDIT_LINK = "Пришли мне ссылку на Reddit-пост."
SEND_POST_TEXT = "Пришли текст поста на английском."
PROCESSING_ALREADY_ACTIVE = "У тебя уже есть задача в обработке. Дождись её завершения."
QUEUED = "Принял ссылку. Начинаю обработку Reddit-поста и комментариев."
QUEUED_TEXT = "Принял текст. Начинаю извлекать слова, фразы и правила."
INVALID_REDDIT_URL = "Это не похоже на корректную ссылку на Reddit-пост."
NO_ITEMS = "Пока нет сохранённых карточек этого типа. Сначала пришли материал для обработки."
RATING_SAVED = "Оценка сохранена."
REVIEW_COMPLETED = "Повторение завершено."
REVIEW_SESSION_EXPIRED = "Сессия повторения устарела. Запусти повторение заново."
REVIEW_NOT_FOUND = "Не нашёл активную сессию повторения."
INVALID_RATING = "Оценка должна быть от 1 до 5."
NO_JOBS = "Задач пока нет."


MANUAL_START_MESSAGE = """Привет! Я бот для изучения английского.

Пришли мне текст поста на английском, и я извлеку из него:
— важные слова;
— полезные фразы;
— правила и конструкции.

Команды:
/review_words — повторить слова
/review_phrases — повторить фразы
/review_rules — повторить правила
/status — статус обработки"""


def start_message(has_reddit_credentials: bool) -> str:
    if has_reddit_credentials:
        return START_MESSAGE
    return MANUAL_START_MESSAGE


def help_message(has_reddit_credentials: bool) -> str:
    return start_message(has_reddit_credentials)


def format_job_status(status: str, error_message: str | None = None) -> str:
    if status == "queued":
        return "Последняя задача: queued."
    if status == "processing":
        return "Последняя задача: processing."
    if status == "done":
        return "Последняя задача завершена успешно."
    if status == "failed":
        return f"Последняя задача завершилась ошибкой:\n{error_message or ''}".rstrip()
    return f"Последняя задача: {status}."


def format_review_card(card: ReviewCardDTO) -> str:
    if isinstance(card, WordCard):
        return "\n".join(
            [
                f"Слово: {card.surface_form}",
                f"Лемма: {card.lemma}",
                "",
                f"EN: {card.meaning_en}",
                f"RU: {card.meaning_ru}",
                "",
                "Example:",
                card.usage_note,
                "",
                "Перевод:",
                card.usage_note_translation,
            ]
        )

    if isinstance(card, PhraseCard):
        return "\n".join(
            [
                f"Фраза: {card.phrase}",
                "",
                "Функция:",
                card.function,
                "",
                f"EN: {card.meaning_en}",
                f"RU: {card.meaning_ru}",
                "",
                "Example:",
                card.example,
                "",
                "Перевод:",
                card.example_translation,
            ]
        )

    if isinstance(card, RuleCard):
        return "\n".join(
            [
                "Rule:",
                card.rule_en,
                "",
                "Правило:",
                card.rule_ru,
                "",
                "Example:",
                card.example,
                "",
                "Перевод:",
                card.example_translation,
            ]
        )

    raise TypeError(f"Unsupported review card type: {type(card)!r}")


format_item = format_review_card
