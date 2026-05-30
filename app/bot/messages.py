from __future__ import annotations

import json

from app.db.models import UserLearningProfile
from app.services.review import PhraseCard, ReviewCardDTO, RuleCard, WordCard
from app.services.profile_schemas import LearningProfilePayload
from app.services.sources.availability import SourceAvailability
from app.services.sources.types import SourceType


START_MESSAGE = """Привет! Я бот для изучения английского по текстам.

Пришли мне английский текст или ссылку на поддерживаемый источник, и я извлеку:
— важные слова;
— полезные фразы;
— правила и конструкции.

Команды:
/profile — посмотреть учебный профиль
/profile_edit — изменить учебный профиль
/profile_cancel — отменить изменение профиля
/sources — доступные источники
/review_words — повторить слова
/review_phrases — повторить фразы
/review_rules — повторить правила
/status — статус обработки"""

HELP_MESSAGE = START_MESSAGE

USER_LIMIT_REACHED = "Доступ ограничен первыми 5 пользователями."
SEND_SOURCE_INPUT = "Пришли английский текст или ссылку на Reddit-пост."
PROCESSING_ALREADY_ACTIVE = "У тебя уже есть задача в обработке. Дождись её завершения."
QUEUED = "Принял Reddit-ссылку. Загружу пост и комментарии, затем извлеку слова, фразы и правила."
QUEUED_REDDIT = QUEUED
QUEUED_TEXT = "Принял текст. Начинаю извлекать слова, фразы и правила."
UNKNOWN_SOURCE = "Я пока не умею работать с этим источником. Пришли текст напрямую или ссылку на Reddit-пост."
REDDIT_SOURCE_UNAVAILABLE = "Reddit API сейчас не настроен. Пришли текст поста вручную."
NO_ITEMS = "Пока нет сохранённых карточек этого типа. Сначала пришли материал для обработки."
RATING_SAVED = "Оценка сохранена."
REVIEW_COMPLETED = "Повторение завершено."
REVIEW_SESSION_EXPIRED = "Сессия повторения устарела. Запусти повторение заново."
REVIEW_NOT_FOUND = "Не нашёл активную сессию повторения."
INVALID_RATING = "Оценка должна быть от 1 до 5."
NO_JOBS = "Задач пока нет."
PROFILE_SETUP_REQUEST = """Привет! Сначала настроим твой учебный профиль.

Напиши одним сообщением:
— твой уровень английского, если знаешь;
— что хочешь прокачать;
— для чего тебе английский;
— какие темы тебе особенно интересны.

Пример:
B1. Хочу читать Reddit и статьи про machine learning, лучше понимать фразовые глаголы, разговорные фразы и технические термины. Грамматику хочу только самую полезную."""
PROFILE_SETUP_REQUIRED = """Перед обработкой текста нужно настроить учебный профиль.

Напиши одним сообщением свой уровень и цели."""
PROFILE_EDIT_REQUEST = (
    "Напиши новый профиль одним сообщением: уровень, цели, интересные темы "
    "и что особенно хочешь прокачать."
)
PROFILE_GENERATING = "Собираю учебный профиль. Это займёт немного времени."
PROFILE_SAVED = "Профиль сохранён."
PROFILE_CANCELLED = "Настройка профиля отменена."
PROFILE_NOT_FOUND = "Профиль пока не настроен."
PROFILE_GENERATION_FAILED = (
    "Не получилось собрать профиль. "
    "Попробуй описать уровень и цели чуть проще одним сообщением."
)


def start_message(has_reddit_credentials: bool) -> str:
    return START_MESSAGE


def help_message(has_reddit_credentials: bool) -> str:
    return start_message(has_reddit_credentials)


def format_sources_status(sources: list[SourceAvailability]) -> str:
    lines = ["Доступные источники:", ""]
    for source in sources:
        status = "настроено" if source.is_configured else "не настроено"
        lines.append(f"{source.display_name}: {status}")

    reddit_available = any(
        source.source_type == SourceType.REDDIT_POST and source.is_configured
        for source in sources
    )
    reddit_hint = "— ссылку на Reddit-пост."
    if not reddit_available:
        reddit_hint = "— ссылку на Reddit-пост, если Reddit настроен."

    lines.extend(
        [
            "",
            "Можно прислать:",
            "— обычный английский текст;",
            reddit_hint,
        ]
    )
    return "\n".join(lines)


def format_job_status(
    status: str,
    error_message: str | None = None,
    source_type: str | None = None,
) -> str:
    if status == "queued":
        message = "Последняя задача: queued."
    elif status == "processing":
        message = "Последняя задача: processing."
    elif status == "done":
        message = "Последняя задача завершена успешно."
    elif status == "failed":
        message = f"Последняя задача завершилась ошибкой:\n{error_message or ''}".rstrip()
    else:
        message = f"Последняя задача: {status}."

    if source_type is None:
        return message
    return f"{message}\nИсточник: {_source_display_name(source_type)}."


def format_profile(profile: UserLearningProfile) -> str:
    payload = LearningProfilePayload.model_validate(json.loads(profile.profile_json))
    lines = [
        "Твой учебный профиль:",
        "",
        f"Уровень: {payload.cefr_level}",
        f"Цели: {payload.goals_summary}",
    ]
    if payload.focus_areas:
        lines.append(f"Фокус: {_format_list(payload.focus_areas)}")
    if payload.domain_interests:
        lines.append(f"Темы: {_format_list(payload.domain_interests)}")
    if payload.include:
        lines.append(f"Что включать: {_format_list(payload.include)}")
    if payload.exclude:
        lines.append(f"Что не включать: {_format_list(payload.exclude)}")
    lines.extend(["", "Чтобы изменить профиль, отправь /profile_edit."])
    return "\n".join(lines)


def _format_list(values: list[str]) -> str:
    return "; ".join(values)


def _source_display_name(source_type: str) -> str:
    if source_type == SourceType.MANUAL_TEXT:
        return "текст вручную"
    if source_type == SourceType.REDDIT_POST:
        return "Reddit"
    return source_type


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
