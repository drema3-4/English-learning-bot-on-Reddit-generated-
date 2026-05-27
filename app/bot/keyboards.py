from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def rating_keyboard(session_type: str, review_session_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for score in range(1, 6):
        builder.button(text=str(score), callback_data=f"rate:{session_type}:{review_session_id}:{score}")
    builder.adjust(5)
    return builder.as_markup()
