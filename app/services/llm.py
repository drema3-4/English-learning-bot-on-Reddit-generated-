from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.utils.json_parse import JSONParseError, parse_json_object


class LLMError(RuntimeError):
    pass


SYSTEM_PROMPT = """You are a JSON-only assistant for an English-learning Telegram bot.
Follow the user prompt exactly. Return only valid JSON. Do not add markdown or explanations."""


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = AsyncOpenAI(
            api_key=self._settings.openai_api_key or "missing",
            base_url=self._settings.openai_base_url,
        )

    async def complete_json(self, prompt: str) -> str:
        if not self._settings.openai_api_key:
            raise LLMError("OpenAI-compatible API key is missing")

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content if response.choices else ""
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"LLM response parsing failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM returned an empty response")
        return content

    async def extract(self, source_text: str) -> dict[str, Any]:
        try:
            return parse_json_object(await self.complete_json(source_text))
        except JSONParseError as exc:
            raise LLMError(str(exc)) from exc


LlmClient = LLMClient
LlmError = LLMError
