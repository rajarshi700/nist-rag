import json
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from pydantic import ValidationError

from .config import AppError, Settings


def retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                return max(0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return 2**attempt + random.uniform(0, 0.3)


class GroqLLM:
    def __init__(self, settings: Settings, *, client=None, sleep=time.sleep):
        if not settings.api_key:
            raise AppError(
                "Set GROQ_API_KEY in .env to use chat. Ingest and search do not need a key."
            )
        self.settings = settings
        self.sleep = sleep
        self.client = client or httpx.Client(timeout=settings.timeout)

    def complete(self, schema, system: str, payload: dict, *, max_tokens=900):
        instructions = (
            system
            + "\nReturn only a JSON object matching this schema:\n"
            + json.dumps(schema.model_json_schema())
        )
        body = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.settings.groq_model.startswith("openai/gpt-oss-"):
            body.update(reasoning_effort="low", include_reasoning=False)
            body["max_completion_tokens"] = max_tokens + 1024
        for attempt in range(self.settings.max_api_retries + 1):
            delay = None
            try:
                response = self.client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.api_key}"},
                    json=body,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    delay = retry_delay(response.headers.get("retry-after"), attempt)
                    if attempt == self.settings.max_api_retries:
                        raise AppError(
                            "The LLM is rate-limited or unavailable. Please try again later."
                        )
                    if delay > self.settings.max_retry_wait:
                        raise AppError(
                            f"The API asks for a wait of {delay:.0f} seconds. "
                            "Please retry after that."
                        )
                elif response.status_code in {401, 403}:
                    raise AppError(
                        "The LLM rejected the API key. Check GROQ_API_KEY and account access."
                    )
                elif response.is_error:
                    raise AppError(
                        f"The LLM rejected the request (HTTP {response.status_code}). "
                        "Check GROQ_MODEL, model access, and JSON mode support in the Groq console."
                    )
                else:
                    result = response.json()["choices"][0]
                    if result.get("finish_reason") == "length":
                        raise ValueError("truncated response")
                    return schema.model_validate_json(result["message"]["content"])
            except httpx.TransportError as exc:
                if attempt == self.settings.max_api_retries:
                    raise AppError(
                        "Could not reach the LLM API. Check the connection and retry."
                    ) from exc
            except (ValidationError, ValueError, KeyError, IndexError, TypeError) as exc:
                if attempt == self.settings.max_api_retries:
                    raise AppError(
                        "The LLM returned an invalid structured response. Please try again."
                    ) from exc
                # A retry gets the same task and a short formatting reminder, not invented content.
                body["messages"][0]["content"] = instructions + (
                    "\nThe previous response was invalid. Use the exact JSON schema, "
                    "keep it concise, and do not use Markdown fences."
                )
            self.sleep(
                delay
                if delay is not None
                else min(retry_delay(None, attempt), self.settings.max_retry_wait)
            )
        raise AppError("The LLM request failed.")

    def close(self):
        self.client.close()
