import os
import time
import re
from typing import Type, TypeVar

from pydantic import BaseModel

from ..config import settings
from ..utils.logger import get_logger


logger = get_logger("gemini_client")

T = TypeVar("T", bound=BaseModel)


class GeminiQuotaError(Exception):
    """Raised when Gemini API quota has been exceeded."""
    pass


class GeminiClient:

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY or os.environ.get(
            "GEMINI_API_KEY",
            ""
        )

        self.text_model = settings.GEMINI_TEXT_MODEL
        self.image_model = settings.GEMINI_IMAGE_MODEL

        self._client = None

        if self.api_key:
            try:
                from google import genai

                self._client = genai.Client(
                    api_key=self.api_key
                )

                logger.info(
                    f"Initialized Gemini Client with text model "
                    f"{self.text_model}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to initialize google-genai client: {e}"
                )

        else:
            logger.info(
                "No GEMINI_API_KEY found. "
                "AI services will use intelligent fallbacks."
            )

    @property
    def is_configured(self) -> bool:
        return bool(
            self.api_key and self._client
        )

    def _is_quota_error(self, error: Exception) -> bool:
        """
        Detect Gemini quota/rate-limit errors.
        """

        error_text = str(error).upper()

        return (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "QUOTA EXCEEDED" in error_text
        )

    def _is_temporary_error(self, error: Exception) -> bool:
        """
        Detect temporary Gemini server errors.
        """

        error_text = str(error).upper()

        return (
            "503" in error_text
            or "UNAVAILABLE" in error_text
            or "500" in error_text
            or "INTERNAL" in error_text
        )

    def generate_text(
        self,
        prompt: str,
        system_instruction: str = ""
    ) -> str:
        """
        Generate text using Gemini.

        429 quota errors are not retried because the quota
        cannot be restored by waiting a few seconds.
        """

        if not self.is_configured:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        max_retries = 3
        backoff = 5

        for attempt in range(max_retries):

            try:
                from google.genai import types

                config = types.GenerateContentConfig()

                if system_instruction:
                    config.system_instruction = (
                        system_instruction
                    )

                response = self._client.models.generate_content(
                    model=self.text_model,
                    contents=prompt,
                    config=config
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                logger.info(
                    "Gemini text generation successful."
                )

                return response.text

            except Exception as e:

                # Do NOT retry quota errors
                if self._is_quota_error(e):
                    logger.error(
                        "Gemini API quota exceeded. "
                        "Using fallback generation."
                    )

                    raise GeminiQuotaError(
                        "Gemini API quota exceeded. "
                        "Please wait for the quota to reset "
                        "or check your Gemini API billing/quota."
                    ) from e

                logger.warning(
                    f"Gemini generate_text attempt "
                    f"{attempt + 1}/{max_retries} failed: {e}"
                )

                if attempt == max_retries - 1:
                    raise

                # Retry only temporary server errors
                if self._is_temporary_error(e):
                    wait_time = min(
                        30,
                        backoff * (2 ** attempt)
                    )

                    logger.info(
                        f"Retrying text generation in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

        raise RuntimeError(
            "Failed to generate text."
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: str = ""
    ) -> T:
        """
        Generate structured JSON using Gemini.

        The response is validated using the supplied
        Pydantic schema.
        """

        if not self.is_configured:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        max_retries = 3
        backoff = 5

        for attempt in range(max_retries):

            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                )

                if system_instruction:
                    config.system_instruction = (
                        system_instruction
                    )

                response = self._client.models.generate_content(
                    model=self.text_model,
                    contents=prompt,
                    config=config
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                raw_text = response.text.strip()

                raw_json = self._extract_json(
                    raw_text
                )

                result = schema.model_validate_json(
                    raw_json
                )

                logger.info(
                    "Gemini structured generation successful."
                )

                return result

            except Exception as e:

                # Do NOT retry quota errors
                if self._is_quota_error(e):
                    logger.error(
                        "Gemini API quota exceeded "
                        "during structured generation. "
                        "Using fallback generator."
                    )

                    raise GeminiQuotaError(
                        "Gemini API quota exceeded. "
                        "Please wait for the quota to reset "
                        "or check your Gemini API billing/quota."
                    ) from e

                logger.warning(
                    f"Structured generation attempt "
                    f"{attempt + 1}/{max_retries} failed: {e}"
                )

                if attempt == max_retries - 1:
                    raise

                # Retry temporary Gemini server errors
                if self._is_temporary_error(e):

                    wait_time = min(
                        30,
                        backoff * (2 ** attempt)
                    )

                    logger.info(
                        f"Retrying structured generation in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

        raise RuntimeError(
            "Failed to generate structured data."
        )

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from Gemini response.
        """

        if not text:
            raise ValueError(
                "Empty response received while "
                "extracting JSON."
            )

        text = text.strip()

        # Handle Markdown JSON blocks
        match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and end != -1
            and end > start
        ):
            return text[start:end + 1]

        return text


gemini_client = GeminiClient()