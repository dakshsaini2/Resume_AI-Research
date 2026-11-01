"""
gemini_client.py

Robust, resilient client for the Gemini REST API.
Implements:
1. Bounded retries with exponential backoff and jitter for transient errors (503, 429, 500, 502, 504).
2. Schema-constrained JSON generation (responseMimeType="application/json").
3. Robust JSON extraction, multi-stage fallback cleaning, and parse error retries.
4. Explicit diagnostic error tracking and structured execution logs.
"""

import os
import re
import json
import time
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
import requests
from pydantic import BaseModel, ValidationError

from datetime import datetime, timezone

logger = logging.getLogger("gemini_client")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK_MODELS = [DEFAULT_MODEL, "gemini-flash-lite-latest", "gemini-pro-latest"]


class GeminiAPIException(Exception):
    """Raised when an unrecoverable Gemini API error occurs."""
    def __init__(self, message: str, status_code: Optional[int] = None, payload: Optional[Any] = None, attempts: int = 1):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload
        self.attempts = attempts


class GeminiQuotaExhaustedException(GeminiAPIException):
    """Raised when the Gemini API quota has been completely exhausted (e.g. daily quota reached)."""
    def __init__(self, message: str, retry_delay_seconds: Optional[float] = None, payload: Optional[Any] = None):
        super().__init__(message, status_code=429, payload=payload)
        self.retry_delay_seconds = retry_delay_seconds


class GeminiJSONParseException(Exception):
    """Raised when the LLM response cannot be parsed into the expected JSON/Pydantic schema."""
    def __init__(self, message: str, raw_text: str, attempts: int = 1):
        super().__init__(message)
        self.raw_text = raw_text
        self.attempts = attempts


def extract_retry_delay(err_payload: Any) -> Optional[float]:
    """Extracts retryDelay from Google RPC error details if present."""
    if not isinstance(err_payload, dict):
        return None
    try:
        details = err_payload.get("error", {}).get("details", [])
        for d in details:
            if isinstance(d, dict) and "retryDelay" in d:
                raw_delay = str(d["retryDelay"]).rstrip("s")
                return float(raw_delay)
    except Exception:
        pass
    return None


def is_hard_quota_exhausted(err_payload: Any) -> Tuple[bool, Optional[float]]:
    """Determines whether HTTP 429 represents a hard daily quota exhaustion."""
    delay = extract_retry_delay(err_payload)
    if delay is not None and delay > 120.0:
        return True, delay
    if isinstance(err_payload, dict):
        err = err_payload.get("error", {})
        if err.get("status") == "RESOURCE_EXHAUSTED":
            for d in err.get("details", []):
                if isinstance(d, dict) and "QuotaFailure" in d.get("@type", ""):
                    return True, delay
    return False, delay


def sanitize_json_text(text: str) -> str:
    """
    Cleans markdown code fences, leading/trailing whitespace,
    and strips common LLM formatting artifacts.
    """
    text = text.strip()
    # Strip markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return text


def repair_and_parse_json(text: str) -> Dict[str, Any]:
    """
    Attempts multi-stage JSON parsing:
    1. Standard json.loads with strict=False (allows unescaped newlines/tabs inside strings)
    2. Regex-based JSON boundary extraction
    3. Trailing comma cleanup and newline escaping
    """
    cleaned = sanitize_json_text(text)
    
    # Stage 1: lenient parsing
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError as err1:
        pass

    # Stage 2: Extract substring between outer braces { ... }
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            pass

    # Stage 3: Fix common JSON syntax errors like unescaped literal newlines in strings
    # Replace unescaped control characters in string literals
    try:
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r",\s*([\}\]])", r"\1", cleaned)
        return json.loads(fixed, strict=False)
    except json.JSONDecodeError as err_final:
        raise GeminiJSONParseException(
            f"Failed to parse JSON response: {str(err_final)}",
            raw_text=text
        )


def execute_gemini_request(
    prompt: str,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_output_tokens: int = 4096,
    enforce_json_mime: bool = True,
    system_instruction: Optional[str] = None,
    timeout: float = 60.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Performs a single HTTP POST to Gemini generateContent.
    Returns (raw_text, response_metadata).
    """
    clean_model = model_name if not model_name.startswith("models/") else model_name[7:]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"

    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if enforce_json_mime:
        generation_config["responseMimeType"] = "application/json"

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    response = requests.post(url, json=payload, timeout=timeout)
    status = response.status_code

    if status != 200:
        try:
            err_json = response.json()
        except Exception:
            err_json = {"raw": response.text}
        raise GeminiAPIException(
            f"Gemini API returned HTTP {status}: {json.dumps(err_json)}",
            status_code=status,
            payload=err_json
        )

    resp_json = response.json()
    candidates = resp_json.get("candidates", [])
    if not candidates:
        raise GeminiAPIException("No candidates returned from Gemini API", status_code=200, payload=resp_json)

    first_cand = candidates[0]
    finish_reason = first_cand.get("finishReason", "UNKNOWN")
    parts = first_cand.get("content", {}).get("parts", [])
    if not parts or "text" not in parts[0]:
        raise GeminiAPIException(f"Empty parts or missing text (finishReason={finish_reason})", status_code=200, payload=resp_json)

    raw_text = parts[0]["text"]
    return raw_text, {"finish_reason": finish_reason, "usage": resp_json.get("usageMetadata", {})}


def call_gemini_with_retry(
    prompt: str,
    api_key: str,
    schema_model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    max_backoff: float = 45.0,
    max_output_tokens: int = 4096,
    system_instruction: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes a Gemini call with bounded exponential backoff on transient errors (503, 429, 500, etc.)
    and multi-stage parse validation against schema_model (if provided).

    Returns:
        (parsed_dict, telemetry_dict)
    """
    if not api_key:
        raise ValueError("Google API key is missing or empty.")

    target_models = [model_name] if model_name else [DEFAULT_MODEL]
    for fb in FALLBACK_MODELS:
        if fb not in target_models:
            target_models.append(fb)

    total_attempts = 0
    last_error: Optional[Exception] = None

    for current_model in target_models:
        backoff = initial_backoff

        for attempt in range(1, max_retries + 1):
            total_attempts += 1
            try:
                raw_text, meta = execute_gemini_request(
                    prompt=prompt,
                    api_key=api_key,
                    model_name=current_model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    enforce_json_mime=True,
                    system_instruction=system_instruction,
                )

                # Parse JSON
                parsed_data = repair_and_parse_json(raw_text)

                # Validate schema if provided
                if schema_model:
                    validated_obj = schema_model(**parsed_data)
                    parsed_data = validated_obj.dict()

                telemetry = {
                    "model_used": current_model,
                    "attempts": total_attempts,
                    "finish_reason": meta.get("finish_reason"),
                    "raw_output_length": len(raw_text),
                }
                return parsed_data, telemetry

            except (GeminiAPIException, requests.exceptions.RequestException) as api_err:
                last_error = api_err
                status = getattr(api_err, "status_code", None)

                # If model is not found (404), do not retry the same model—fall back immediately
                if status == 404:
                    logger.warning(f"[Gemini Model Not Found] Model '{current_model}' returned 404. Falling back to alternative model.")
                    break

                # Check for hard daily quota exhaustion
                if status == 429:
                    payload = getattr(api_err, "payload", {})
                    is_hard, retry_sec = is_hard_quota_exhausted(payload)
                    if is_hard:
                        delay_msg = f"{retry_sec:.0f}s" if retry_sec else "daily limit"
                        logger.error(f"[Gemini Daily Quota Exhausted] Hard quota reached ({delay_msg}). Halting retries immediately.")
                        raise GeminiQuotaExhaustedException(
                            f"Daily Gemini API quota exhausted (Google RPC retryDelay: {delay_msg}).",
                            retry_delay_seconds=retry_sec,
                            payload=payload
                        )

                # Check if error is transient
                is_transient = status in [429, 500, 502, 503, 504] or isinstance(api_err, (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError
                ))

                if not is_transient and status is not None and status not in [400]:
                    # Non-transient server error, bubble up
                    logger.error(f"[Gemini Non-Transient Error] HTTP {status}: {api_err}")
                    raise

                # Add jitter: +/- 25% of backoff
                jitter = random.uniform(0.1, 0.8)
                sleep_duration = min(max_backoff, backoff + jitter)

                logger.warning(
                    f"[Gemini Transient Error] Model '{current_model}' (HTTP {status}) attempt {attempt}/{max_retries}. "
                    f"Retrying in {sleep_duration:.1f}s..."
                )
                time.sleep(sleep_duration)
                backoff = min(max_backoff, backoff * 2.0)

            except (GeminiJSONParseException, ValidationError) as parse_err:
                last_error = parse_err
                logger.warning(
                    f"[Gemini Parse/Validation Error] Model '{current_model}' attempt {attempt}/{max_retries}: {parse_err}. Retrying..."
                )
                time.sleep(1.0 + random.uniform(0.1, 0.5))

    # All retries and fallback models exhausted
    raise GeminiAPIException(
        f"Exhausted all {total_attempts} attempts across models {target_models}. Last error: {last_error}",
        payload={"last_error": str(last_error), "total_attempts": total_attempts}
    )
