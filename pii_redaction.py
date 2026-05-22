"""PII redaction for Langfuse observability."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import anthropic
from dotenv import load_dotenv
from langfuse import get_client, observe

load_dotenv()

GENERATION_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

langfuse = get_client()
anthropic_client = anthropic.Anthropic()


@dataclass
class RedactionResult:
    """Redacted text plus lightweight metadata for audit/debugging."""

    text: str
    redaction_counts: dict[str, int]


class PIIRedactor:
    """Redact common PII patterns before logging data to observability tools."""

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    def redact_with_counts(self, text: str) -> RedactionResult:
        """Redact all known PII patterns and count what was removed."""

        result = text
        counts: dict[str, int] = {}

        for pii_type, pattern in self.PATTERNS.items():
            result, count = re.subn(
                pattern,
                f"[REDACTED_{pii_type.upper()}]",
                result,
            )
            if count:
                counts[pii_type] = count

        return RedactionResult(text=result, redaction_counts=counts)

    def redact(self, text: str) -> str:
        """Return only the redacted text."""

        return self.redact_with_counts(text).text

    def redact_data(self, data: Any) -> Any:
        """Recursively redact strings inside dicts and lists."""

        if isinstance(data, str):
            return self.redact(data)
        if isinstance(data, dict):
            return {key: self.redact_data(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self.redact_data(value) for value in data]
        if isinstance(data, tuple):
            return tuple(self.redact_data(value) for value in data)
        return data


redactor = PIIRedactor()


@observe(name="call_claude_secure", as_type="generation", capture_input=False, capture_output=False)
def call_claude(prompt: str) -> str:
    """Call Claude while logging only redacted prompt/response data."""

    response = anthropic_client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.content[0].text

    redacted_prompt = redactor.redact_with_counts(prompt)
    redacted_response = redactor.redact_with_counts(response_text)

    langfuse.update_current_generation(
        model=GENERATION_MODEL,
        input=[{"role": "user", "content": redacted_prompt.text}],
        output=redacted_response.text,
        usage_details={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "total": response.usage.input_tokens + response.usage.output_tokens,
        },
        metadata={
            "pii_redaction_enabled": True,
            "input_redactions": redacted_prompt.redaction_counts,
            "output_redactions": redacted_response.redaction_counts,
        },
    )

    return response_text


@observe(name="secure_llm_call", as_type="span", capture_input=False, capture_output=False)
def secure_llm_call(prompt: str) -> str:
    """Call an LLM while preventing raw PII from being stored in Langfuse."""

    redacted_prompt = redactor.redact_with_counts(prompt)
    response_text = call_claude(prompt)
    redacted_response = redactor.redact_with_counts(response_text)

    langfuse.update_current_span(
        input={"prompt": redacted_prompt.text},
        output={"response": redacted_response.text},
        metadata={
            "pii_redaction_enabled": True,
            "input_redactions": redacted_prompt.redaction_counts,
            "output_redactions": redacted_response.redaction_counts,
        },
    )

    return response_text


if __name__ == "__main__":
    test_prompt = """
    Please help me with my account. My email is john.doe@example.com,
    my phone number is 555-123-4567, my SSN is 123-45-6789,
    and my card number is 4242 4242 4242 4242.
    """

    try:
        print("Testing PII redaction with an LLM call...")
        result = secure_llm_call(test_prompt)
        print(f"Response: {result}")
    finally:
        langfuse.flush()
