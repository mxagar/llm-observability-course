"""Prompt optimization helper with Langfuse tracing.

This is intentionally simple: it removes obvious prompt bloat and records the
before/after size so the optimization can be inspected in Langfuse.
"""

from __future__ import annotations

import re

from dotenv import load_dotenv
from langfuse import get_client, observe

load_dotenv()

langfuse = get_client()

# Common phrases that usually add tokens without adding useful instruction.
FILLER_PHRASES = [
    "I want you to",
    "You are a highly intelligent",
    "Please note that",
    "It's important to remember that",
    "In your response, make sure to",
    "As an AI assistant,",
]


def estimate_tokens(text: str) -> int:
    """Estimate token count without adding tokenizer dependencies."""
    return max(1, round(len(text) / 4)) if text else 0


@observe(name="optimize_prompt", as_type="span")
def optimize_prompt(prompt: str) -> str:
    """Remove obvious prompt bloat while preserving unique instructions.

    This is not a replacement for manual prompt review; it only catches common
    filler phrases, excess whitespace, and repeated sentence-level instructions.
    """
    original_prompt = prompt

    # Remove common filler phrases with case-insensitive matching.
    optimized = prompt
    for filler in FILLER_PHRASES:
        optimized = re.sub(re.escape(filler), "", optimized, flags=re.IGNORECASE)

    # Collapse repeated whitespace introduced by removals.
    optimized = " ".join(optimized.split())

    # Remove repeated sentence-level instructions while preserving order.
    sentences = re.split(r"(?<=[.!?])\s+", optimized)
    seen: set[str] = set()
    unique_sentences: list[str] = []
    for sentence in sentences:
        normalized = sentence.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence.strip())

    optimized = " ".join(unique_sentences)

    original_tokens = estimate_tokens(original_prompt)
    optimized_tokens = estimate_tokens(optimized)
    reduction_pct = (
        ((original_tokens - optimized_tokens) / original_tokens) * 100
        if original_tokens
        else 0.0
    )

    # Record the optimization metrics in the current Langfuse span.
    langfuse.update_current_span(
        input={"prompt": original_prompt},
        output={"optimized_prompt": optimized},
        metadata={
            "original_chars": len(original_prompt),
            "optimized_chars": len(optimized),
            "estimated_original_tokens": original_tokens,
            "estimated_optimized_tokens": optimized_tokens,
            "estimated_token_reduction_pct": round(reduction_pct, 2),
            "fillers_checked": FILLER_PHRASES,
        },
    )

    return optimized


if __name__ == "__main__":
    bloated_prompt = """
    As an AI assistant, I want you to explain observability.
    Please note that it's important to remember that in your response, make sure to be accurate.
    In your response, make sure to be accurate.
    Be concise.
    """

    optimized_prompt = optimize_prompt(bloated_prompt)
    print("Original:")
    print(bloated_prompt.strip())
    print("\nOptimized:")
    print(optimized_prompt)

    # Flush in short-lived scripts so the span is sent to Langfuse.
    langfuse.flush()
