"""Production-ready LLM wrappers with Langfuse observability.

This example shows three production-oriented patterns:
- Return structured response objects from provider calls.
- Record model, token, cost, latency, and metadata on generation observations.
- Propagate trace-level user/session/tags through a higher-level workflow.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from openai import OpenAI

load_dotenv()

# Provider clients read their API keys from environment variables.
anthropic_client = Anthropic()
openai_client = OpenAI()

# Langfuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.
langfuse = get_client()


@dataclass
class LLMResponse:
    """Normalized response shape used across providers."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    duration_ms: float
    cost: float


# Example standard prices per 1M tokens, last checked 2026-05-22.
# This intentionally covers both the current course defaults and a few common
# alternatives. It does not include prompt caching, batch, flex, priority, long
# context, regional, or tool-specific pricing modifiers.
PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude API model IDs.
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # OpenAI standard short-context prices.
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost from token usage."""
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


@observe(name="call_claude", as_type="generation")
def call_claude(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    system: str | None = None,
    max_tokens: int = 1024,
    metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Call Claude and enrich the active Langfuse generation."""
    start = time.perf_counter()

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = anthropic_client.messages.create(**kwargs)

    duration_ms = (time.perf_counter() - start) * 1000
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    content = response.content[0].text

    # Because this function is observed as a generation, model-specific fields
    # belong on the current generation rather than generic span metadata.
    langfuse.update_current_generation(
        model=model,
        input=[{"role": "user", "content": prompt}],
        output=content,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
        cost_details={"total": cost},
        metadata={
            **(metadata or {}),
            "provider": "anthropic",
            "duration_ms": duration_ms,
            "stop_reason": response.stop_reason,
        },
    )

    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        duration_ms=duration_ms,
        cost=cost,
    )


@observe(name="call_openai", as_type="generation")
def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    system: str | None = None,
    max_tokens: int = 1024,
    metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Call OpenAI and enrich the active Langfuse generation."""
    start = time.perf_counter()

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = openai_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )

    duration_ms = (time.perf_counter() - start) * 1000
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    content = response.choices[0].message.content or ""

    langfuse.update_current_generation(
        model=model,
        input=messages,
        output=content,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
        cost_details={"total": cost},
        metadata={
            **(metadata or {}),
            "provider": "openai",
            "duration_ms": duration_ms,
        },
    )

    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        duration_ms=duration_ms,
        cost=cost,
    )


@observe(name="compare_models", as_type="span")
def compare_models(
    prompt: str,
    user_id: str = "demo-user",
    session_id: str | None = "demo-session",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Compare Claude and OpenAI for the same prompt in one trace."""
    trace_tags = tags or ["production-example", "model-comparison"]

    # These attributes are applied to this trace and propagated to child
    # generations so the run can be filtered by user, session, tags, and metadata.
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=trace_tags,
        metadata={"comparison": True},
        trace_name="llm-model-comparison",
    ):
        claude_response = call_claude(
            prompt,
            metadata={"comparison_role": "candidate_a"},
        )
        openai_response = call_openai(
            prompt,
            metadata={"comparison_role": "candidate_b"},
        )

    total_cost = claude_response.cost + openai_response.cost
    total_duration_ms = claude_response.duration_ms + openai_response.duration_ms

    langfuse.update_current_span(
        output={
            "claude": asdict(claude_response),
            "openai": asdict(openai_response),
            "total_cost": total_cost,
            "total_duration_ms": total_duration_ms,
        },
        metadata={
            "comparison": True,
            "total_cost": total_cost,
            "total_duration_ms": total_duration_ms,
        },
    )

    return {
        "claude": claude_response,
        "openai": openai_response,
        "total_cost": total_cost,
        "total_duration_ms": total_duration_ms,
    }


if __name__ == "__main__":
    result = compare_models("Explain the theory of relativity in simple terms.")

    print("Claude:", result["claude"].content)
    print("OpenAI:", result["openai"].content)
    print(f"Total cost: ${result['total_cost']:.6f}")
    print(f"Total duration: {result['total_duration_ms']:.0f}ms")

    # Always flush in scripts and notebooks so buffered observations are sent.
    langfuse.flush()
