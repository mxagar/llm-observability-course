"""Smart model router with Langfuse instrumentation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import get_client, observe
from openai import OpenAI

load_dotenv()


class TaskType(Enum):
    SIMPLE = "simple"  # Yes/no answers and basic classification.
    MODERATE = "moderate"  # Summarization and information extraction.
    COMPLEX = "complex"  # Analysis, comparison, and reasoning.
    CODE = "code"  # Code generation or debugging.
    CREATIVE = "creative"  # Creative writing tasks.


@dataclass
class ModelCallResult:
    """Normalized response data across Anthropic and OpenAI calls."""

    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    duration_ms: float


PRICING = {
    # Prices are USD per 1M tokens; keep this table aligned with vendor pricing pages.
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


class ModelRouter:
    """Route requests to the lowest-cost model that should handle the task."""

    MODELS = {
        TaskType.SIMPLE: "claude-haiku-4-5-20251001",
        TaskType.MODERATE: "gpt-4o-mini",
        TaskType.CODE: "claude-sonnet-4-6",
        TaskType.COMPLEX: "claude-sonnet-4-6",
        TaskType.CREATIVE: "gpt-4o",
    }

    def classify_task(self, prompt: str) -> TaskType:
        """Classify a prompt with simple, explainable keyword patterns."""

        prompt_lower = prompt.lower()

        simple_patterns = [
            r"\b(yes or no)\b",
            r"\b(true or false)\b",
            r"\b(classify|categorize)\b",
            r"^is (this|it|the)",
            r"\b(which one|choose|select)\b",
        ]
        if any(re.search(pattern, prompt_lower) for pattern in simple_patterns):
            return TaskType.SIMPLE

        code_patterns = [
            r"\b(write|create|generate|fix|debug).*(code|function|class|script)\b",
            r"\b(python|javascript|typescript|java|rust)\b",
            r"```",
        ]
        if any(re.search(pattern, prompt_lower) for pattern in code_patterns):
            return TaskType.CODE

        complex_patterns = [
            r"\b(analyze|evaluate|compare|critique)\b",
            r"\b(why|how).*(work|happen|cause)\b",
            r"\b(pros and cons|trade-?offs)\b",
            r"\b(explain.*(detail|depth))\b",
        ]
        if any(re.search(pattern, prompt_lower) for pattern in complex_patterns):
            return TaskType.COMPLEX

        creative_patterns = [
            r"\b(write|create|compose).*(story|poem|essay|blog)\b",
            r"\b(creative|imaginative|original)\b",
        ]
        if any(re.search(pattern, prompt_lower) for pattern in creative_patterns):
            return TaskType.CREATIVE

        return TaskType.MODERATE

    def route(self, prompt: str, override_model: Optional[str] = None) -> str:
        """Return an explicit override or the model mapped to the detected task."""

        if override_model:
            return override_model

        task_type = self.classify_task(prompt)
        return self.MODELS[task_type]


langfuse = get_client()
router = ModelRouter()
anthropic_client = Anthropic()
openai_client = OpenAI()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate provider cost from the local pricing table."""

    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


@observe(name="call_claude_routed", as_type="generation")
def call_claude(prompt: str, model: str, max_tokens: int = 1024) -> ModelCallResult:
    """Call Anthropic and record the provider request as a Langfuse generation."""

    start = time.perf_counter()
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = input_tokens + output_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)
    duration_ms = (time.perf_counter() - start) * 1000

    langfuse.update_current_generation(
        model=model,
        input=[{"role": "user", "content": prompt}],
        output=content,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
        cost_details={"total": cost},
        metadata={"provider": "anthropic", "duration_ms": duration_ms},
    )

    return ModelCallResult(
        content=content,
        provider="anthropic",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost=cost,
        duration_ms=duration_ms,
    )


@observe(name="call_openai_routed", as_type="generation")
def call_openai(prompt: str, model: str) -> ModelCallResult:
    """Call OpenAI and record the provider request as a Langfuse generation."""

    start = time.perf_counter()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = response.usage.total_tokens if response.usage else 0
    cost = estimate_cost(model, input_tokens, output_tokens)
    duration_ms = (time.perf_counter() - start) * 1000

    langfuse.update_current_generation(
        model=model,
        input=[{"role": "user", "content": prompt}],
        output=content,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
        cost_details={"total": cost},
        metadata={"provider": "openai", "duration_ms": duration_ms},
    )

    return ModelCallResult(
        content=content,
        provider="openai",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost=cost,
        duration_ms=duration_ms,
    )


@observe(name="routed_llm_call", as_type="span")
def routed_llm_call(prompt: str, override_model: Optional[str] = None) -> str:
    """Classify the prompt, route it, call the provider, and trace the decision."""

    task_type = router.classify_task(prompt)
    selected_model = router.route(prompt, override_model)

    if selected_model.startswith("claude"):
        result = call_claude(prompt, model=selected_model)
    else:
        result = call_openai(prompt, model=selected_model)

    langfuse.update_current_span(
        input={"prompt": prompt, "override_model": override_model},
        output={"content": result.content},
        metadata={
            "task_type": task_type.value,
            "routed_model": selected_model,
            "override_used": override_model is not None,
            "provider": result.provider,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "estimated_cost": result.estimated_cost,
            "duration_ms": result.duration_ms,
        },
    )

    return result.content


if __name__ == "__main__":
    examples = [
        "Is 2 + 2 = 4? Yes or no",
        "Write a Python function to sort a list",
    ]

    for prompt in examples:
        print(f"\nPrompt: {prompt}")
        print(routed_llm_call(prompt))

    langfuse.flush()
