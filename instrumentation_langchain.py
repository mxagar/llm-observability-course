"""LangChain integration with Langfuse.

Langfuse can trace LangChain chains through its CallbackHandler. The handler
captures prompt formatting, model calls, token usage, latency, metadata, tags,
and nested runnable steps without manually instrumenting every LangChain object.
"""

from __future__ import annotations

from inspect import signature
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse, get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

load_dotenv()

# Langfuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.
langfuse = get_client()


def create_langfuse_handler(trace_seed: str) -> CallbackHandler:
    """Create a Langfuse LangChain callback handler.

    Newer Langfuse SDK versions accept `langfuse_client=...`; some installed
    versions accept only `trace_context=...`. This keeps the example compatible
    while still using the current callback-based integration pattern.
    """
    trace_context = {"trace_id": Langfuse.create_trace_id(seed=trace_seed)}
    handler_params = signature(CallbackHandler).parameters

    if "langfuse_client" in handler_params:
        return CallbackHandler(
            langfuse_client=langfuse,
            trace_context=trace_context,
        )

    return CallbackHandler(trace_context=trace_context)


@observe(name="run_langchain_example", as_type="chain")
def run_langchain_example(
    topic: str = "quantum computing",
    user_id: str = "demo-user",
    session_id: str | None = "langchain-demo-session",
) -> str:
    """Run a LangChain LCEL chain and trace it with Langfuse."""
    handler = create_langfuse_handler(trace_seed=f"langchain-{topic}")

    # Trace-level attributes are propagated to the observed wrapper and child
    # LangChain callback observations where supported by the SDK.
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=["langchain", "callback-handler", "demo"],
        metadata={"topic": topic, "framework": "langchain"},
        trace_name="langchain-demo",
    ):
        # ChatAnthropic is the LangChain chat-model wrapper for Anthropic.
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")

        # The prompt variable name must match the dict passed to chain.invoke().
        prompt = ChatPromptTemplate.from_template(
            "Explain {topic} in simple terms."
        )
        chain = prompt | llm

        # Attach the Langfuse callback at invocation time. The callback captures
        # the prompt step, model generation, token usage, latency, metadata, and tags.
        response = chain.invoke(
            {"topic": topic},
            config={
                "callbacks": [handler],
                "metadata": {"use_case": "langchain_example"},
                "tags": ["course", "langchain"],
            },
        )

    langfuse.update_current_span(
        output={"content": response.content},
        metadata={"topic": topic},
    )
    return response.content


if __name__ == "__main__":
    answer = run_langchain_example("quantum computing")
    print(answer)

    # Always flush in scripts and notebooks so buffered observations are sent.
    langfuse.flush()
