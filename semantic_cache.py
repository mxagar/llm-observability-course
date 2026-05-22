"""Semantic cache for LLM responses with Langfuse tracing.

Semantic caching stores embeddings of previous queries and reuses the cached
answer when a new query is similar enough. This avoids redundant LLM calls for
questions that mean the same thing but use different wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import anthropic
import chromadb
from dotenv import load_dotenv
from langfuse import get_client, observe
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "chroma_cache_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL = "claude-sonnet-4-20250514"

anthropic_client = anthropic.Anthropic()
langfuse = get_client()


@dataclass
class CacheLookup:
    """Result returned by the semantic cache lookup."""

    response: str
    similarity: float
    cached_query: str
    age_seconds: float


@observe(name="call_claude_for_cache_miss", as_type="generation")
def call_claude(query: str) -> str:
    """Call Claude only when the semantic cache misses."""
    response = anthropic_client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
    )
    response_text = response.content[0].text

    langfuse.update_current_generation(
        model=GENERATION_MODEL,
        input=[{"role": "user", "content": query}],
        output=response_text,
        usage_details={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "total": response.usage.input_tokens + response.usage.output_tokens,
        },
        metadata={"cache_hit": False},
    )
    return response_text


class SemanticCache:
    """Persistent ChromaDB-backed semantic response cache."""

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_hours: int = 24,
        persist_directory: str | Path = CACHE_DIR,
    ) -> None:
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="llm_cache",
            metadata={"hnsw:space": "cosine"},
        )
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)

    def _get_embedding(self, text: str) -> list[float]:
        return self.encoder.encode(text).tolist()

    def _is_expired(self, timestamp: str) -> tuple[bool, float]:
        cached_time = datetime.fromisoformat(timestamp)
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - cached_time
        return age > self.ttl, age.total_seconds()

    @observe(name="semantic_cache_get", as_type="retriever")
    def get(self, query: str) -> CacheLookup | None:
        """Return a cached response when a semantically similar query exists."""
        query_embedding = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        if not documents[0]:
            langfuse.update_current_span(
                output={"cache_hit": False},
                metadata={"reason": "empty_cache", "threshold": self.threshold},
            )
            return None

        distance = distances[0][0]
        similarity = 1 - distance
        metadata = metadatas[0][0]
        expired, age_seconds = self._is_expired(metadata["timestamp"])

        if similarity < self.threshold:
            langfuse.update_current_span(
                output={"cache_hit": False},
                metadata={
                    "reason": "below_threshold",
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "cached_query": documents[0][0],
                },
            )
            return None

        if expired:
            langfuse.update_current_span(
                output={"cache_hit": False},
                metadata={
                    "reason": "expired",
                    "similarity": similarity,
                    "ttl_hours": self.ttl.total_seconds() / 3600,
                    "age_seconds": age_seconds,
                    "cached_query": documents[0][0],
                },
            )
            return None

        cached_response = json.loads(metadata["response"])
        lookup = CacheLookup(
            response=cached_response,
            similarity=similarity,
            cached_query=documents[0][0],
            age_seconds=age_seconds,
        )

        langfuse.update_current_span(
            output={
                "cache_hit": True,
                "cached_query": lookup.cached_query,
                "response_preview": lookup.response[:240],
            },
            metadata={
                "similarity": lookup.similarity,
                "threshold": self.threshold,
                "age_seconds": lookup.age_seconds,
            },
        )
        return lookup

    @observe(name="semantic_cache_set", as_type="embedding")
    def set(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a query embedding and response for future semantic matches."""
        query_embedding = self._get_embedding(query)
        doc_id = hashlib.sha256(query.encode("utf-8")).hexdigest()
        timestamp = datetime.now(timezone.utc).isoformat()

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[query_embedding],
            documents=[query],
            metadatas=[
                {
                    "response": json.dumps(response),
                    "timestamp": timestamp,
                    **(metadata or {}),
                }
            ],
        )

        langfuse.update_current_span(
            output={"cached": True, "doc_id": doc_id},
            metadata={
                "query_length": len(query),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_dim": len(query_embedding),
                "timestamp": timestamp,
            },
        )


cache = SemanticCache(similarity_threshold=0.85)


@observe(name="cached_llm_call", as_type="span")
def cached_llm_call(query: str) -> str:
    """Check semantic cache before calling the LLM."""
    cached = cache.get(query)
    if cached:
        langfuse.update_current_span(
            output={"response": cached.response},
            metadata={
                "cache_hit": True,
                "similarity": cached.similarity,
                "cached_query": cached.cached_query,
                "api_call_saved": True,
            },
        )
        print(f"  CACHE HIT - similarity: {cached.similarity:.2%}")
        return cached.response

    print("  CACHE MISS - calling Claude API")
    response = call_claude(query)
    cache.set(query, response, metadata={"source": "claude"})

    langfuse.update_current_span(
        output={"response": response},
        metadata={"cache_hit": False, "api_call_saved": False},
    )
    return response


@observe(name="simulate_semantic_cache", as_type="span")
def simulate_semantic_cache() -> dict[str, float | int]:
    """Demonstrate semantic cache hits with similar questions."""
    question_groups = [
        {
            "topic": "Python Programming",
            "questions": [
                "What is a Python list comprehension?",
                "Explain list comprehensions in Python",
                "How do list comprehensions work in Python?",
                "What are Python list comprehensions and how to use them?",
            ],
        },
        {
            "topic": "Machine Learning",
            "questions": [
                "What is the difference between supervised and unsupervised learning?",
                "Explain supervised vs unsupervised machine learning",
                "How does supervised learning differ from unsupervised learning?",
                "Compare supervised and unsupervised learning in ML",
            ],
        },
        {
            "topic": "API Concepts",
            "questions": [
                "What is a REST API?",
                "Explain what REST APIs are",
                "What does REST API mean?",
                "Can you describe what a RESTful API is?",
            ],
        },
    ]

    total_queries = 0
    cache_hits = 0
    cache_misses = 0

    for group in question_groups:
        print(f"\nTopic: {group['topic']}")
        for question in group["questions"]:
            total_queries += 1
            print(f'Query {total_queries}: "{question}"')

            before = cache.get(question)
            if before:
                cache_hits += 1
            else:
                cache_misses += 1

            response = cached_llm_call(question)
            display_response = response[:120] + "..." if len(response) > 120 else response
            print(f"  Response: {display_response}")

    hit_rate = (cache_hits / total_queries * 100) if total_queries else 0.0
    summary = {
        "total_queries": total_queries,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hit_rate_pct": round(hit_rate, 2),
        "api_calls_saved": cache_hits,
    }

    langfuse.update_current_span(output=summary, metadata=summary)

    print("\nCache performance summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return summary


if __name__ == "__main__":
    simulate_semantic_cache()

    # Flush in short-lived scripts so spans are sent to Langfuse.
    langfuse.flush()
