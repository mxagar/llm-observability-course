"""RAG pipeline with Langfuse observability.

This example instruments the complete RAG lifecycle:
- Load Markdown documents.
- Split them into chunks.
- Embed and store chunks in ChromaDB.
- Embed a user query.
- Retrieve relevant chunks.
- Build the final context.
- Generate an answer with Claude.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import get_client, observe, propagate_attributes
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL = "claude-sonnet-4-20250514"

# Langfuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.
langfuse = get_client()

# Chroma keeps the vector index on disk so indexing can be reused across runs.
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# Load the embedding model once. all-MiniLM-L6-v2 is small and returns 384 dims.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
anthropic_client = anthropic.Anthropic()


@observe(name="load_and_index_documents", as_type="span")
def load_and_index_documents(docs_dir: str | Path = DOCS_DIR) -> int:
    """Load, chunk, embed, and index Markdown documents."""
    documents = load_markdown_docs(docs_dir)
    if not documents:
        langfuse.update_current_span(
            output={"chunks_indexed": 0},
            metadata={"reason": "no_documents"},
        )
        return 0

    chunks = chunk_documents(documents)
    chunks_indexed = index_chunks(chunks)

    langfuse.update_current_span(
        output={"chunks_indexed": chunks_indexed},
        metadata={"docs_loaded": len(documents), "chunks_created": len(chunks)},
    )
    return chunks_indexed


@observe(name="load_markdown_docs", as_type="span")
def load_markdown_docs(docs_dir: str | Path) -> list[Document]:
    """Load all Markdown files from a directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path}")

    loader = DirectoryLoader(
        str(docs_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    langfuse.update_current_span(
        output={"documents_loaded": len(documents)},
        metadata={"docs_dir": str(docs_path), "glob": "**/*.md"},
    )
    return documents


@observe(name="chunk_documents", as_type="span")
def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into retrieval-sized chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    langfuse.update_current_span(
        output={"chunks": len(chunks)},
        metadata={
            "documents": len(documents),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )
    return chunks


@observe(name="index_chunks", as_type="embedding")
def index_chunks(chunks: list[Document]) -> int:
    """Create embeddings and upsert chunks into ChromaDB."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for index, chunk in enumerate(chunks):
        chunk_id = f"chunk_{index}"
        embedding = embedding_model.encode(chunk.page_content).tolist()

        ids.append(chunk_id)
        documents.append(chunk.page_content)
        metadatas.append(
            {
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_index": index,
            }
        )
        embeddings.append(embedding)

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    langfuse.update_current_span(
        output={"chunks_indexed": len(ids)},
        metadata={
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": len(embeddings[0]) if embeddings else 0,
            "collection": "documents",
        },
    )
    return len(ids)


@observe(name="rag_pipeline", as_type="span")
def rag_pipeline(
    query: str,
    top_k: int = 5,
    user_id: str = "demo-user",
    session_id: str | None = "rag-demo-session",
) -> str:
    """Run the full RAG query pipeline in one trace."""
    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=["rag", "retrieval", "generation"],
        metadata={"top_k": top_k, "embedding_model": EMBEDDING_MODEL_NAME},
        trace_name="rag-query",
    ):
        query_embedding = embed_query(query)
        chunks = retrieve_chunks(query_embedding, top_k=top_k)
        context = build_context(chunks)
        response = generate_response(query, context)

    langfuse.update_current_span(
        output={"response": response},
        metadata={
            "query": query,
            "chunks_retrieved": len(chunks),
            "context_length": len(context),
        },
    )
    return response


@observe(name="embed_query", as_type="embedding")
def embed_query(query: str) -> list[float]:
    """Embed the user query before vector search."""
    embedding = embedding_model.encode(query).tolist()

    langfuse.update_current_span(
        output={"embedding_dim": len(embedding)},
        metadata={"query_length": len(query), "embedding_model": EMBEDDING_MODEL_NAME},
    )
    return embedding


@observe(name="retrieve_chunks", as_type="retriever")
def retrieve_chunks(embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant document chunks from ChromaDB."""
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict[str, Any]] = []
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    for index, document in enumerate(documents[0]):
        chunks.append(
            {
                "content": document,
                "metadata": metadatas[0][index],
                "distance": distances[0][index],
            }
        )

    avg_distance = (
        sum(chunk["distance"] for chunk in chunks) / len(chunks) if chunks else 0.0
    )

    langfuse.update_current_span(
        output={
            "chunks": [
                {
                    "source": chunk["metadata"].get("source", "unknown"),
                    "distance": chunk["distance"],
                    "content_preview": chunk["content"][:240],
                }
                for chunk in chunks
            ]
        },
        metadata={
            "chunks_retrieved": len(chunks),
            "top_k": top_k,
            "avg_distance": avg_distance,
        },
    )
    return chunks


@observe(name="build_context", as_type="span")
def build_context(chunks: list[dict[str, Any]]) -> str:
    """Assemble retrieved chunks into the context sent to the LLM."""
    if not chunks:
        context = "No relevant context found."
    else:
        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk["metadata"].get("source", "unknown")
            context_parts.append(f"[Source {index} - {source}]: {chunk['content']}")
        context = "\n\n".join(context_parts)

    langfuse.update_current_span(
        output={"context_preview": context[:500]},
        metadata={"context_length": len(context), "num_chunks_used": len(chunks)},
    )
    return context


@observe(name="generate_response", as_type="generation")
def generate_response(query: str, context: str) -> str:
    """Generate an answer from the retrieved context using Claude."""
    prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""

    response = anthropic_client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text

    langfuse.update_current_generation(
        model=GENERATION_MODEL,
        input=[{"role": "user", "content": prompt}],
        output=answer,
        usage_details={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "total": response.usage.input_tokens + response.usage.output_tokens,
        },
        metadata={"prompt_length": len(prompt), "context_length": len(context)},
    )
    return answer


if __name__ == "__main__":
    print("Indexing documents from ./docs folder...")
    try:
        num_indexed = load_and_index_documents(DOCS_DIR)
        print(f"Indexed {num_indexed} chunks")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Create a 'docs' folder with .md files first.")
        raise SystemExit(1) from exc

    print("\nQuerying RAG pipeline...")
    result = rag_pipeline("What is LlamaIndex and how do I set it up?", top_k=5)
    print(f"\nResponse:\n{result}")

    # Always flush in scripts and notebooks so buffered observations are sent.
    langfuse.flush()
