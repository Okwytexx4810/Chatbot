"""Simple retrieval layer for a document-based knowledge chatbot."""

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

SUPPORTED_EXTENSIONS = {".md", ".txt"}
CACHE_VERSION = 1


@dataclass
class Chunk:
    source: str
    text: str
    embedding: list[float]


@dataclass
class Match:
    chunk: Chunk
    score: float


class KnowledgeBase:
    def __init__(
        self,
        client: genai.Client,
        directory: str | Path | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent
        self.client = client
        self.directory = Path(directory) if directory else root / "knowledge"
        self.cache_path = Path(cache_path) if cache_path else root / ".kb_cache.json"
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.dimension = int(os.getenv("EMBEDDING_DIM", "768"))
        self.chunk_size = int(os.getenv("KB_CHUNK_SIZE", "1000"))
        self.top_k = int(os.getenv("KB_TOP_K", "4"))
        self.min_similarity = float(os.getenv("KB_MIN_SIMILARITY", "0.55"))
        self.batch_size = 50
        self.documents: list[Path] = []
        self.chunks: list[Chunk] = []

    def refresh(self) -> None:
        """Load documents, reusing cached embeddings when nothing changed."""
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Knowledge directory not found: {self.directory}")

        self.documents = sorted(
            path
            for path in self.directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not self.documents:
            raise FileNotFoundError(
                f"No supported documents in {self.directory}. "
                f"Add .md or .txt files and run again."
            )

        raw_chunks: list[Chunk] = []
        fingerprint = hashlib.sha256()
        fingerprint.update(
            f"{CACHE_VERSION}|{self.embedding_model}|{self.dimension}|{self.chunk_size}".encode()
        )

        for path in self.documents:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            source = str(path.relative_to(self.directory))
            fingerprint.update(source.encode())
            fingerprint.update(text.encode())
            for piece in chunk_text(text, self.chunk_size):
                raw_chunks.append(Chunk(source=source, text=piece, embedding=[]))

        if not raw_chunks:
            raise ValueError("The knowledge base contains no readable text.")

        digest = fingerprint.hexdigest()
        cached = self._load_cache(digest)
        if cached is not None:
            self.chunks = cached
            return

        try:
            vectors = self._embed(
                [chunk.text for chunk in raw_chunks], task_type="RETRIEVAL_DOCUMENT"
            )
        except Exception:
            self.chunks = []
            raise

        for chunk, vector in zip(raw_chunks, vectors):
            chunk.embedding = vector
        self.chunks = raw_chunks
        self._save_cache(digest)

    def retrieve(self, question: str, top_k: int | None = None) -> list[Match]:
        """Return the most relevant chunks for a question."""
        if not self.chunks:
            return []

        query_vector = self._embed([question], task_type="RETRIEVAL_QUERY")[0]
        matches = [
            Match(chunk=chunk, score=cosine_similarity(query_vector, chunk.embedding))
            for chunk in self.chunks
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        limit = top_k or self.top_k
        return [
            match
            for match in matches[:limit]
            if match.score >= self.min_similarity
        ]

    def sources(self) -> list[str]:
        return [str(path.relative_to(self.directory)) for path in self.documents]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimension,
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._embed_with_retry(batch, config)
            vectors.extend([list(embedding.values) for embedding in response.embeddings])
        return vectors

    def _embed_with_retry(self, texts: list[str], config: types.EmbedContentConfig):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=texts,
                    config=config,
                )
            except Exception as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code not in {429, 500, 503} or attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise last_error  # pragma: no cover

    def _load_cache(self, digest: str) -> list[Chunk] | None:
        if not self.cache_path.is_file():
            return None
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if data.get("fingerprint") != digest:
                return None
            return [
                Chunk(
                    source=item["source"],
                    text=item["text"],
                    embedding=[float(value) for value in item["embedding"]],
                )
                for item in data["chunks"]
            ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save_cache(self, digest: str) -> None:
        payload = {
            "fingerprint": digest,
            "chunks": [
                {
                    "source": chunk.source,
                    "text": chunk.text,
                    "embedding": chunk.embedding,
                }
                for chunk in self.chunks
            ],
        }
        try:
            self.cache_path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
        except OSError:
            pass


def chunk_text(text: str, size: int) -> list[str]:
    """Split text into chunks of roughly `size` characters."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph, size))
        elif current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        chunks.append(current)
    return chunks


def _split_long_text(text: str, size: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and length + addition > size:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += addition

    if current:
        chunks.append(" ".join(current))
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = math.fsum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
