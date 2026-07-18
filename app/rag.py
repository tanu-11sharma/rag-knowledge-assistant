"""
Minimal, dependency-light RAG (Retrieval-Augmented Generation) core.

Design choices, on purpose:
- No external LLM API key required to run the demo. Retrieval uses a
  hand-rolled TF-IDF + cosine similarity index (pure Python, stdlib only).
- "Generation" is extractive: the answer is assembled from the most
  relevant sentences in the top-matching chunks, each tagged with a
  citation back to its source document. This keeps the whole pipeline
  runnable offline with zero API keys and fully deterministic for tests.
- Swapping the extractive answerer for a real LLM call (OpenAI, Anthropic,
  etc.) is a one-function change — see `answer_with_llm_stub` below.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

WORD_RE = re.compile(r"[a-zA-Z0-9']+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "for", "with", "as", "by",
    "at", "from", "that", "this", "it", "its", "into", "than", "then",
    "so", "such", "can", "could", "should", "would", "will", "shall",
    "do", "does", "did", "not", "no", "if", "we", "you", "your", "our",
    "they", "their", "he", "she", "his", "her", "which", "who", "whom",
    "what", "when", "where", "how", "why", "all", "any", "each", "more",
    "most", "other", "some", "these", "those", "i", "am", "also", "about",
}


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.replace("\n", " ").strip()
        if not para:
            continue
        for part in re.split(r"(?<=[.!?])\s+", para):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    term_freq: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class TfidfIndex:
    """A tiny inverted-index TF-IDF retriever. No third-party deps."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.doc_freq: dict[str, int] = {}
        self._built = False

    def add_document(self, doc_id: str, text: str, chunk_size: int = 60) -> None:
        sentences = split_sentences(text)
        buf: list[str] = []
        count = 0
        idx = 0

        def flush():
            nonlocal buf, count, idx
            if not buf:
                return
            chunk_text = " ".join(buf)
            chunk = Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{idx}", text=chunk_text)
            self.chunks.append(chunk)
            idx += 1
            buf = []
            count = 0

        for sent in sentences:
            words = sent.split()
            if count + len(words) > chunk_size and buf:
                flush()
            buf.append(sent)
            count += len(words)
        flush()
        self._built = False

    def build(self) -> None:
        self.doc_freq = {}
        for chunk in self.chunks:
            terms = tokenize(chunk.text)
            tf: dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            chunk.term_freq = tf
            for t in tf:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        self._built = True

    def _vector(self, term_freq: dict) -> dict[str, float]:
        n_docs = max(len(self.chunks), 1)
        vec = {}
        for term, freq in term_freq.items():
            idf = math.log((n_docs + 1) / (self.doc_freq.get(term, 0) + 1)) + 1
            vec[term] = (1 + math.log(freq)) * idf
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not self._built:
            self.build()
        q_terms: dict[str, int] = {}
        for t in tokenize(query):
            q_terms[t] = q_terms.get(t, 0) + 1
        q_vec = self._vector(q_terms)

        scored = []
        for chunk in self.chunks:
            c_vec = self._vector(chunk.term_freq)
            score = self._cosine(q_vec, c_vec)
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def load_corpus(index: TfidfIndex, docs_dir: Path) -> int:
    count = 0
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        index.add_document(doc_id=path.stem, text=text)
        count += 1
    index.build()
    return count


def extractive_answer(query: str, retrieved: list[RetrievedChunk], max_sentences: int = 3) -> dict:
    """
    Build a grounded answer by picking the sentences (from retrieved
    chunks) most similar to the query, each with a citation.
    """
    q_terms = set(tokenize(query))
    candidates = []
    for r in retrieved:
        for sent in split_sentences(r.chunk.text):
            s_terms = set(tokenize(sent))
            overlap = len(q_terms & s_terms)
            if overlap == 0:
                continue
            candidates.append((overlap, r.score, sent, r.chunk.doc_id, r.chunk.chunk_id))

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    top = candidates[:max_sentences]

    if not top:
        return {
            "answer": "I couldn't find anything in the knowledge base that answers this question.",
            "citations": [],
        }

    answer_text = " ".join(c[2] for c in top)
    citations = [
        {"doc_id": c[3], "chunk_id": c[4], "sentence": c[2]} for c in top
    ]
    return {"answer": answer_text, "citations": citations}


def answer_with_llm_stub(query: str, retrieved: list[RetrievedChunk]) -> str:
    """
    Placeholder showing where a real LLM call would go. Not used by the
    default pipeline so the demo runs with zero API keys.
    """
    context = "\n\n".join(r.chunk.text for r in retrieved)
    return f"[LLM call would go here with context of {len(context)} chars for query: {query!r}]"
