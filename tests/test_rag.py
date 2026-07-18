from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.rag import TfidfIndex, extractive_answer, load_corpus, split_sentences, tokenize

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

client = TestClient(app)


def test_tokenize_strips_stopwords_and_lowercases():
    tokens = tokenize("The Employee's Guide to Onboarding and Training")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "employee's" in tokens
    assert "onboarding" in tokens


def test_split_sentences_basic():
    sentences = split_sentences("First sentence. Second sentence! Third one?")
    assert sentences == ["First sentence.", "Second sentence!", "Third one?"]


def test_index_loads_all_sample_documents():
    index = TfidfIndex()
    count = load_corpus(index, DOCS_DIR)
    assert count == 3
    assert len(index.chunks) > 0


def test_search_returns_relevant_chunk_for_expense_question():
    index = TfidfIndex()
    load_corpus(index, DOCS_DIR)
    results = index.search("How are meal expenses reimbursed while traveling?", top_k=3)
    assert results, "expected at least one retrieved chunk"
    assert results[0].chunk.doc_id == "expense_policy"


def test_extractive_answer_includes_citation():
    index = TfidfIndex()
    load_corpus(index, DOCS_DIR)
    retrieved = index.search("What is the daily meal reimbursement cap?", top_k=3)
    result = extractive_answer("What is the daily meal reimbursement cap?", retrieved)
    assert result["citations"], "expected at least one citation"
    assert result["citations"][0]["doc_id"] == "expense_policy"
    assert "expense_policy#" in result["citations"][0]["chunk_id"]


def test_no_match_returns_graceful_fallback():
    index = TfidfIndex()
    load_corpus(index, DOCS_DIR)
    retrieved = index.search("zzz completely unrelated gibberish query", top_k=3)
    result = extractive_answer("zzz completely unrelated gibberish query", retrieved)
    assert "couldn't find" in result["answer"] or result["citations"] == []


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents_loaded"] == 3


def test_docs_info_endpoint_lists_documents():
    resp = client.get("/docs-info")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["documents"]) == {"expense_policy", "onboarding", "remote_work"}


def test_ask_endpoint_returns_grounded_answer_with_citation():
    resp = client.post("/ask", json={"question": "How many days can I work remotely?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"], "expected citations in response"
    assert any(c["doc_id"] == "remote_work" for c in body["citations"])
