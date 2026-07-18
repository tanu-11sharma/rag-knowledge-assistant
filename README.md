# RAG Knowledge Assistant

A small, self-contained Retrieval-Augmented Generation (RAG) Q&A agent that answers questions over a bundled set of sample HR policy documents, with every answer citing the exact source chunk it came from.

## What it does

You ask a question ("How many days can I work remotely?") and the agent:

1. Retrieves the most relevant chunks from the document set using a hand-rolled TF-IDF + cosine-similarity index.
2. Assembles a grounded, extractive answer from the sentences in those chunks that best match the question.
3. Returns the answer along with citations pointing to the exact document and chunk each sentence came from.

No LLM API key is required to run the demo — retrieval and answering are both implemented from the Python standard library, so the whole thing runs offline and deterministically. A `answer_with_llm_stub` function in `app/rag.py` shows exactly where you'd plug in a real LLM call (OpenAI, Anthropic, etc.) if you wanted to swap the extractive answerer for a generative one.

## Why this is relevant

RAG is one of the most common patterns in production AI applications right now: ground a model's answers in your own documents instead of relying purely on parametric knowledge, and cite sources so answers are verifiable. This project demonstrates the full pipeline — ingestion, chunking, retrieval, grounded answer generation, and citation — in miniature, without hiding the mechanics behind a vector-database SDK or a hosted embeddings API.

## Project structure

```
rag-knowledge-assistant/
├── app/
│   ├── main.py       FastAPI app (health, docs-info, ask endpoints)
│   └── rag.py         TF-IDF index, chunking, extractive answering
├── data/docs/          Sample knowledge base (3 HR policy documents)
├── tests/test_rag.py    Unit + API tests
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

The API is then available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

## Test

```bash
pytest -v
```

## Example usage

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days can I work remotely?"}'
```

Example response:

```json
{
  "question": "How many days can I work remotely?",
  "answer": "Employees may work remotely up to three days per week with manager approval, recorded in the team calendar so coworkers know when someone is out of the office.",
  "citations": [
    {
      "doc_id": "remote_work",
      "chunk_id": "remote_work#0",
      "sentence": "Employees may work remotely up to three days per week with manager approval, recorded in the team calendar so coworkers know when someone is out of the office."
    }
  ]
}
```

Other endpoints:

- `GET /health` — liveness + how many documents are loaded
- `GET /docs-info` — list of document IDs and chunk count

## Notes

- All documents in `data/docs/` are synthetic sample HR policy text written for this demo — they don't describe any real company or real policy.
- This is a demo/reference implementation, not a production RAG system. There's no vector database, no persistence, and no auth — it re-indexes the sample corpus in memory on startup.
