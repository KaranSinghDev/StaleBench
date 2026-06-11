# StaleBench

A benchmark that measures answer freshness in RAG (Retrieval-Augmented Generation) systems.

Most freshness tools check the index. They ask: is the new document stored? StaleBench checks the answer. It asks: after a fact changes, how long until the system gives the new answer, and how often does it keep giving the old one?

## What it measures

- **Catch-up latency**: the number of steps from a fact changing until the answer becomes correct and stays correct.
- **Recovery rate**: the share of facts whose answer catches up. It is reported with a Wilson 95 percent confidence interval.

These are measured across refresh policies (never, batch, immediate) and across any system you connect.

The benchmark controls the documents, the clock, and the correct answers. So answers are scored by exact match, with no LLM acting as a judge. Results are reported over many trials, so they stay stable even though LLM outputs vary.

## Install

```bash
pip install -r requirements.txt
```

This installs numpy, scikit-learn, and openai. For the dense retriever, also install sentence-transformers.

## Quick start

Point it at any OpenAI-compatible endpoint, such as OpenAI, OpenRouter, vLLM, Ollama, or LM Studio:

```bash
python -m stalebench --model qwen2.5-3b-instruct --base-url http://localhost:1234/v1
```

Example output:

```
never      recovery=0.00 CI[0.0, 0.03]
batch      recovery=0.25 CI[0.19, 0.33]
immediate  recovery=0.58 CI[0.50, 0.66]
```

## Benchmark your own RAG system

Write two methods:

```python
from stalebench import RAGSystem, make_scenario, benchmark

class MyRAG(RAGSystem):
    def index(self, documents):   # build or refresh your retrieval index
        ...
    def answer(self, query):      # run your retrieve and generate, return a string
        ...

aggregate, _ = benchmark(MyRAG(), make_scenario(n_facts=24), trials=3)
```

Any system that can take a set of documents and answer a query can be measured. See `examples/custom_system.py`.

## Scoring

An `AnswerChecker` decides if an answer is correct. The default `TokenChecker` matches whole words, so a value like "Park" is not matched inside "Parkinson". You can write your own checker for other value types, such as numbers or dates.

## Findings

In experiments across model sizes, retrievers, and many random scenarios:

- About 40 to 60 percent of answers stay stale even with immediate re-indexing. A fresh index does not give a fresh answer.
- The cause is position. When both the old and new documents are retrieved, the model tends to follow document order, not recency.
- Placing the newest document last (the `--recency-order` option) reduces the problem. It helps most for models with the strongest position bias.

## Scope and limits

- It works for any RAG system whose documents you can change and refresh. A closed system where you cannot change the documents cannot be measured.
- The exact numbers depend on the scenario, as with any benchmark. StaleBench measures relative freshness and shows the size of the problem in a reliable way. Use your own data for conclusions about your own system.
- The included scenario uses controlled, synthetic facts with single-value questions.

## Validate the metric

```bash
python -m pytest -q
```

The test in `tests/test_ruler.py` checks that a fresh system scores higher than a stale one. If the metric cannot tell them apart, it is broken.

## License

MIT. See `LICENSE`.
