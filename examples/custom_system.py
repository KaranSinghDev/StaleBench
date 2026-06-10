"""Example: benchmark YOUR OWN RAG system by implementing the two-method adapter.

Run:  python examples/custom_system.py
This uses a trivial in-memory system so it runs with no LLM or API key.
"""
from stalebench import RAGSystem, make_scenario, benchmark


class MyRAG(RAGSystem):
    """Replace the bodies with calls into your real retriever + LLM."""

    def __init__(self):
        self._docs = []

    def index(self, documents):
        # your (re)indexing — vector store upsert, etc.
        self._docs = [d.text for d in documents]

    def answer(self, query):
        # your retrieve + generate. (Toy: return the last matching statement found.)
        hits = [t for t in self._docs if query.split("of")[-1].strip(" ?") in t]
        return hits[-1] if hits else ""


if __name__ == "__main__":
    scenario = make_scenario(n_facts=8, seed=0)
    aggregate, _ = benchmark(MyRAG(), scenario, trials=1)
    for policy, stats in aggregate.items():
        print(f"{policy:10s} recovery={stats['rate']:.2f} CI{stats['rate_ci']} "
              f"mean_catchup={stats['mean_catchup']}")
