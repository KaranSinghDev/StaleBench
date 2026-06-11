"""Ruler validation: the benchmark must rank a deliberately-fresh system above a stale one.

This is the credibility self-check. If the metric cannot separate an oracle that always uses the
newest indexed document from a system that ignores updates, the metric is broken. Runs offline
(mock systems, no LLM).
"""
from stalebench.corpus import make_scenario
from stalebench.system import RAGSystem
from stalebench.runner import benchmark


class AlwaysFresh(RAGSystem):
    """Oracle: answers the new value as soon as the new document is in the index."""

    def __init__(self, scenario):
        self.scenario = scenario
        self._texts = set()

    def index(self, documents):
        self._texts = {d.text for d in documents}

    def answer(self, query):
        for f in self.scenario.facts:
            if query == f.query():
                return f.new if f.statement(f.new) in self._texts else f.old
        return ""


class NeverFresh(RAGSystem):
    """Ignores updates: always answers the old value."""

    def __init__(self, scenario):
        self.scenario = scenario

    def index(self, documents):
        pass

    def answer(self, query):
        for f in self.scenario.facts:
            if query == f.query():
                return f.old
        return ""


def test_ruler_orders_correctly():
    sc = make_scenario(n_facts=8, seed=1)
    fresh, _ = benchmark(AlwaysFresh(sc), sc, trials=1, horizon=14)
    stale, _ = benchmark(NeverFresh(sc), sc, trials=1, horizon=14)

    # Under immediate indexing, the fresh oracle recovers; the stale system never does.
    assert fresh["immediate"]["rate"] == 1.0
    assert stale["immediate"]["rate"] == 0.0
    assert fresh["immediate"]["rate"] > stale["immediate"]["rate"]

    # Even a perfect system cannot recover if the index never ingests changes.
    assert fresh["never"]["rate"] == 0.0
