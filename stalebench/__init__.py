"""StaleBench — a benchmark for answer-freshness in Retrieval-Augmented Generation.

Measures how long after a fact changes a RAG system's *answer* becomes (and stays) correct
(catch-up latency) and the stale-answer rate — at the answer level, not the index level.
"""
from .corpus import Fact, Document, Scenario, make_scenario
from .system import RAGSystem, AnswerChecker, TokenChecker
from .runner import benchmark, run_policy, competence, POLICIES
from .metrics import summarize, wilson

__all__ = [
    "Fact", "Document", "Scenario", "make_scenario",
    "RAGSystem", "AnswerChecker", "TokenChecker",
    "benchmark", "run_policy", "competence", "POLICIES", "summarize", "wilson",
]
__version__ = "0.2.0"
