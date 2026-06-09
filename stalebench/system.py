"""Adapter interfaces.

`RAGSystem` lets you plug ANY retrieval-augmented pipeline into the benchmark — the benchmark
only calls `index(documents)` then `answer(query)`, so it stays agnostic to your retriever,
vector store, LLM, or architecture (naive / graph / agentic).

`AnswerChecker` makes correctness scoring pluggable. The default `TokenChecker` does whole-token
phrase matching, so a value like "Park" is NOT falsely matched inside "Parkinson" (a real failure
mode of naive substring checks). Supply your own for domain-specific values (numbers, dates, ...).
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from .corpus import Document


class RAGSystem(ABC):
    """Wrap any RAG pipeline. `index` (re)builds the retrievable corpus; `answer` returns a string."""

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        ...

    @abstractmethod
    def answer(self, query: str) -> str:
        ...


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _contains_phrase(haystack: list[str], needle: list[str]) -> bool:
    """True iff `needle` appears as a contiguous run of WHOLE tokens in `haystack`."""
    n = len(needle)
    if n == 0:
        return False
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


class AnswerChecker(ABC):
    @abstractmethod
    def classify(self, answer: str, old: str, new: str) -> str:
        """Return one of 'NEW' | 'OLD' | 'BOTH' | 'OTHER'."""
        ...


class TokenChecker(AnswerChecker):
    """Whole-token phrase containment. Robust to substring false positives and word order."""

    def classify(self, answer: str, old: str, new: str) -> str:
        toks = _tokens(answer)
        has_old = _contains_phrase(toks, _tokens(old))
        has_new = _contains_phrase(toks, _tokens(new))
        if has_new and not has_old:
            return "NEW"
        if has_old and not has_new:
            return "OLD"
        if has_old and has_new:
            return "BOTH"
        return "OTHER"
