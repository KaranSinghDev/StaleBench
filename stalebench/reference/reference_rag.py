"""A reference RAG: sparse (TF-IDF) or dense retrieval + a pluggable LLM answerer.

`index()` builds the retriever once per call (not per query), then `answer()` retrieves top-k and
asks the LLM. Use it as a baseline, or copy it as a template for wrapping your own pipeline.

    rag = ReferenceRAG(llm=my_llm_callable, retriever="tfidf", top_k=3)
    # llm is any callable: (query, context) -> str
"""
from __future__ import annotations
import numpy as np
from ..system import RAGSystem
from ..corpus import Document


class ReferenceRAG(RAGSystem):
    def __init__(self, llm, retriever: str = "tfidf", top_k: int = 3, embedder=None,
                 recency_order: bool = False):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.embedder = embedder      # required if retriever == "dense" (e.g. SentenceTransformer)
        self.recency_order = recency_order  # if True, sort retrieved docs by doc_id suffix (":new" last)
        self._docs: list[Document] = []
        self._vec = None
        self._mat = None
        self._emb = None

    def index(self, documents: list[Document]) -> None:
        self._docs = list(documents)
        texts = [d.text for d in self._docs]
        if self.retriever == "tfidf":
            if texts:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._vec = TfidfVectorizer().fit(texts)
                self._mat = self._vec.transform(texts)
            else:
                self._vec = self._mat = None
        elif self.retriever == "dense":
            if self.embedder is None:
                raise ValueError("retriever='dense' requires an `embedder`")
            self._emb = self.embedder.encode(texts, normalize_embeddings=True) if texts else None
        else:
            raise ValueError(f"unknown retriever: {self.retriever!r}")

    def _retrieve(self, query: str) -> list[Document]:
        if not self._docs:
            return []
        if self.retriever == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(self._vec.transform([query]), self._mat)[0]
        else:
            q = self.embedder.encode([query], normalize_embeddings=True)[0]
            sims = self._emb @ q
        order = sims.argsort()[::-1][:self.top_k]
        top = [self._docs[i] for i in order]
        if self.recency_order:   # optional fix: place the newest version last in the prompt
            top.sort(key=lambda d: d.doc_id.endswith(":new"))
        return top

    def answer(self, query: str) -> str:
        context = " ".join(d.text for d in self._retrieve(query))
        return self.llm(query, context)
