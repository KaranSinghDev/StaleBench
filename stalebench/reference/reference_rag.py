"""A reference RAG: sparse (TF-IDF / BM25) or dense retrieval, optional reranking, + a pluggable LLM.

`index()` builds the retriever once per call (not per query), then `answer()` retrieves top-k and
asks the LLM. Use it as a baseline, or copy it as a template for wrapping your own pipeline.

    rag = ReferenceRAG(llm=my_llm_callable, retriever="tfidf", top_k=3)
    # llm is any callable: (query, context) -> str

`retriever` is one of "tfidf", "bm25", "dense" (dense needs an `embedder`). An optional `reranker`
(any object with `.predict([(query, text), ...]) -> scores`, e.g. a sentence-transformers
CrossEncoder) re-scores a larger candidate pool down to top_k.
"""
from __future__ import annotations
import re
import numpy as np
from ..system import RAGSystem
from ..corpus import Document

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class ReferenceRAG(RAGSystem):
    def __init__(self, llm, retriever: str = "tfidf", top_k: int = 3, embedder=None,
                 recency_order: bool = False, reranker=None, rerank_pool: int = 10):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.embedder = embedder      # required if retriever == "dense" (e.g. SentenceTransformer)
        self.recency_order = recency_order  # if True, sort retrieved docs by doc_id suffix (":new" last)
        self.reranker = reranker      # optional CrossEncoder-like; re-scores the candidate pool
        self.rerank_pool = rerank_pool
        self._docs: list[Document] = []
        self._vec = None
        self._mat = None
        self._emb = None
        self._bm25 = None

    def index(self, documents: list[Document]) -> None:
        self._docs = list(documents)
        texts = [d.text for d in self._docs]
        self._vec = self._mat = self._emb = self._bm25 = None
        if not texts:
            return
        if self.retriever == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer().fit(texts)
            self._mat = self._vec.transform(texts)
        elif self.retriever == "bm25":
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([_tokenize(t) for t in texts])
        elif self.retriever == "dense":
            if self.embedder is None:
                raise ValueError("retriever='dense' requires an `embedder`")
            self._emb = self.embedder.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError(f"unknown retriever: {self.retriever!r}")

    def _scores(self, query: str) -> np.ndarray:
        if self.retriever == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(self._vec.transform([query]), self._mat)[0]
        if self.retriever == "bm25":
            return np.asarray(self._bm25.get_scores(_tokenize(query)))
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        return self._emb @ q

    def _retrieve(self, query: str) -> list[Document]:
        if not self._docs:
            return []
        sims = self._scores(query)
        pool = max(self.top_k, self.rerank_pool) if self.reranker else self.top_k
        order = sims.argsort()[::-1][:pool]
        top = [self._docs[i] for i in order]
        if self.reranker is not None:            # re-score the pool, keep the best top_k
            rs = np.asarray(self.reranker.predict([(query, d.text) for d in top]))
            top = [top[i] for i in rs.argsort()[::-1][:self.top_k]]
        if self.recency_order:   # optional fix: place the newest version last in the prompt
            top.sort(key=lambda d: d.doc_id.endswith(":new"))
        return top

    def answer(self, query: str) -> str:
        context = " ".join(d.text for d in self._retrieve(query))
        return self.llm(query, context)
