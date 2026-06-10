"""Command-line entry point: benchmark the reference RAG against an OpenAI-compatible LLM.

    python -m stalebench --model qwen2.5-3b-instruct --base-url http://localhost:1234/v1

Point --base-url at any OpenAI-compatible server (OpenAI, OpenRouter, vLLM, Ollama, LM Studio).
"""
from __future__ import annotations
import argparse
from .corpus import make_scenario
from .reference import ReferenceRAG
from .backends import OpenAICompatLLM
from .runner import benchmark


def main():
    p = argparse.ArgumentParser(description="StaleBench — RAG answer-freshness benchmark")
    p.add_argument("--model", required=True, help="model id on the OpenAI-compatible endpoint")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (or env STALEBENCH_BASE_URL)")
    p.add_argument("--api-key", default=None, help="API key (or env STALEBENCH_API_KEY; default 'not-needed')")
    p.add_argument("--retriever", default="tfidf", choices=["tfidf", "dense"])
    p.add_argument("--recency-order", action="store_true", help="place the newest doc last (the fix)")
    p.add_argument("--n-facts", type=int, default=24)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    embedder = None
    if args.retriever == "dense":
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    llm = OpenAICompatLLM(model=args.model, base_url=args.base_url, api_key=args.api_key,
                          temperature=args.temperature)
    rag = ReferenceRAG(llm=llm, retriever=args.retriever, top_k=args.top_k,
                       embedder=embedder, recency_order=args.recency_order)
    scenario = make_scenario(n_facts=args.n_facts, seed=args.seed)

    print(f"StaleBench | model={args.model} retriever={args.retriever} "
          f"recency_order={args.recency_order} facts={args.n_facts} trials={args.trials}\n")
    aggregate, _ = benchmark(rag, scenario, trials=args.trials)
    for policy, s in aggregate.items():
        print(f"  {policy:10s} recovery={s['rate']:.2f} CI{s['rate_ci']} "
              f"mean_catchup={s['mean_catchup']} (n={s['n']})")
    print("\n(recovery = fraction of facts whose answer caught up; higher = fresher)")


if __name__ == "__main__":
    main()
