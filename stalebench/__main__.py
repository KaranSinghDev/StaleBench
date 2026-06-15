"""Command-line entry point: benchmark the reference RAG against an OpenAI-compatible LLM.

    python -m stalebench --model qwen2.5-3b-instruct --base-url http://localhost:1234/v1

Point --base-url at any OpenAI-compatible server (OpenAI, OpenRouter, vLLM, Ollama, LM Studio).
Sweep components with --retriever {tfidf,bm25,dense}, --embedder, --reranker, --recency-order.
Use --no-think for reasoning models and --competence-check for the control condition.
"""
from __future__ import annotations
import argparse
from .corpus import make_scenario
from .reference import ReferenceRAG
from .backends import OpenAICompatLLM
from .runner import benchmark, competence


def main():
    p = argparse.ArgumentParser(description="StaleBench — RAG answer-freshness benchmark")
    p.add_argument("--model", required=True, help="model id on the OpenAI-compatible endpoint")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (or env STALEBENCH_BASE_URL)")
    p.add_argument("--api-key", default=None, help="API key (or env STALEBENCH_API_KEY; default 'not-needed')")
    p.add_argument("--retriever", default="tfidf", choices=["tfidf", "bm25", "dense"])
    p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5", help="dense retriever model")
    p.add_argument("--reranker", default=None, help="optional CrossEncoder model name (re-ranks the pool)")
    p.add_argument("--recency-order", action="store_true", help="place the newest doc last (the fix)")
    p.add_argument("--no-think", action="store_true", help="disable reasoning for thinking models (Qwen3+)")
    p.add_argument("--max-tokens", type=int, default=16, help="raise for reasoning models")
    p.add_argument("--device", default="cpu",
                   help="device for embedder/reranker; cpu keeps the GPU free for the LLM server")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="needed by some embedders (e.g. nomic-embed-text)")
    p.add_argument("--competence-check", action="store_true",
                   help="control: answer with ONLY the new doc present (isolates staleness from task failure)")
    p.add_argument("--n-facts", type=int, default=24)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    embedder = None
    if args.retriever == "dense":
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.embedder, device=args.device,
                                       trust_remote_code=args.trust_remote_code)
    reranker = None
    if args.reranker:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker, device=args.device)

    llm = OpenAICompatLLM(model=args.model, base_url=args.base_url, api_key=args.api_key,
                          temperature=args.temperature, max_tokens=args.max_tokens,
                          no_think=args.no_think)
    rag = ReferenceRAG(llm=llm, retriever=args.retriever, top_k=args.top_k,
                       embedder=embedder, recency_order=args.recency_order, reranker=reranker)
    scenario = make_scenario(n_facts=args.n_facts, seed=args.seed)

    label = (f"model={args.model} retriever={args.retriever}"
             + (f"+{args.embedder}" if args.retriever == "dense" else "")
             + (f" rerank={args.reranker}" if args.reranker else "")
             + f" recency={args.recency_order} no_think={args.no_think}")

    if args.competence_check:
        c = competence(rag, scenario, trials=args.trials)
        print(f"StaleBench competence | {label}")
        print(f"  competence rate={c['rate']:.2f} CI{c['rate_ci']} (n={c['n']})")
        print("\n(competence = fraction correct when ONLY the new doc is present; "
              "low means task failure, not staleness)")
        return

    print(f"StaleBench | {label} facts={args.n_facts} trials={args.trials}\n")
    aggregate, _ = benchmark(rag, scenario, trials=args.trials)
    for policy, s in aggregate.items():
        print(f"  {policy:10s} recovery={s['rate']:.2f} CI{s['rate_ci']} "
              f"mean_catchup={s['mean_catchup']} (n={s['n']})")
    print("\n(recovery = fraction of facts whose answer caught up; higher = fresher)")


if __name__ == "__main__":
    main()
