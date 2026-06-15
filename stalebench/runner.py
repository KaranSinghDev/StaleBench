"""The benchmark loop: drive a refresh policy over logical time and measure catch-up latency.

For each tick we compute which document versions a given refresh policy makes visible, (re)index
the system with exactly that set, then query each not-yet-recovered fact. A fact's catch-up
latency is the first tick its answer is correct AND stays correct for `W` ticks (anti-flicker),
minus its change tick. `never`/`batch`/`immediate` are reference policies; you can pass your own.
"""
from __future__ import annotations
import math
from .corpus import Scenario
from .system import RAGSystem, AnswerChecker, TokenChecker

POLICIES = ("never", "batch", "immediate")


def effective_tick(policy: str, t: int, batch_T: int) -> int:
    """Which corpus version the index reflects at tick `t` under `policy`."""
    if policy == "never":
        return 0          # never ingests changes -> only the initial corpus is ever visible
    if policy == "immediate":
        return t          # streaming: changes are visible the moment they occur
    if policy == "batch":
        return (t // batch_T) * batch_T   # rebuilt every batch_T ticks
    raise ValueError(f"unknown policy: {policy!r}")


def run_policy(system: RAGSystem, scenario: Scenario, policy: str, *,
               checker: AnswerChecker = None, horizon: int = 14,
               batch_T: int = 4, anti_flicker: int = 2) -> dict[str, float]:
    """Return {fact_id: catch_up_latency} for one pass of one policy (inf = never caught up)."""
    checker = checker or TokenChecker()
    state = {f.id: {"stable": None, "latency": None} for f in scenario.facts}
    for t in range(0, horizon + 1):
        system.index(scenario.documents_at(effective_tick(policy, t, batch_T)))
        for f in scenario.facts:
            s = state[f.id]
            if t < f.change_tick or s["latency"] is not None:
                continue
            correct = checker.classify(system.answer(f.query()), f.old, f.new) == "NEW"
            if correct:
                if s["stable"] is None:
                    s["stable"] = t
                elif t - s["stable"] >= anti_flicker:
                    s["latency"] = s["stable"] - f.change_tick
            else:
                s["stable"] = None
    return {f.id: (state[f.id]["latency"] if state[f.id]["latency"] is not None else math.inf)
            for f in scenario.facts}


def competence(system: RAGSystem, scenario: Scenario, *, checker: AnswerChecker = None,
               trials: int = 1) -> dict:
    """Control condition: can the system answer the NEW value when only the new document is
    present (no old/new conflict)? If competence is low, a low recovery rate in the full
    benchmark reflects task failure, not staleness. Returns {n, correct, rate, rate_ci}."""
    from .metrics import wilson
    from .corpus import Document
    checker = checker or TokenChecker()
    docs = list(scenario.distractors) + [
        Document(f.statement(f.new), f"{f.id}:new") for f in scenario.facts]
    correct = total = 0
    for _ in range(trials):
        system.index(docs)
        for f in scenario.facts:
            total += 1
            if checker.classify(system.answer(f.query()), f.old, f.new) == "NEW":
                correct += 1
    rate = correct / total if total else 0.0
    return {"n": total, "correct": correct, "rate": round(rate, 3), "rate_ci": wilson(correct, total)}


def benchmark(system: RAGSystem, scenario: Scenario, *, checker: AnswerChecker = None,
              policies=POLICIES, trials: int = 1, horizon: int = 14, batch_T: int = 4,
              anti_flicker: int = 2):
    """Run all policies over `trials` repeats. Returns (aggregate_per_policy, raw_latencies).

    `aggregate_per_policy[policy]` = {n, recovered, rate, rate_ci, mean_catchup} (see metrics).
    Trials capture LLM non-determinism; report the rate with its confidence interval.
    """
    from .metrics import summarize
    checker = checker or TokenChecker()
    raw: dict[str, list[float]] = {}
    for policy in policies:
        lats: list[float] = []
        for _ in range(trials):
            result = run_policy(system, scenario, policy, checker=checker, horizon=horizon,
                                batch_T=batch_T, anti_flicker=anti_flicker)
            lats.extend(result.values())
        raw[policy] = lats
    return {p: summarize(v) for p, v in raw.items()}, raw
