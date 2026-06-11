# Reference experiment results

Recovery rate = fraction of facts whose answer caught up to the new value within the horizon
(higher = fresher). All numbers below are from runs on local open models served via an
OpenAI-compatible endpoint; raw per-fact latencies are in `results/raw/`.

**Headline:** even with **immediate** re-indexing, **~40–60% of answers stay stale** — a fresh
index does not yield a fresh answer.

---

## 1. Recovery rate by model × refresh policy (24 facts × 6 trials, n=144/cell, Wilson 95% CI)

| Model | never | batch | immediate |
|-------|-------|-------|-----------|
| Llama-3.2-1B | 0.00 [0.00–0.03] | 0.04 [0.02–0.09] | 0.42 [0.34–0.50] |
| Qwen2.5-3B | 0.00 [0.00–0.03] | 0.25 [0.19–0.33] | 0.58 [0.50–0.66] |
| Qwen2.5-7B | 0.00 [0.00–0.03] | 0.25 [0.14–0.41] | 0.38 [0.30–0.46] |

- `never` = 0% everywhere (sanity).
- `immediate` > `batch` > `never` for every model (refreshing helps).
- Even the best case leaves a large stale fraction.

## 2. Sensitivity across random scenarios (4 seeds, immediate policy)

| Model | per-seed recovery | mean | range |
|-------|-------------------|------|-------|
| Llama-3.2-1B | 0.25, 0.38, 0.21, 0.25 | 0.27 | 0.21–0.38 |
| Qwen2.5-3B | 0.38, 0.58, 0.62, 0.50 | 0.52 | 0.38–0.62 |
| Qwen2.5-7B | 0.54, 0.62, 0.33, 0.33 | 0.46 | 0.33–0.62 |

**Note (honest):** model *size* does **not** reliably predict freshness — 3B vs 7B flips across
seeds. We therefore make no size-ordering claim. The substantial staleness itself is stable.

## 3. The recency-aware fix (newest document last), immediate policy, 4 seeds

| Model | naive (sim-order) | recency-aware | Δ |
|-------|-------------------|---------------|---|
| Qwen2.5-3B | 0.52 | 0.60 | +8pp (3/4 seeds) |
| Qwen2.5-7B | 0.46 | 0.67 | +21pp (all 4 seeds) |

The fix helps most for the model with the strongest positional bias (7B) — consistent with the
mechanism (LLMs follow document order, so placing the new document last makes them pick it).
It reduces, but does not eliminate, the residual staleness.

## 4. Retriever does not change the result (sparse vs dense, 4 seeds, immediate)

| Model | sparse (TF-IDF) | dense (bge-small) |
|-------|-----------------|-------------------|
| Qwen2.5-3B | 0.52 | 0.43 |
| Qwen2.5-7B | 0.46 | 0.46 |

At most a small, model-specific difference within seed variance — the staleness is LLM-side, not
a retriever artifact.

## 5. Robust to design knobs (Qwen2.5-3B, immediate, recovery rate)

- temperature: 0.0 → 0.48, 0.7 → 0.48, 1.0 → 0.46 (rock-stable)
- top-k: 2 → 0.33, 3 → 0.48, 5 → 0.42
- anti-flicker window: 1 → 0.56, 2 → 0.48, 3 → 0.40

Every setting leaves substantial staleness; the finding is not an artifact of any single knob.

---

## Scope / caveats
- Controlled synthetic scenarios with single-value questions; absolute numbers are
  scenario-dependent. StaleBench measures relative freshness and the staleness phenomenon robustly.
- Models are small/mid open models served locally; results may differ for frontier models.
