# Reference experiment results (v0.2)

Recovery rate = fraction of facts whose answer caught up to the new value within the horizon
(higher = fresher). All numbers are from local open models served through an OpenAI-compatible
endpoint, at temperature 0, 24 facts per scenario, with reasoning turned off for the models that
support it. The headline runs use 6 trials (n=144 per cell); the broad component sweep uses 3
trials (n=72 per cell). A 95% confidence interval is about ±0.08 (n=144) or ±0.12 (n=72).

**Headline:** even with **immediate** re-indexing, **about half of all answers stay stale**, across
ten models, three families, and two model generations. A fresh index does not give a fresh answer.

Models: Qwen2.5 (1.5B/3B/7B, Sept 2024), Llama-3.1-8B (Jul 2024), Llama-3.2 (1B/3B, Sept 2024),
Gemma-2 (2B/9B, 2024), Gemma-4 (E2B/E4B, Apr 2026). Qwen3.5 was excluded: its reasoning could not
be disabled in the serving stack.

---

## 1. Recovery rate by model and refresh policy (TF-IDF, n=144, Wilson 95% CI on immediate)

| Model | never | batch | immediate [95% CI] |
|-------|-------|-------|--------------------|
| Qwen2.5-1.5B | 0.00 | 0.50 | 0.50 [0.42, 0.58] |
| Qwen2.5-3B   | 0.00 | 0.67 | 0.62 [0.54, 0.70] |
| Qwen2.5-7B   | 0.00 | 0.58 | 0.51 [0.43, 0.59] |
| Llama-3.2-1B | 0.00 | 0.33 | 0.33 [0.26, 0.41] |
| Llama-3.2-3B | 0.00 | 0.54 | 0.50 [0.42, 0.58] |
| Llama-3.1-8B | 0.00 | 0.58 | 0.54 [0.46, 0.62] |
| Gemma-2-2B   | 0.00 | 0.50 | 0.50 [0.42, 0.58] |
| Gemma-2-9B   | 0.00 | 0.50 | 0.46 [0.38, 0.54] |
| Gemma-4-E2B  | 0.00 | 0.58 | 0.54 [0.46, 0.62] |
| Gemma-4-E4B  | 0.00 | 0.50 | 0.50 [0.42, 0.58] |

`never` = 0 everywhere (sanity). Immediate ≈ 0.50 (range 0.33–0.62): about half of answers stay stale.

## 2. The recency-order fix is model-dependent (immediate, TF-IDF, n=144)

| Model | naive | recency | change |
|-------|-------|---------|--------|
| Qwen2.5-1.5B | 0.50 | 0.04 | −46 pp (backfire) |
| Qwen2.5-3B   | 0.62 | 0.75 | +13 pp |
| Qwen2.5-7B   | 0.51 | 0.54 | +3 pp |
| Llama-3.2-1B | 0.33 | 0.54 | +21 pp |
| Llama-3.2-3B | 0.50 | 0.62 | +12 pp |
| Llama-3.1-8B | 0.54 | 1.00 | +46 pp (solved) |
| Gemma-2-2B   | 0.50 | 0.67 | +17 pp |
| Gemma-2-9B   | 0.46 | 0.92 | +46 pp |
| Gemma-4-E2B  | 0.54 | 0.79 | +25 pp |
| Gemma-4-E4B  | 0.50 | 0.33 | −17 pp (backfire) |

The fix helps most models and fully removes the staleness for the most capable ones, but it
backfires on some that anchor on the first document. Mechanism: a diagnostic on Qwen2.5-1.5B shows
that with the newest document last it answers with the OLD value on 6 of 6 changed facts (primacy
bias). The split matches the U-shaped (primacy vs recency) position bias known for LLMs.

## 3. Retriever and embedder invariance, and the competence control (immediate, naive, n=72)

| Model | TF-IDF | BM25 | dense | competence |
|-------|--------|------|-------|------------|
| Qwen2.5-1.5B | 0.50 | 0.62 | 0.46 | 1.00 |
| Qwen2.5-3B   | 0.62 | 0.62 | 0.42 | 1.00 |
| Qwen2.5-7B   | 0.51 | 0.42 | 0.50 | 1.00 |
| Llama-3.2-1B | 0.33 | 0.46 | 0.33 | 0.62 |
| Llama-3.2-3B | 0.54 | 0.50 | 0.46 | 1.00 |
| Llama-3.1-8B | 0.54 | 0.46 | 0.58 | 1.00 |
| Gemma-2-2B   | 0.50 | 0.62 | 0.54 | 1.00 |
| Gemma-2-9B   | 0.46 | 0.46 | 0.62 | 1.00 |
| Gemma-4-E2B  | 0.54 | 0.62 | 0.42 | 1.00 |
| Gemma-4-E4B  | 0.50 | 0.54 | 0.33 | 1.00 |

The retriever does not change the result (differences are within noise), so the staleness is on the
model side. On the three mid-size models, four embedders (all-MiniLM, bge-small, bge-large, nomic)
all fall between 0.46 and 0.62, and adding a cross-encoder reranker does not help (about 0.42).
**Competence control:** nine of ten models answer correctly when only the new document is present,
so their staleness is genuine. The exception is Llama-3.2-1B (0.62), whose low recovery is partly
task weakness, not only staleness.

---

## Scope and caveats
- Controlled synthetic facts with single-value questions; absolute numbers are scenario-dependent.
  StaleBench measures the size of the problem and relative freshness, not fixed constants.
- Models are small and mid-size open models served locally at 4-bit. Larger, closed, or reasoning
  models may behave differently.
- The two 2026 Gemma-4 models are multimodal, used in text-only mode, so the generation comparison
  mixes a change in model generation with a change in model type.
