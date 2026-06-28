---
title: 'StaleBench: A Benchmark for Answer Freshness in Retrieval-Augmented Generation'
tags:
  - Python
  - retrieval-augmented generation
  - large language models
  - evaluation
  - benchmark
authors:
  - name: Karan Singh
    orcid: 0009-0000-0920-2379
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 15 June 2026
bibliography: paper.bib
---

# Summary

Retrieval-augmented generation (RAG) systems answer questions by retrieving documents
and letting a language model read them [@lewis2020rag]. A common assumption is that such
systems stay current: when a fact changes, a new document is added and the system returns
the new answer. In practice, a system can store the new document in its index in under a
second yet keep returning the old answer, because storing a document and using it are not
the same step. Index-level freshness checks miss this, since they verify whether the new
document is stored, not whether the answer changed.

StaleBench measures freshness at the answer. After a fact changes, it measures how long a
system keeps returning the old value (catch-up latency) and how often it does so
(recovery rate), across three refresh policies (never, batch, immediate). Because the
benchmark controls the documents, the clock, and the ground-truth answers, responses are
scored by exact match, with no language model acting as a judge, and results are reported
with Wilson confidence intervals. StaleBench is a black-box tool: any system that can take
a set of documents and answer a query, with any retriever or model, can be measured.

# Statement of need

Existing RAG evaluations focus on answer quality or on keeping the test set fresh to avoid
data leakage [@ouyang2025hoh; @chernogorskii2025dragon], and prior work shows that language
models do not always follow retrieved text and are sensitive to document position
[@liu2024lost]. However, no open tool measures, at the answer level, *how long* a system
stays stale after a fact changes, across refresh policies, in a way a practitioner can run
on their own system. StaleBench fills this gap.

Using StaleBench across ten open models from three families, about half of all answers
remained stale even under immediate re-indexing, and this held across sparse, dense, and
reranked retrieval, indicating a model-side rather than a retrieval-side problem. The cause
is document position: a simple change, placing the newest document last, fully removed the
staleness for the most capable models but backfired on others. Because the effect is
model-dependent, practitioners cannot assume a fix works; they need to measure it.
StaleBench provides that measurement, with a small two-method interface, command-line
component sweeps, and a competence control that distinguishes genuine staleness from task
failure.

# Acknowledgements

The author thanks the open-source machine-learning community for the models and libraries
that made this evaluation possible.

# References
