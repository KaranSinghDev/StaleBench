"""A general OpenAI-compatible LLM answerer.

Works with any OpenAI-compatible endpoint — OpenAI, OpenRouter, vLLM, Ollama, LM Studio, etc.
Configure via constructor args or env vars (STALEBENCH_BASE_URL, STALEBENCH_API_KEY).
It is a plain callable: `llm(query, context) -> answer`, which the reference RAG plugs in.
"""
from __future__ import annotations
import os
import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, seed: int | None = 42, max_tokens: int = 16,
                 prompt_template: str | None = None, timeout: float = 120,
                 no_think: bool = False, strip_think: bool = True):
        from openai import OpenAI
        base_url = base_url or os.environ.get("STALEBENCH_BASE_URL")
        api_key = api_key or os.environ.get("STALEBENCH_API_KEY", "not-needed")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.no_think = no_think        # append "/no_think" to disable reasoning (Qwen3+ convention)
        self.strip_think = strip_think  # remove any <think>...</think> block before scoring
        self.prompt_template = prompt_template or (
            "Use ONLY the context to answer. Reply with just the value, nothing else.\n"
            "Context: {context}\nQuestion: {query}\nAnswer:")
        self.calls = 0

    def __call__(self, query: str, context: str) -> str:
        self.calls += 1
        prompt = self.prompt_template.format(query=query, context=context)
        if self.no_think:
            prompt += " /no_think"
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(model=self.model, messages=messages,
                      max_tokens=self.max_tokens, temperature=self.temperature)
        try:
            resp = self.client.chat.completions.create(seed=self.seed, **kwargs)
        except TypeError:   # endpoint doesn't accept `seed`
            resp = self.client.chat.completions.create(**kwargs)
        out = resp.choices[0].message.content or ""
        if self.strip_think:
            out = _THINK.sub("", out)
        return out.strip()
