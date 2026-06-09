"""LLM backends. The reference RAG takes any callable `(query, context) -> str` as its answerer."""
from .openai_compat import OpenAICompatLLM

__all__ = ["OpenAICompatLLM"]
