"""
LangChainKit - A Python toolkit for working with Large Language Models (LLMs) using LangChain.
"""

from .local_vllm import LocalLLM, ApiLLM, GeneralLLM
from .structured_llm import prompt_parsing
from .utils import batch_with_retry

__version__ = "0.1.0"
__all__ = ["LocalLLM", "ApiLLM", "GeneralLLM", "prompt_parsing", "batch_with_retry"]