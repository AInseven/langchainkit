"""LLM provider management for LangKit."""

import os
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatModel
from pydantic import PrivateAttr


class CustomChatOpenAI(ChatOpenAI):
    """add max_concurrency"""
    _max_concurrency: int = PrivateAttr(default=4)

    @property
    def max_concurrency(self): return self._max_concurrency

    @max_concurrency.setter
    def max_concurrency(self, v): self._max_concurrency = v


class CustomChatDeepSeek(ChatDeepSeek):
    """add max_concurrency"""
    _max_concurrency: int = PrivateAttr(default=4)

    @property
    def max_concurrency(self): return self._max_concurrency

    @max_concurrency.setter
    def max_concurrency(self, v): self._max_concurrency = v


class LocalLLM:
    _qwen3_14b_awq_think = None
    _qwen3_14b_awq_no_think = None
    _qwen3_32b_think = None
    _qwen3_30b_a3b_think = None
    _qwen3_30b_a3b_instruct = None

    @classmethod
    def qwen3_14b_awq_think(cls) -> BaseChatModel:
        if cls._qwen3_14b_awq_think is None:
            cls._qwen3_14b_awq_think = CustomChatDeepSeek(
                model="Qwen3-14B-AWQ",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                timeout=300
            )
            cls._qwen3_14b_awq_think.max_concurrency = 100
        return cls._qwen3_14b_awq_think

    @classmethod
    def qwen3_14b_awq_no_think(cls) -> BaseChatModel:
        if cls._qwen3_14b_awq_no_think is None:
            cls._qwen3_14b_awq_no_think = CustomChatDeepSeek(
                model="Qwen3-14B-AWQ",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=300
            )
            cls._qwen3_14b_awq_no_think.max_concurrency = 300
        return cls._qwen3_14b_awq_no_think

    @classmethod
    def qwen3_32b_think(cls) -> BaseChatModel:
        if cls._qwen3_32b_think is None:
            cls._qwen3_32b_think = CustomChatDeepSeek(
                model="Qwen3-32B",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                timeout=300
            )
            cls._qwen3_32b_think.max_concurrency = 200
        return cls._qwen3_32b_think

    @classmethod
    def qwen3_30b_a3b_think(cls) -> BaseChatModel:
        if cls._qwen3_30b_a3b_think is None:
            cls._qwen3_30b_a3b_think = CustomChatDeepSeek(
                model="Qwen3-30B-A3B-Thinking-2507",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                timeout=200
            )
            cls._qwen3_30b_a3b_think.max_concurrency = 500
        return cls._qwen3_30b_a3b_think

    @classmethod
    def qwen3_30b_a3b_instruct(cls) -> BaseChatModel:
        if cls._qwen3_30b_a3b_instruct is None:
            cls._qwen3_30b_a3b_instruct = CustomChatOpenAI(
                model="Qwen3-30B-A3B-Thinking-2507",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                timeout=200
            )
            cls._qwen3_30b_a3b_instruct.max_concurrency = 500
        return cls._qwen3_30b_a3b_instruct


class ApiLLM:
    _qwen3_235b_think = None
    _qwen3_235b_no_think = None

    @classmethod
    def qwen3_235b_think(cls) -> BaseChatModel:
        if cls._qwen3_235b_think is None:
            cls._qwen3_235b_think = CustomChatOpenAI(
                model="qwen3-235b-a22b-thinking-2507",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming=True,
                timeout=300,
                extra_body={"enable_thinking": True}
            )
            cls._qwen3_235b_think.max_concurrency = 10
        return cls._qwen3_235b_think

    @classmethod
    def qwen3_235b_no_think(cls) -> BaseChatModel:
        if cls._qwen3_235b_no_think is None:
            cls._qwen3_235b_no_think = CustomChatOpenAI(
                model="qwen3-235b-a22b-instruct-2507",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming=True,
                timeout=300,
                extra_body={"enable_thinking": False}
            )
            cls._qwen3_235b_no_think.max_concurrency = 10
        return cls._qwen3_235b_no_think


class GeneralLLM:
    _deepseek_reasoner = None
    _deepseek_chat = None
    _gpt_4o = None
    _gpt_5_mini = None
    _gemini_2_5_pro = None
    _kimi_k2 = None
    _grok_4 = None
    _gemini_2_5_flash = None
    _qwen3_235b_think = None
    _openrouter_clients = {}
    _doubao_clients = {}

    @classmethod
    def deepseek_reasoner(cls) -> BaseChatModel:
        if cls._deepseek_reasoner is None:
            cls._deepseek_reasoner = CustomChatDeepSeek(
                model="deepseek-reasoner",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._deepseek_reasoner.max_concurrency = 300
        return cls._deepseek_reasoner

    @classmethod
    def deepseek_chat(cls) -> BaseChatModel:
        if cls._deepseek_chat is None:
            cls._deepseek_chat = CustomChatDeepSeek(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._deepseek_chat.max_concurrency = 300
        return cls._deepseek_chat

    @classmethod
    def kimi(cls, model_name: str="kimi-k2-turbo-preview") -> BaseChatModel:
        if cls._kimi_k2 is None:
            cls._kimi_k2 = CustomChatOpenAI(
                model=model_name,
                api_key=os.getenv("MOONSHOT_API_KEY"),
                base_url="https://api.moonshot.cn/v1",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._kimi_k2.max_concurrency = 100
        return cls._kimi_k2

    @classmethod
    def gpt_4o(cls) -> BaseChatModel:
        if cls._gpt_4o is None:
            cls._gpt_4o = CustomChatOpenAI(
                model="openai/gpt-4o",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._gpt_4o.max_concurrency = 100
        return cls._gpt_4o

    @classmethod
    def gpt_5_mini(cls) -> BaseChatModel:
        if cls._gpt_5_mini is None:
            cls._gpt_5_mini = CustomChatOpenAI(
                model="openai/gpt-5-mini",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._gpt_5_mini.max_concurrency = 100
        return cls._gpt_5_mini

    # google / gemini - 2.5 - pro
    @classmethod
    def gemini_2_5_pro(cls) -> BaseChatModel:
        if cls._gemini_2_5_pro is None:
            cls._gemini_2_5_pro = CustomChatOpenAI(
                model="google/gemini-2.5-pro",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._gemini_2_5_pro.max_concurrency = 100
        return cls._gemini_2_5_pro

    @classmethod
    def grok_4(cls) -> BaseChatModel:
        if cls._grok_4 is None:
            cls._grok_4 = CustomChatOpenAI(
                model="x-ai/grok-4",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._grok_4.max_concurrency = 100
        return cls._grok_4

    @classmethod
    def gemini_2_5_flash(cls) -> BaseChatModel:
        if cls._gemini_2_5_flash is None:
            cls._gemini_2_5_flash = CustomChatOpenAI(
                model="google/gemini-2.5-flash",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._gemini_2_5_flash.max_concurrency = 100
        return cls._gemini_2_5_flash

    @classmethod
    def qwen3_235b_think(cls) -> BaseChatModel:
        if cls._qwen3_235b_think is None:
            cls._qwen3_235b_think = CustomChatOpenAI(
                model="qwen/qwen3-235b-a22b-thinking-2507",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5,
                timeout=300
            )
            cls._qwen3_235b_think.max_concurrency = 100
        return cls._qwen3_235b_think

    @classmethod
    def openrouter(cls, model_name: str):
        if model_name not in cls._openrouter_clients:
            client = CustomChatOpenAI(
                model=model_name,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5,
                timeout=300,
            )
            client.max_concurrency = 100
            cls._openrouter_clients[model_name] = client

        return cls._openrouter_clients[model_name]

    @classmethod
    def doubao(cls, model_name: str = "doubao-seed-1-6-251015"):
        if model_name not in cls._doubao_clients:
            client = CustomChatOpenAI(
                model=model_name,
                api_key=os.getenv("ARK_API_KEY"),
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                streaming=True,
                max_retries=5,
                timeout=300,
            )
            client.max_concurrency = 100
            cls._doubao_clients[model_name] = client

        return cls._doubao_clients[model_name]