"""LLM provider management for LangKit."""

import os
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatModel


class LocalLLM:
    _qwen3_14b_awq_think = None
    _qwen3_14b_awq_no_think = None
    _qwen3_32b_think = None

    @classmethod
    def qwen3_14b_awq_think(cls)->BaseChatModel:
        if cls._qwen3_14b_awq_think is None:
            cls._qwen3_14b_awq_think = ChatDeepSeek(
                model="Qwen3-14B-AWQ",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
        return cls._qwen3_14b_awq_think

    @classmethod
    def qwen3_14b_awq_no_think(cls)->BaseChatModel:
        if cls._qwen3_14b_awq_no_think is None:
            cls._qwen3_14b_awq_no_think = ChatDeepSeek(
                model="Qwen3-14B-AWQ",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        return cls._qwen3_14b_awq_no_think

    @classmethod
    def qwen3_32b_think(cls)->BaseChatModel:
        if cls._qwen3_32b_think is None:
            cls._qwen3_32b_think = ChatDeepSeek(
                model="Qwen3-32B",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
        return cls._qwen3_32b_think



class ApiLLM:
    _qwen3_235b_think = None
    _qwen3_235b_no_think = None

    @classmethod
    def qwen3_235b_think(cls)->BaseChatModel:
        if cls._qwen3_235b_think is None:
            cls._qwen3_235b_think = ChatOpenAI(
                model="qwen3-235b-a22b-thinking-2507",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming=True,
                extra_body={"enable_thinking": True}
            )
        return cls._qwen3_235b_think

    @classmethod
    def qwen3_235b_no_think(cls)->BaseChatModel:
        if cls._qwen3_235b_no_think is None:
            cls._qwen3_235b_no_think = ChatOpenAI(
                model="qwen3-235b-a22b-instruct-2507",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming=True,
                extra_body={"enable_thinking": False}
            )
        return cls._qwen3_235b_no_think


class GeneralLLM:
    _deepseek_reasoner = None
    _deepseek_chat = None
    _gpt_4o = None
    _gpt_5_mini = None
    _gemini_2_5_pro = None
    _kimi_k2 = None
    _grok_4 = None

    @classmethod
    def deepseek_reasoner(cls)->BaseChatModel:
        if cls._deepseek_reasoner is None:
            cls._deepseek_reasoner = ChatDeepSeek(
                model="deepseek-reasoner",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                streaming=True,
                max_retries=5
            )
        return cls._deepseek_reasoner

    @classmethod
    def deepseek_chat(cls)->BaseChatModel:
        if cls._deepseek_chat is None:
            cls._deepseek_chat = ChatDeepSeek(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                streaming=True,
                max_retries=5
            )
        return cls._deepseek_chat

    @classmethod
    def kimi_k2(cls)->BaseChatModel:
        if cls._kimi_k2 is None:
            cls._kimi_k2 = ChatOpenAI(
                model="kimi-k2-0711-preview",
                api_key=os.getenv("MOONSHOT_API_KEY"),
                base_url="https://api.moonshot.cn/v1",
                streaming=True,
                max_retries=5
            )
        return cls._kimi_k2

    @classmethod
    def gpt_4o(cls)->BaseChatModel:
        if cls._gpt_4o is None:
            cls._gpt_4o = ChatOpenAI(
                model="openai/gpt-4o",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5
            )
        return cls._gpt_4o

    @classmethod
    def gpt_5_mini(cls)->BaseChatModel:
        if cls._gpt_5_mini is None:
            cls._gpt_5_mini = ChatOpenAI(
                model="openai/gpt-5-mini",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5
            )
        return cls._gpt_5_mini

    #google / gemini - 2.5 - pro
    @classmethod
    def gemini_2_5_pro(cls)->BaseChatModel:
        if cls._gemini_2_5_pro is None:
            cls._gemini_2_5_pro = ChatOpenAI(
                model="google/gemini-2.5-pro",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5
            )
        return cls._gemini_2_5_pro

    @classmethod
    def grok_4(cls)->BaseChatModel:
        if cls._grok_4 is None:
            cls._grok_4 = ChatOpenAI(
                model="x-ai/grok-4",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5
            )
        return cls._grok_4


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    # llm=GeneralLLM.gpt_4o()
    llm=ApiLLM.qwen3_235b_think()
    print(llm.invoke('hello'))
