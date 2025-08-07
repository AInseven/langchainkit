"""LLM provider management for LangKit."""

import os
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatModel
from dotenv import load_dotenv

load_dotenv()

class LocalLLM:
    _qwen3_14b_awq_think = None
    _qwen3_14b_awq_no_think = None
    _qwen3_32b_think = None

    @property
    def qwen3_14b_awq_think(self)->BaseChatModel:
        if self._qwen3_14b_awq_think is None:
            self._qwen3_14b_awq_think = ChatDeepSeek(
                model="Qwen3-14B-AWQ",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
        return self._qwen3_14b_awq_think

    @property
    def qwen3_14b_awq_no_think(self)->BaseChatModel:
        if self._qwen3_14b_awq_no_think is None:
            self._qwen3_14b_awq_no_think = ChatDeepSeek(
                model="Qwen3-14B-AWQ",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        return self._qwen3_14b_awq_no_think

    @property
    def qwen3_32b_think(self)->BaseChatModel:
        if self._qwen3_32b_think is None:
            self._qwen3_32b_think = ChatDeepSeek(
                model="Qwen3-32B",
                api_key=os.getenv("LOCAL_VLLM_API_KEY"),
                api_base=os.getenv("LOCAL_VLLM_BASE_URL"),
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
        return self._qwen3_32b_think



class ApiLLM:
    _qwen3_235b_think = None
    _qwen3_235b_no_think = None

    @property
    def qwen3_235b_think(self)->BaseChatModel:
        if self._qwen3_235b_think is None:
            self._qwen3_235b_think = ChatDeepSeek(
                model="qwen3-235b-a22b",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
        return self._qwen3_235b_think

    @property
    def qwen3_235b_no_think(self)->BaseChatModel:
        if self._qwen3_235b_no_think is None:
            self._qwen3_235b_no_think = ChatDeepSeek(
                model="qwen3-235b-a22b",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                streaming=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        return self._qwen3_235b_no_think


class GeneralLLM:
    _deepseek_reasoner = None
    _deepseek_chat = None
    _gpt_4o = None
    @property
    def deepseek_reasoner(self)->BaseChatModel:
        if self._deepseek_reasoner is None:
            self._deepseek_reasoner = ChatDeepSeek(
                model="deepseek-reasoner",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                streaming=True,
                max_retries=5
            )
        return self._deepseek_reasoner

    @property
    def deepseek_chat(self)->BaseChatModel:
        if self._deepseek_chat is None:
            self._deepseek_chat = ChatDeepSeek(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                streaming=True,
                max_retries=5
            )
        return self._deepseek_chat

    @property
    def gpt_4o(self)->BaseChatModel:
        if self._gpt_4o is None:
            self._gpt_4o = ChatOpenAI(
                model="openai/gpt-4o",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                streaming=True,
                max_retries=5
            )
        return self._gpt_4o



if __name__ == '__main__':
    llm=GeneralLLM().gpt_4o
    print(llm.invoke('hello'))
