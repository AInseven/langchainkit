"""Structured output parsing functionality for LangKit."""

import time
from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing import Type, Union
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
handler = CallbackHandler()


def prompt_parsing(model: Type[BaseModel],
                   failed_model: BaseModel,
                   query: Union[str, list[str]],
                   llm: BaseChatModel,
                   langfuse_user_id: str = 'user_1',
                   langfuse_session_id: str = 'session_1',
                   max_concurrency: int = 1000) -> Union[BaseModel, list[BaseModel]]:
    """
    让LLM按照query的要求， 返回一个pydantic model
    :param model: pydantic model
    :param failed_model:多次请求失败，返回一个instance of BaseModel
    :param query:
    :param llm:
    :param langfuse_user_id:
    :param langfuse_session_id:
    :param max_concurrency:
    :return:
    """
    invoke_configs = RunnableConfig(max_concurrency=max_concurrency,
                                    callbacks=[handler],
                                    metadata={
                                        "langfuse_user_id": langfuse_user_id,
                                        "langfuse_session_id": langfuse_session_id,
                                        "langfuse_tags": ["langchain"]
                                    })
    parser = PydanticOutputParser(pydantic_object=model)

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user query. Wrap the output  in ```json and ``` tags\n{format_instructions}",
            ),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    # 如果query是单个请求str，则直接调用
    if isinstance(query, str):
        return chain.invoke({"query": query}, config=invoke_configs)

    # 如果query是多个请求list[str]，则批量调用
    inputs = [{"query": q} for q in query]
    results = [failed_model] * len(inputs)
    max_retries = 10

    # chain.batch对出错的request会return_exceptions，对报错的request进行重试
    to_retry = list(range(len(inputs)))

    for attempt in range(1, max_retries + 1):
        if not to_retry:
            break

        retry_inputs = [inputs[i] for i in to_retry]

        batch_out = chain.batch(retry_inputs, config=invoke_configs, return_exceptions=True)

        new_to_retry = []
        for i, out in zip(to_retry, batch_out):
            if isinstance(out, Exception):
                logger.warning(f"[Attempt {attempt}] Failed on input {i}: {inputs[i]['query']}")
                # 失败的加入new_to_retry，下次一起batch retry
                new_to_retry.append(i)
            else:
                results[i] = out

        to_retry = new_to_retry
        if to_retry:
            time.sleep(1.5)  # Optional: small delay between retries

    return results