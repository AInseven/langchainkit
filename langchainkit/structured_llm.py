"""Structured output parsing functionality for LangKit."""

import time
from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing import Type, Union
from langfuse.langchain import CallbackHandler
from loguru import logger


def prompt_parsing(model: Type[BaseModel],
                   failed_model: BaseModel,
                   query: Union[str, list[str]],
                   llm: BaseChatModel,
                   langfuse_user_id: str = 'user_1',
                   langfuse_session_id: str = 'session_1',
                   max_concurrency: int = 1000) -> Union[BaseModel, list[BaseModel]]:
    """
    Forces LLM output to conform to a specified Pydantic model structure.
    
    This function wraps LLM calls with structured output parsing, ensuring responses
    follow the defined schema. It supports both single and batch processing with
    automatic retry logic for failed requests.
    
    :param model: Pydantic model class defining the expected output structure
    :param failed_model: Fallback instance returned after max retries (10 attempts)
    :param query: Single query string or list of queries to process
    :param llm: LangChain chat model instance for inference
    :param langfuse_user_id: User identifier for Langfuse observability tracking
    :param langfuse_session_id: Session identifier for Langfuse observability tracking  
    :param max_concurrency: Maximum concurrent requests for batch processing.If not provied,
    it will use the default value of llm.max_concurrency
    :return: Single BaseModel instance or list of BaseModel instances matching input queries

    Example:
    from langchainkit import prompt_parsing,LocalLLM
    from pydantic import BaseModel

    llm = LocalLLM.qwen3_14b_awq_think()

    class Response(BaseModel):
        answer: str
        confidence: float

    result = prompt_parsing(
        model=Response,
        failed_model=Response(answer="no_answer", confidence=0.0),
        query="What is the capital of France?",
        llm=llm
    )
    print(result.answer)  # "Paris"
    print(result.confidence)  # 1.0

    result = prompt_parsing(
        model=Response,
        failed_model=Response(answer="no_answer", confidence=0.0),
        query=["What is the capital of France?",
               "What is the capital of Germany?",
               "What is the capital of Italy?"],
        llm=llm
    )
    for each in result:
        print(each.answer)
        print(each.confidence)

    # Paris
    # 0.95
    # Berlin
    # 0.95
    # Rome
    # 1.0
    """
    handler = CallbackHandler()
    if hasattr(llm, 'max_concurrency'):
        max_concurrency=llm.max_concurrency
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