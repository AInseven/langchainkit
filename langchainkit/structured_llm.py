"""Structured output parsing functionality for LangKit."""

import time
from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing import Type, Union, TypeVar, overload, List
from langfuse.langchain import CallbackHandler
from loguru import logger
from tqdm import tqdm
from datetime import datetime

M = TypeVar("M", bound=BaseModel)


@overload
def prompt_parsing(
        model: Type[M],
        failed_model: M,
        query: str,
        llm,
        use_langfuse: bool = ...,
        langfuse_user_id: str = ...,
        langfuse_session_id: str = ...,
        max_concurrency: int = ...
) -> M: ...


@overload
def prompt_parsing(
        model: Type[M],
        failed_model: M,
        query: List[str],
        llm,
        use_langfuse: bool = ...,
        langfuse_user_id: str = ...,
        langfuse_session_id: str = ...,
        max_concurrency: int = ...
) -> List[M]: ...


def prompt_parsing(model: Type[M],
                   failed_model: M,
                   query: Union[str, list[str]],
                   llm: BaseChatModel,
                   use_langfuse: bool = True,
                   langfuse_user_id: str = 'user_1',
                   langfuse_session_id: str = 'session_1',
                   max_concurrency: int = 1000) -> Union[M, list[M]]:
    """
    Force LLM outputs to conform to a specified Pydantic model schema.

    This function wraps LLM calls with structured output parsing, ensuring that
    responses strictly follow the given Pydantic model definition. It supports
    both single-query and batch-query processing, and includes automatic retry
    logic (up to 10 attempts) for failed requests.

    Parameters
    ----------
    model : Type[BaseModel]
        Pydantic model class defining the expected output schema.
    failed_model : BaseModel
        Fallback instance returned if all retries are exhausted.
    query : str or list of str
        A single query string or a list of queries to process.
    llm : BaseChatModel
        LangChain chat model instance used for inference.
    use_langfuse: bool
        Whether to use Langfuse.
    langfuse_user_id : str, optional
        User identifier for Langfuse observability tracking. Default is "user_1".
    langfuse_session_id : str, optional
        Session identifier for Langfuse observability tracking. Default is "session_1".
    max_concurrency : int, optional
        Maximum number of concurrent requests for batch processing. If not
        provided, defaults to ``llm.max_concurrency``.

    Returns
    -------
    BaseModel or list of BaseModel
        A single model instance or a list of model instances, depending on the
        input query.

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
    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)

    handler = CallbackHandler()
    if hasattr(llm, 'max_concurrency'):
        max_concurrency = llm.max_concurrency
    invoke_configs = RunnableConfig(max_concurrency=max_concurrency,
                                    callbacks=[handler] if use_langfuse else [],
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
                # "Answer the user query. Wrap the output  in ```json and ``` tags\n{format_instructions}",
                "回答用户的问题. 把输出结果包裹在 ```json 和 ``` 标签里.\n{format_instructions}",
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
        new_to_retry_set = set()

        with tqdm(total=len(retry_inputs), desc=f"{model_name} Attempt {attempt}", leave=False) as pbar:
            for j, out in chain.batch_as_completed(
                    retry_inputs,
                    config=invoke_configs,
                    return_exceptions=True,
            ):
                i = to_retry[j]
                if isinstance(out, Exception):
                    logger.warning(f"[Attempt {attempt}] Failed on input {i}: {inputs[i]['query']}")
                    new_to_retry_set.add(i)
                else:
                    results[i] = out
                # add current time to progress bar
                now = datetime.now().strftime("%H:%M:%S")
                pbar.set_postfix_str(now)
                pbar.update(1)

        to_retry = sorted(new_to_retry_set)
        if to_retry:
            time.sleep(1.5)  # Optional: small delay between retries

    return results


def print_instructions(model: Type[BaseModel], query: str):
    """
    Print instructions for promp engineering

    Parameters
    ----------
    model : Type[BaseModel]
        Pydantic model class defining the expected output schema.
    query : str
        A single query string.
    """
    parser = PydanticOutputParser(pydantic_object=model)
    system_prompt = """Answer the user query. Wrap the output  in ```json and ``` tags\n{format_instructions}\n\n"""
    prefix = system_prompt.format(format_instructions=parser.get_format_instructions())

    print(prefix + query)
