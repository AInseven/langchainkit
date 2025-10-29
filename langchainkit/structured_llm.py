"""Structured output parsing functionality for LangKit."""

from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing import Type, Union, TypeVar, overload, List
from langfuse.langchain import CallbackHandler
from loguru import logger
from utils import batch_with_retry

M = TypeVar("M", bound=BaseModel)


@overload
def prompt_parsing(
        model: Type[M],
        failed_model: M,
        query: str,
        llm,
        use_langfuse: bool = ...,
        langfuse_user_id: str = ...,
        langfuse_session_id: Union[str, list[str]] = ...,
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
        langfuse_session_id: Union[str, list[str]] = ...,
        max_concurrency: int = ...
) -> List[M]: ...


def prompt_parsing(model: Type[M],
                   failed_model: M,
                   query: Union[str, list[str]],
                   llm: BaseChatModel,
                   use_langfuse: bool = True,
                   langfuse_user_id: str = 'user_1',
                   langfuse_session_id: Union[str, list[str]] = 'session_1',
                   max_concurrency: int = None) -> Union[M, list[M]]:
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
    langfuse_session_id : str or list of str, optional
        Session identifier for Langfuse observability tracking. Default is "session_1".
        If it is a str, then all query will use same session_id
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
    if hasattr(llm, 'max_concurrency') and max_concurrency is None:
        max_concurrency = llm.max_concurrency
    elif max_concurrency is None:
        max_concurrency = 10

    # Invoke configs
    if isinstance(langfuse_session_id,list):
        assert len(langfuse_session_id) == len(query), "langfuse_session_id must be list with same length as query"
        invoke_configs=[]
        for session_id in langfuse_session_id:
            invoke_configs.append(RunnableConfig(max_concurrency=max_concurrency,
                                    callbacks=[handler] if use_langfuse else [],
                                    metadata={
                                        "langfuse_user_id": langfuse_user_id,
                                        "langfuse_session_id": session_id,
                                        "langfuse_tags": ["langchain"]
                                    }))
    elif isinstance(langfuse_session_id,str) and isinstance(query,list):
        invoke_configs = [RunnableConfig(max_concurrency=max_concurrency,
                                    callbacks=[handler] if use_langfuse else [],
                                    metadata={
                                        "langfuse_user_id": langfuse_user_id,
                                        "langfuse_session_id": langfuse_session_id,
                                        "langfuse_tags": ["langchain"]
                                    }) for _ in query]
    else:
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

    # 如果query是多个请求list[str]，则使用batch_with_retry批量调用
    inputs = [{"query": q} for q in query]

    results = batch_with_retry(
        llm=chain,
        prompts=inputs,
        input_config=invoke_configs,
        max_retries=10,
        delay=2,
        failed_value=failed_model
    )

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
