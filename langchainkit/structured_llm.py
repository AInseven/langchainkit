"""Structured output parsing functionality for LangKit."""

import base64
from pathlib import Path
from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from typing import Type, Union, TypeVar, overload, List, Optional
from langfuse.langchain import CallbackHandler
from loguru import logger
from langchainkit.utils import batch_with_retry, abatch_with_retry

M = TypeVar("M", bound=BaseModel)


class MultimodalInput(BaseModel):
    """Input schema for multimodal prompts with text, images, and videos.

    Attributes:
        text: The text content of the prompt
        imgs: List of image URLs
        videos: List of video URLs
        img_paths: List of local image file paths (will be converted to base64)
        video_paths: List of local video file paths (will be converted to base64)
    """
    text: str
    imgs: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    img_paths: Optional[List[str]] = None
    video_paths: Optional[List[str]] = None


def _encode_file_to_base64(file_path: str) -> str:
    """Convert local file to base64 data URL.

    Parameters
    ----------
    file_path : str
        Path to the local file

    Returns
    -------
    str
        Base64-encoded data URL with appropriate MIME type
    """
    path = Path(file_path)
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')

    # Determine MIME type based on extension
    ext = path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif',
        '.webp': 'image/webp', '.bmp': 'image/bmp',
        '.mp4': 'video/mp4', '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime', '.webm': 'video/webm'
    }
    mime = mime_types.get(ext, 'application/octet-stream')

    return f"data:{mime};base64,{data}"


def _multimodal_to_content(input_data: Union[str, MultimodalInput]) -> List[dict]:
    """Convert string or MultimodalInput to LangChain message content format.

    Parameters
    ----------
    input_data : str or MultimodalInput
        Input to convert

    Returns
    -------
    list of dict
        Content blocks in LangChain format: [{"type": "text", "text": "..."}, ...]
    """
    if isinstance(input_data, str):
        return [{"type": "text", "text": input_data}]

    content = [{"type": "text", "text": input_data.text}]

    # Handle URL-based images
    if input_data.imgs:
        for img in input_data.imgs:
            content.append({"type": "image_url", "image_url": {"url": img}})

    # Handle URL-based videos
    if input_data.videos:
        for video in input_data.videos:
            content.append({"type": "video_url", "video_url": {"url": video}, "fps": 2})

    # Handle path-based images
    if input_data.img_paths:
        for img_path in input_data.img_paths:
            base64_url = _encode_file_to_base64(img_path)
            content.append({"type": "image_url", "image_url": {"url": base64_url}})

    # Handle path-based videos
    if input_data.video_paths:
        for video_path in input_data.video_paths:
            base64_url = _encode_file_to_base64(video_path)
            content.append({"type": "video_url", "video_url": {"url": base64_url}, "fps": 2})

    return content


def _prepare_configs(
        query: Union[str, MultimodalInput, List[Union[str, MultimodalInput]]],
        llm: BaseChatModel,
        use_langfuse: bool,
        langfuse_user_id: str,
        langfuse_session_id: Union[str, list[str]],
        langfuse_tag: str,
        run_name: str,
        max_concurrency: int
):
    """
    Internal helper to prepare configs for prompt_parsing.

    Returns: (invoke_configs, model_name)
    """
    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)

    handler = CallbackHandler()
    if hasattr(llm, 'max_concurrency') and max_concurrency is None:
        max_concurrency = llm.max_concurrency
    elif max_concurrency is None:
        max_concurrency = 10

    # Invoke configs
    if isinstance(langfuse_session_id, list):
        assert len(langfuse_session_id) == len(query), "langfuse_session_id must be list with same length as query"
        invoke_configs = []
        for session_id in langfuse_session_id:
            invoke_configs.append(RunnableConfig(max_concurrency=max_concurrency,
                                                  callbacks=[handler] if use_langfuse else [],
                                                  run_name=run_name,
                                                  metadata={
                                                      "langfuse_user_id": langfuse_user_id,
                                                      "langfuse_session_id": session_id,
                                                      "langfuse_tags": [langfuse_tag]
                                                  }))
    elif isinstance(langfuse_session_id, str) and isinstance(query, list):
        invoke_configs = [RunnableConfig(max_concurrency=max_concurrency,
                                          callbacks=[handler] if use_langfuse else [],
                                          run_name=run_name,
                                          metadata={
                                              "langfuse_user_id": langfuse_user_id,
                                              "langfuse_session_id": langfuse_session_id,
                                              "langfuse_tags": [langfuse_tag]
                                          }) for _ in query]
    else:
        invoke_configs = RunnableConfig(max_concurrency=max_concurrency,
                                         callbacks=[handler] if use_langfuse else [],
                                         run_name=run_name,
                                         metadata={
                                             "langfuse_user_id": langfuse_user_id,
                                             "langfuse_session_id": langfuse_session_id,
                                             "langfuse_tags": [langfuse_tag]
                                         })

    return invoke_configs, model_name


@overload
def prompt_parsing(
        model: Type[M],
        failed_model: M,
        query: Union[str, MultimodalInput],
        llm,
        use_langfuse: bool = ...,
        langfuse_user_id: str = ...,
        langfuse_session_id: Union[str, list[str]] = ...,
        langfuse_tag: str = ...,
        run_name: str = ...,
        max_concurrency: int = ...
) -> M: ...


@overload
def prompt_parsing(
        model: Type[M],
        failed_model: M,
        query: List[Union[str, MultimodalInput]],
        llm,
        use_langfuse: bool = ...,
        langfuse_user_id: str = ...,
        langfuse_session_id: Union[str, list[str]] = ...,
        langfuse_tag: str = ...,
        run_name: str = ...,
        max_concurrency: int = ...
) -> List[M]: ...


def prompt_parsing(model: Type[M],
                   failed_model: M,
                   query: Union[str, MultimodalInput, List[Union[str, MultimodalInput]]],
                   llm: BaseChatModel,
                   use_langfuse: bool = False,
                   langfuse_user_id: str = 'user_1',
                   langfuse_session_id: Union[str, list[str]] = 'session_1',
                   langfuse_tag: str = 'langchain',
                   run_name: str = 'promptparsing',
                   max_concurrency: int = None) -> Union[M, list[M]]:
    """
    Force LLM outputs to conform to a specified Pydantic model schema.

    This function wraps LLM calls with structured output parsing, ensuring that
    responses strictly follow the given Pydantic model definition. It supports
    both single-query and batch-query processing, multimodal inputs (text, images, videos),
    and includes automatic retry logic (up to 10 attempts) for failed requests.

    Parameters
    ----------
    model : Type[BaseModel]
        Pydantic model class defining the expected output schema.
    failed_model : BaseModel
        Fallback instance returned if all retries are exhausted.
    query : str, MultimodalInput, or list of (str or MultimodalInput)
        A single query or a list of queries to process. Can be plain text strings
        or MultimodalInput objects containing text with images/videos.
    llm : BaseChatModel
        LangChain chat model instance used for inference.
    use_langfuse: bool
        Whether to use Langfuse.
    langfuse_user_id : str, optional
        User identifier for Langfuse observability tracking. Default is "user_1".
    langfuse_session_id : str or list of str, optional
        Session identifier for Langfuse observability tracking. Default is "session_1".
        If it is a str, then all query will use same session_id
    langfuse_tag : str, optional
        Tags for Langfuse observability tracking. Default is "langchain".
    run_name : str, optional
        Run name for RunnableConfig. Default is "prompt_parsing".
    max_concurrency : int, optional
        Maximum number of concurrent requests for batch processing. If not
        provided, defaults to ``llm.max_concurrency``.

    Returns
    -------
    BaseModel or list of BaseModel
        A single model instance or a list of model instances, depending on the
        input query.

    Example:
    from langchainkit import prompt_parsing, MultimodalInput, LocalLLM
    from pydantic import BaseModel

    llm = LocalLLM.qwen3_14b_awq_think()

    class Response(BaseModel):
        answer: str
        confidence: float

    # Text-only query
    result = prompt_parsing(
        model=Response,
        failed_model=Response(answer="no_answer", confidence=0.0),
        query="What is the capital of France?",
        llm=llm
    )
    print(result.answer)  # "Paris"
    print(result.confidence)  # 1.0

    # Multimodal query with image
    result = prompt_parsing(
        model=Response,
        failed_model=Response(answer="no_answer", confidence=0.0),
        query=MultimodalInput(
            text="What's in this image?",
            img_paths=["path/to/image.jpg"]
        ),
        llm=llm
    )

    # Batch queries (mixed text and multimodal)
    result = prompt_parsing(
        model=Response,
        failed_model=Response(answer="no_answer", confidence=0.0),
        query=[
            "What is the capital of France?",
            MultimodalInput(text="Describe this image", imgs=["https://example.com/image.jpg"]),
            "What is the capital of Italy?"
        ],
        llm=llm
    )
    for each in result:
        print(each.answer)
        print(each.confidence)
    """
    invoke_configs, model_name = _prepare_configs(
        query, llm, use_langfuse, langfuse_user_id,
        langfuse_session_id, langfuse_tag, run_name, max_concurrency
    )

    parser = PydanticOutputParser(pydantic_object=model)
    chain = llm | parser
    system_content = f"回答用户的问题. 把输出结果包裹在 ```json 和 ``` 标签里.\n{parser.get_format_instructions()}"

    # 如果query是单个请求
    if isinstance(query, (str, MultimodalInput)):
        content = _multimodal_to_content(query)
        messages = [SystemMessage(content=system_content), HumanMessage(content=content)]
        return chain.invoke(messages, config=invoke_configs)

    # 如果query是多个请求list，统一转换为message batches
    message_batches = [
        [SystemMessage(content=system_content), HumanMessage(content=_multimodal_to_content(q))]
        for q in query
    ]

    return batch_with_retry(
        llm=chain,
        prompts=message_batches,
        input_config=invoke_configs,
        max_retries=10,
        delay=2,
        failed_value=failed_model,
        llm_name=model_name
    )


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


@overload
async def aprompt_parsing(
        model: Type[M],
        failed_model: M,
        query: Union[str, MultimodalInput],
        llm,
        use_langfuse: bool = ...,
        langfuse_user_id: str = ...,
        langfuse_session_id: Union[str, list[str]] = ...,
        langfuse_tag: str = ...,
        run_name: str = ...,
        max_concurrency: int = ...
) -> M: ...


@overload
async def aprompt_parsing(
        model: Type[M],
        failed_model: M,
        query: List[Union[str, MultimodalInput]],
        llm,
        use_langfuse: bool = ...,
        langfuse_user_id: str = ...,
        langfuse_session_id: Union[str, list[str]] = ...,
        langfuse_tag: str = ...,
        run_name: str = ...,
        max_concurrency: int = ...
) -> List[M]: ...


async def aprompt_parsing(model: Type[M],
                          failed_model: M,
                          query: Union[str, MultimodalInput, List[Union[str, MultimodalInput]]],
                          llm: BaseChatModel,
                          use_langfuse: bool = False,
                          langfuse_user_id: str = 'user_1',
                          langfuse_session_id: Union[str, list[str]] = 'session_1',
                          langfuse_tag: str = 'langchain',
                          run_name: str = 'apromptparsing',
                          max_concurrency: int = None) -> Union[M, list[M]]:
    """
    Async version of prompt_parsing. Force LLM outputs to conform to a specified Pydantic model schema.

    This function wraps async LLM calls with structured output parsing, ensuring that
    responses strictly follow the given Pydantic model definition. It supports
    both single-query and batch-query processing, multimodal inputs (text, images, videos),
    and includes automatic retry logic (up to 10 attempts) for failed requests.

    Parameters
    ----------
    model : Type[BaseModel]
        Pydantic model class defining the expected output schema.
    failed_model : BaseModel
        Fallback instance returned if all retries are exhausted.
    query : str, MultimodalInput, or list of (str or MultimodalInput)
        A single query or a list of queries to process. Can be plain text strings
        or MultimodalInput objects containing text with images/videos.
    llm : BaseChatModel
        LangChain chat model instance used for inference.
    use_langfuse: bool
        Whether to use Langfuse.
    langfuse_user_id : str, optional
        User identifier for Langfuse observability tracking. Default is "user_1".
    langfuse_session_id : str or list of str, optional
        Session identifier for Langfuse observability tracking. Default is "session_1".
        If it is a str, then all query will use same session_id
    langfuse_tag : str, optional
        Tags for Langfuse observability tracking. Default is "langchain".
    run_name : str, optional
        Run name for RunnableConfig. Default is "aprompt_parsing".
    max_concurrency : int, optional
        Maximum number of concurrent requests for batch processing. If not
        provided, defaults to ``llm.max_concurrency``.

    Returns
    -------
    BaseModel or list of BaseModel
        A single model instance or a list of model instances, depending on the
        input query.

    Example:
    from langchainkit import aprompt_parsing, MultimodalInput, LocalLLM
    from pydantic import BaseModel
    import asyncio

    llm = LocalLLM.qwen3_14b_awq_think()

    class Response(BaseModel):
        answer: str
        confidence: float

    # Single text query
    result = await aprompt_parsing(
        model=Response,
        failed_model=Response(answer="no_answer", confidence=0.0),
        query="What is the capital of France?",
        llm=llm
    )
    print(result.answer)  # "Paris"
    print(result.confidence)  # 1.0

    # Single multimodal query
    result = await aprompt_parsing(
        model=Response,
        failed_model=Response(answer="no_answer", confidence=0.0),
        query=MultimodalInput(
            text="What's in this image?",
            img_paths=["path/to/image.jpg"]
        ),
        llm=llm
    )

    # Batch queries (mixed text and multimodal)
    result = await aprompt_parsing(
        model=Response,
        failed_model=Response(answer="no_answer", confidence=0.0),
        query=[
            "What is the capital of France?",
            MultimodalInput(text="Describe this", imgs=["https://example.com/img.jpg"]),
            "What is the capital of Italy?"
        ],
        llm=llm
    )
    for each in result:
        print(each.answer)
        print(each.confidence)
    """
    invoke_configs, model_name = _prepare_configs(
        query, llm, use_langfuse, langfuse_user_id,
        langfuse_session_id, langfuse_tag, run_name, max_concurrency
    )

    parser = PydanticOutputParser(pydantic_object=model)
    chain = llm | parser
    system_content = f"回答用户的问题. 把输出结果包裹在 ```json 和 ``` 标签里.\n{parser.get_format_instructions()}"

    # 如果query是单个请求
    if isinstance(query, (str, MultimodalInput)):
        content = _multimodal_to_content(query)
        messages = [SystemMessage(content=system_content), HumanMessage(content=content)]
        return await chain.ainvoke(messages, config=invoke_configs)

    # 如果query是多个请求list，统一转换为message batches
    message_batches = [
        [SystemMessage(content=system_content), HumanMessage(content=_multimodal_to_content(q))]
        for q in query
    ]

    return await abatch_with_retry(
        llm=chain,
        prompts=message_batches,
        input_config=invoke_configs,
        max_retries=10,
        delay=2,
        failed_value=failed_model,
        llm_name=model_name
    )
