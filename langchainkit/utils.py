from time import sleep
from typing import Any, List, Union
from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.runnables import RunnableConfig
from loguru import logger
from tqdm import tqdm

def batch_with_retry(
    llm:BaseChatModel,
    prompts: List,
    input_config: RunnableConfig = None,
    max_retries: int = 10,
    delay: int = 3
) -> List[Union[Any, Exception]]:
    """
    Run llm.batch() with retry on failed items only.

    :param llm: LangChain LLM/Chain instance
    :param prompts: List of prompts (strings, dicts, or LCEL-compatible inputs)
    :param input_config: RunnableConfig
    :param max_retries: Maximum number of retry attempts
    :param delay: Seconds to wait between retries
    :return: List of results in the same order as prompts; failed ones remain as Exception objects
    """
    results: List[Union[Any, Exception]] = [None] * len(prompts)
    remaining_idx = list(range(len(prompts)))
    if input_config is None:
        input_config = RunnableConfig(max_concurrency=llm.max_concurrency)
    for attempt in range(max_retries):
        if not remaining_idx:
            break

        # Run batch on the remaining prompts
        sub_prompts = [prompts[i] for i in remaining_idx]
        sub_results: List[Union[Any, Exception]] = [None] * len(sub_prompts)
        for i, res in tqdm(
                llm.batch_as_completed(
                    sub_prompts,
                    config=input_config,
                    return_exceptions=True
                ),
                total=len(sub_prompts),
                desc=f"Batch attempt {attempt + 1}",
                leave=False
        ):
            sub_results[i] = res

        new_remaining = []
        for idx, res in zip(remaining_idx, sub_results):
            if isinstance(res, Exception):
                new_remaining.append(idx)  # will retry
            else:
                results[idx] = res
        remaining_idx = new_remaining

        if remaining_idx:
            logger.warning(f"[Retry {attempt+1}] {len(remaining_idx)} items failed. Retrying in {delay} seconds...")
            sleep(delay)

    return results
