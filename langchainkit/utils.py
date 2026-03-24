from time import sleep
from typing import Any, List, Union
from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.runnables import RunnableConfig
from loguru import logger
from tqdm import tqdm
from datetime import datetime
import random
import asyncio

# tqdm 描述字符串
tqdm_desc_list = [
    "Model inference (Attempt {attempt})",
    "Generating output (Attempt {attempt})",
    "LLM processing (Round {attempt})",
    "Neural network running (Attempt {attempt})",
    "Tokens generating (Attempt {attempt})",
    "Model predicting (Round {attempt})",
    "Running inference (Attempt {attempt})",
    "LLM synthesizing (Attempt {attempt})",
    "Sequence generation (Attempt {attempt})",
    "Neural compute active (Round {attempt})"
]


def batch_with_retry(
        llm: BaseChatModel,
        prompts: List,
        input_config: Union[RunnableConfig, List[RunnableConfig]] = None,
        max_retries: int = 10,
        delay: int = 3,
        failed_value: Any = None,
        llm_name: str = ''
) -> List[Union[Any, Exception]]:
    """
    Run llm.batch() with retry on failed items only.

    :param llm: LangChain LLM/Chain instance
    :param prompts: List of prompts (strings, dicts, or LCEL-compatible inputs)
    :param input_config: RunnableConfig or List[RunnableConfig]. If list, must match length of prompts
    :param max_retries: Maximum number of retry attempts
    :param delay: Seconds to wait between retries
    :param failed_value: Value to return for failed items after all retries are exhausted (default: None)
    :param llm_name: Name of the LLM model,useful when llm is a RunnableSequence of Langchain
    :return: List of results in the same order as prompts; failed ones return failed_value
    """
    model_name = getattr(llm, "model", llm_name) or getattr(llm, "model_name", llm_name)

    results: List[Union[Any, Exception]] = [failed_value] * len(prompts)
    remaining_idx = list(range(len(prompts)))

    # Handle input_config
    is_config_list = isinstance(input_config, list)
    if input_config is None:
        input_config = RunnableConfig(max_concurrency=getattr(llm, 'max_concurrency', 10))
        is_config_list = False

    for attempt in range(max_retries):
        if not remaining_idx:
            break

        # Run batch on the remaining prompts
        sub_prompts = [prompts[i] for i in remaining_idx]

        # Prepare sub_configs for retry
        if is_config_list:
            sub_configs = [input_config[i] for i in remaining_idx]
        else:
            sub_configs = input_config

        sub_results: List[Union[Any, Exception]] = [None] * len(sub_prompts)

        desc_str = random.choice(tqdm_desc_list).format(attempt=attempt + 1)
        with tqdm(total=len(sub_prompts), desc=desc_str, leave=False) as pbar:
            for i, res in llm.batch_as_completed(
                    sub_prompts,
                    config=sub_configs,
                    return_exceptions=True
            ):
                sub_results[i] = res
                # add current time to progress bar
                now = datetime.now().strftime("%H:%M:%S")
                pbar.set_postfix_str(f'{now} {model_name}')
                pbar.update(1)

        new_remaining = []
        for idx, res in zip(remaining_idx, sub_results):
            if isinstance(res, Exception):
                new_remaining.append(idx)  # will retry
            else:
                results[idx] = res
        remaining_idx = new_remaining

        if remaining_idx:
            logger.warning(f"[Retry {attempt + 1}] {len(remaining_idx)} items failed. Retrying in {delay} seconds...")
            sleep(delay)

    return results


async def abatch_with_retry(
        llm: BaseChatModel,
        prompts: List,
        input_config: Union[RunnableConfig, List[RunnableConfig]] = None,
        max_retries: int = 10,
        delay: int = 3,
        failed_value: Any = None,
        llm_name: str = ''
) -> List[Union[Any, Exception]]:
    """
    Async version of batch_with_retry. Run llm.abatch() with retry on failed items only.

    :param llm: LangChain LLM/Chain instance
    :param prompts: List of prompts (strings, dicts, or LCEL-compatible inputs)
    :param input_config: RunnableConfig or List[RunnableConfig]. If list, must match length of prompts
    :param max_retries: Maximum number of retry attempts
    :param delay: Seconds to wait between retries
    :param failed_value: Value to return for failed items after all retries are exhausted (default: None)
    :param llm_name: Name of the LLM model,useful when llm is a RunnableSequence of Langchain
    :return: List of results in the same order as prompts; failed ones return failed_value
    """
    model_name = getattr(llm, "model", llm_name) or getattr(llm, "model_name", llm_name)

    results: List[Union[Any, Exception]] = [failed_value] * len(prompts)
    remaining_idx = list(range(len(prompts)))

    # Handle input_config
    is_config_list = isinstance(input_config, list)
    if input_config is None:
        input_config = RunnableConfig(max_concurrency=getattr(llm, 'max_concurrency', 10))
        is_config_list = False

    for attempt in range(max_retries):
        if not remaining_idx:
            break

        # Run batch on the remaining prompts
        sub_prompts = [prompts[i] for i in remaining_idx]

        # Prepare sub_configs for retry
        if is_config_list:
            sub_configs = [input_config[i] for i in remaining_idx]
        else:
            sub_configs = input_config

        sub_results: List[Union[Any, Exception]] = [None] * len(sub_prompts)

        desc_str = random.choice(tqdm_desc_list).format(attempt=attempt + 1)
        with tqdm(total=len(sub_prompts), desc=desc_str, leave=False) as pbar:
            async for i, res in llm.abatch_as_completed(
                    sub_prompts,
                    config=sub_configs,
                    return_exceptions=True
            ):
                sub_results[i] = res
                # add current time to progress bar
                now = datetime.now().strftime("%H:%M:%S")
                pbar.set_postfix_str(f'{now} {model_name}')
                pbar.update(1)

        new_remaining = []
        for idx, res in zip(remaining_idx, sub_results):
            if isinstance(res, Exception):
                new_remaining.append(idx)  # will retry
            else:
                results[idx] = res
        remaining_idx = new_remaining

        if remaining_idx:
            logger.warning(f"[Retry {attempt + 1}] {len(remaining_idx)} items failed. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

    return results
