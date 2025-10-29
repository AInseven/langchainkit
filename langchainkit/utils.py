from time import sleep
from typing import Any, List, Union
from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.runnables import RunnableConfig
from loguru import logger
from tqdm import tqdm
from datetime import datetime

def batch_with_retry(
    llm:BaseChatModel,
    prompts: List,
    input_config: Union[RunnableConfig, List[RunnableConfig]] = None,
    max_retries: int = 10,
    delay: int = 3,
    failed_value: Any = None
) -> List[Union[Any, Exception]]:
    """
    Run llm.batch() with retry on failed items only.

    :param llm: LangChain LLM/Chain instance
    :param prompts: List of prompts (strings, dicts, or LCEL-compatible inputs)
    :param input_config: RunnableConfig or List[RunnableConfig]. If list, must match length of prompts
    :param max_retries: Maximum number of retry attempts
    :param delay: Seconds to wait between retries
    :param failed_value: Value to return for failed items after all retries are exhausted (default: None)
    :return: List of results in the same order as prompts; failed ones return failed_value
    """
    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
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

        with tqdm(total=len(sub_prompts), desc=f"{model_name} Attempt {attempt + 1}", leave=False) as pbar:
            for i, res in llm.batch_as_completed(
                    sub_prompts,
                    config=sub_configs,
                    return_exceptions=True
            ):
                sub_results[i] = res
                # add current time to progress bar
                now = datetime.now().strftime("%H:%M:%S")
                pbar.set_postfix_str(now)
                pbar.update(1)

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
