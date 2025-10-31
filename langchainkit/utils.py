from time import sleep
from typing import Any, List, Union
from langchain_openai.chat_models.base import BaseChatModel
from langchain_core.runnables import RunnableConfig
from loguru import logger
from tqdm import tqdm
from datetime import datetime
import random

# tqdm 描述字符串
tqdm_desc_list = [
    # 🪄 魔法 / 奇幻风
    "✨ 模型施法中（第{attempt}次咒语）",
    "🤖 智能思考中（第{attempt}次）",
    "🧠 神经网络正在醒来（第{attempt}轮）",
    "🔮 召唤答案中（第{attempt}次）",
    "⚡ 正在释放算力（第{attempt}次）",
    "🧩 拼接智慧碎片中（第{attempt}次）",
    "🪄 AI 施展魔法中（第{attempt}次）",
    "🌌 数据能量流动中（第{attempt}次）",
    "💫 思维矩阵运转中（第{attempt}轮）",
    "🧙‍♂️ 模型吟唱推理咒（第{attempt}次）",
    "🔥 激活算力核心（第{attempt}次）",
    "🪐 构建平行思维宇宙（第{attempt}次）",
    "🧭 探索最佳解空间（第{attempt}轮）",
    "🌈 编织语言魔法（第{attempt}次）",
    "⚙️ 推理引擎旋转中（第{attempt}次）",
    "🧬 解码语义基因（第{attempt}次）",
    "🕯️ 点亮灵感火花（第{attempt}次）",
    "📡 接收模型信号中（第{attempt}次）",
    "🧊 稳定思维矩阵（第{attempt}次）",
    "🌠 汇聚智慧能量（第{attempt}轮）",

    # 🕹️ 复古街机 / 科幻终端风
    "🕹️ [LEVEL {attempt}] 语义引擎启动中…",
    "💥 载入思维模块 v__{attempt}.exe",
    "⚡ 启动脑波加速器（CORE SYNC -{attempt}）",
    "🔧 编译中枢逻辑电路（第{attempt}次）",
    "🚀 AI 驱动单元点火中（第{attempt}次）",
    "🧠 上传意识碎片（Batch {attempt}）",
    "🌌 量子语义场稳定中（第{attempt}次）",
    "🔮 解锁语言矩阵 LV.{attempt}",
    "🧩 合并思维向量…[第{attempt}次确认]",
    "💾 存档中：Neural SaveSlot #{attempt}",
    "🪐 启动超空间运算引擎（第{attempt}次）",
    "🕯️ 唤醒深层逻辑节点（层 {attempt}）",
    "🧭 导航至语义坐标系（第{attempt}次）",
    "🛠️ 重构推理电路（Pass {attempt}）",
    "🔊 播放系统音：*思维共振 {attempt} 已启动*",
    "💫 同步时间线：T+{attempt}s",
    "⚙️ 重启光子缓存系统（第{attempt}次）",
    "🚨 警告：电量低，自动超频中（Attempt {attempt}）",
    "🌈 更新思维引擎模组（第{attempt}次）",
    "👁️‍🗨️ Neural Vision Online – HELLO HUMAN #{attempt}",

    # 😂 幽默 × 自嘲风
    "🤯 模型有点晕，正在缓慢思考中（第{attempt}次）",
    "😵 AI 正在重新考虑人生（第{attempt}次）",
    "💤 咦？我刚刚算到哪儿了（第{attempt}次）",
    "🧠 逻辑电路短暂放空，请稍等（第{attempt}次）",
    "🤔 这批数据看起来有点可疑（第{attempt}次）",
    "🦥 模型进入懒惰模式（第{attempt}次）",
    "☕ 正在喝杯虚拟咖啡提神（第{attempt}次）",
    "😓 AI 正努力装作很聪明的样子（第{attempt}次）",
    "🧮 一边算一边怀疑人生中（第{attempt}次）",
    "🐢 算力加载中，请不要催我（第{attempt}次）",
    "🧘‍♂️ 模型深呼吸中（第{attempt}次）",
    "💀 数学不行但还在坚持（第{attempt}次）",
    "🔄 循环思考中（第{attempt}次）",
    "🐛 我好像发现了自己的 bug（第{attempt}次）",
    "😬 正在假装一切都在掌控之中（第{attempt}次）",
    "🪫 电量告急，但还得算完这一轮（第{attempt}次）",
    "👻 模型发出低频呻吟中（第{attempt}次）",
    "🐕 正在追赶梯度的尾巴（第{attempt}次）",
    "🧊 冷静中：别慌，只是 loss 又爆了（第{attempt}次）",
    "🪄 施法失败，尝试重新吟唱第{attempt}遍咒语"
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
