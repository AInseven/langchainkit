# LangChainKit

LangChainKit simplifies the process of prompting LLMs to return structured outputs using [LangChain](https://github.com/langchain-ai/langchain) and [LangFuse](https://github.com/langfuse/langfuse).

--- 

## 🚀 Features

- 🔧 **Simplified Qwen3 + vLLM integration**  
  Automatically configure `enable_thinking` and other complex settings for Qwen3 models when using vLLM.

- 🧠 **Structured Output via LangChain**  
  Easily prompt the LLM to generate structured outputs, including batch prompting support, with minimal setup.

- 📊 **LangFuse Integration**  
  Track and evaluate LLM performance using LangFuse, without writing boilerplate code.

---

## Installation

```bash
pip install langchainkit
```

## Quick Start

### Configuration

Set up your environment variables in .env file:

```bash
DEEPSEEK_API_KEY=your deepseek api key
MOONSHOT_API_KEY=...
OPENROUTER_API_KEY=...
ARK_API_KEY=...
DASHSCOPE_API_KEY=...
LOCAL_VLLM_BASE_URL=http://172.20.14.28:8000/v1
LOCAL_VLLM_API_KEY=...

LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=...
```

```python
from langchainkit import GeneralLLM,prompt_parsing
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv() # load .env file

llm = GeneralLLM.deepseek_chat()

class Response(BaseModel):
    answer: str
    confidence: float

result = prompt_parsing(
    model=Response,
    failed_model=Response(answer="no_answer", confidence=0.0),
    query="What is the capital of France?",
    llm=llm,
    use_langfuse=False 
)
print(result.answer)  # "Paris"
print(result.confidence)  # 1.0

result = prompt_parsing(
    model=Response,
    failed_model=Response(answer="no_answer", confidence=0.0),
    query=["What is the capital of France?",
           "What is the capital of Germany?",
           "What is the capital of Italy?"],
    llm=llm,
    use_langfuse=False
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
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the core framework
- [vLLM](https://github.com/vllm-project/vllm) for high-throughput LLM inference
- [Langfuse](https://github.com/langfuse/langfuse) for observability and monitoring