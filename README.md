# LangChainKit

LangChainKit makes it easier to work with Qwen3 models via [vLLM](https://github.com/vllm-project/vllm), and simplifies the process of prompting LLMs to return structured outputs using [LangChain](https://github.com/langchain-ai/langchain) and [LangFuse](https://github.com/langfuse/langfuse).

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

### Basic Usage

```python
from dotenv import load_dotenv

load_dotenv() # load .env file

from langchainkit import LocalLLM

llm = LocalLLM().qwen3_14b_awq_think
res= llm.invoke('hello')
print(res.content) # Hello! How can I assist you today? 😊
```

### Structured Output

```python
from langchainkit import prompt_parsing
from pydantic import BaseModel

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
```

## Configuration

Set up your environment variables in .env file:

```bash
LOCAL_VLLM_BASE_URL=http://172.20.14.28:8000/v1
LOCAL_VLLM_API_KEY=your vLLM api key
LANGFUSE_SECRET_KEY=your langfuse secret key
LANGFUSE_PUBLIC_KEY=your langfuse public key
LANGFUSE_HOST=your langfuse host
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the core framework
- [vLLM](https://github.com/vllm-project/vllm) for high-throughput LLM inference
- [Langfuse](https://github.com/langfuse/langfuse) for observability and monitoring