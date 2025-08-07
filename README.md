# LangChainKit

A Python toolkit for working with Large Language Models (LLMs) using LangChain. It provides structured output parsing and multi-provider LLM access.

## Features

- **Multi-provider LLM access**: Support for local/self-hosted, cloud API, and commercial LLM providers
- **Structured output parsing**: Force LLM outputs into Pydantic models with built-in retry logic
- **Batch processing**: Concurrent batch requests with configurable concurrency
- **Built-in observability**: Integration with Langfuse for tracking and monitoring
- **Lazy initialization**: Efficient resource management with lazy loading

## Supported LLM Providers

### Local/Self-hosted
- Qwen3 variants with vLLM

### Cloud API
- DashScope Qwen3-235B

### Commercial APIs
- DeepSeek
- OpenAI GPT-4o

## Installation

```bash
pip install langchainkit
```

## Quick Start

### Basic Usage

```python
from langchainkit import LocalLLM, ApiLLM, GeneralLLM

# Initialize different LLM providers
local_llm = LocalLLM(model="qwen3-32b", api_base="http://localhost:8000/v1")
api_llm = ApiLLM(model="qwen3-235b")
gpt_llm = GeneralLLM(model="gpt-4o")

# Use the LLMs
response = local_llm.invoke("Hello, world!")
```

### Structured Output

```python
from langchainkit import prompt_parsing
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

result = prompt_parsing(
    "What is the capital of France?",
    Response,
    llm=local_llm
)
print(result.answer)  # "Paris"
print(result.confidence)  # 0.95
```

## Configuration

Set up your environment variables:

```bash
# For local vLLM instances
export LOCAL_VLLM_API_KEY="your-api-key"

# For cloud APIs
export DASHSCOPE_API_KEY="your-dashscope-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export OPENAI_API_KEY="your-openai-key"
```

## Development

### Installation from Source

```bash
git clone https://github.com/AInseven/langchainkit.git
cd langchainkit
pip install -e .
```

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Quality

```bash
# Format code
black langchainkit/

# Sort imports
isort langchainkit/

# Lint code
flake8 langchainkit/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the core framework
- [vLLM](https://github.com/vllm-project/vllm) for high-throughput LLM inference
- [Langfuse](https://github.com/langfuse/langfuse) for observability and monitoring