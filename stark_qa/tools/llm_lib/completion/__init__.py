import os
import os.path as osp
import warnings
import openai
from .claude import complete_text_claude
from .gpt import get_gpt_output
from .huggingface import complete_text_hf
from .ollama import complete_text_ollama

try:
    assert os.environ.get("ANTHROPIC_API_KEY") is not None
    print("Anthropic API key found.")
except Exception as e:
    pass


try:
    assert os.environ.get("VOYAGE_API_KEY") is not None
    print("Voyage API key found.")
except Exception as e:
    pass


try:
    assert os.environ.get("OPENAI_API_KEY") is not None
    print("OpenAI API key found.")

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    openai.organization = os.environ.get("OPENAI_ORG")
except Exception as e:
    pass


# Register the available text completion LLMs
REGISTERED_TEXT_COMPLETION_LLMS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "claude-2.1",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "huggingface/codellama/CodeLlama-7b-hf",
    # Ollama models (common sizes)
    "llama3.2",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.2:70b",
    "llama3.1",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.1:405b",
    "llama3",
    "llama3:8b",
    "llama3:70b",
    "mistral",
    "mistral:7b",
    "codellama",
    "codellama:7b",
    "codellama:13b",
    "codellama:34b",
    "phi3",
    "phi3:3.8b",
    "phi3:14b",
    "qwen2.5",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "qwen2.5:72b",
    "gemma2",
    "gemma2:2b",
    "gemma2:9b",
    "gemma2:27b",
]
