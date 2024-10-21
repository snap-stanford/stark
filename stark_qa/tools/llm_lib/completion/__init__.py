import os
import os.path as osp
import warnings
import openai
from .claude import complete_text_claude
from .gpt import get_gpt_output
from .huggingface import complete_text_hf

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
    "huggingface/codellama/CodeLlama-7b-hf"
]