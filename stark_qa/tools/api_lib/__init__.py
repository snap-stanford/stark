import os
import os.path as osp
import warnings
import openai

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

