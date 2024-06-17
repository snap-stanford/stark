import os
import os.path as osp
import warnings

# Get the current directory
cur_dir = osp.dirname(os.path.realpath(__file__))

# Define the outer directory path
outer_dir = osp.join(cur_dir, "..", "..", "..")

# Setup Anthropic API key
anthropic_client = None
try:
    import anthropic
    
    # Read the API key from the configuration file
    api_key_path = osp.join(outer_dir, "config", "claude_api_key.txt")
    with open(api_key_path, 'r') as key_file:
        api_key = key_file.read().strip()
    
    # Initialize the Anthropic client with the API key
    anthropic_client = anthropic.Anthropic(api_key=api_key)
except Exception as e:
    try: 
        assert os.environ.get("ANTHROPIC_API_KEY") is not None
        anthropic_client  = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    except Exception as e:
        warnings.warn(f"Could not load Anthropic API key from {api_key_path} or environment variable ANTHROPIC_API_KEY.")

# Setup OpenAI API key
try:
    import openai
    
    # Read the API key and organization ID from the configuration file
    api_key_path = osp.join(outer_dir, "config", "openai_api_key.txt")
    with open(api_key_path, 'r') as key_file:
        openai_org, openai_key = key_file.read().strip().split(":")
    
    # Set the OpenAI organization and API key
    openai.organization = openai_org
    openai.api_key = openai_key
    
    # Set the OpenAI API key in the environment variables
    os.environ["OPENAI_API_KEY"] = openai_key
except Exception as e:
    try:
        assert os.environ.get("OPENAI_API_KEY") is not None
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.organization = os.environ.get("OPENAI_ORG")
    except Exception as e:
        warnings.warn(f"Could not load OpenAI API key from {api_key_path} or environment variable OPENAI_API_KEY. ")
