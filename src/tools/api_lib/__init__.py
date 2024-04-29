import os
import os.path as osp
import warnings

# get current dir
cur_dir = osp.dirname(os.path.realpath(__file__))
outer_dir = osp.join(cur_dir, "..", "..", "..")

# setup anthropic API key
try:   
    import anthropic
    api_key = open(osp.join(outer_dir, "config/claude_api_key.txt")).read().strip()
    anthropic_client = anthropic.Anthropic(api_key=api_key)
except Exception as e:
    print(e)
    print("Could not load anthropic API key config/claude_api_key.txt.")
    
# setup OpenAI API key
try:
    import openai
    openai.organization, openai.api_key = open(
        osp.join(outer_dir, "config/openai_api_key.txt")
        ).read().strip().split(":")    
    os.environ["OPENAI_API_KEY"] = openai.api_key 
except Exception as e:
    print(e)
    print("Could not load OpenAI API key config/openai_api_key.txt.")
