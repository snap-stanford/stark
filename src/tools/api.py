import os
import os.path as osp
import warnings
import multiprocessing
from functools import partial
from src.tools.api_lib.claude import complete_text_claude
from src.tools.api_lib.gpt import get_gpt_output
from src.tools.api_lib.openai_emb import get_openai_embedding, get_openai_embeddings
from src.tools.api_lib.huggingface import complete_text_hf
    

# setup parameters for retrying API calls and the sleep time between retries
MAX_OPENAI_RETRY, OPENAI_SLEEP_TIME = 5, 60
MAX_CLAUDE_RETRY, CLAUDE_SLEEP_TIME = 10, 0


registered_text_completion_llms = {
    "gpt-4-1106-preview",
    "gpt-4-0125-preview", "gpt-4-turbo-preview",
    "gpt-4-turbo", "gpt-4-turbo-2024-04-09"
    "gpt-4-turbo",
    "claude-2.1",
    "claude-3-opus-20240229", 
    "claude-3-sonnet-20240229", 
    "claude-3-haiku-20240307",
    "huggingface/codellama/CodeLlama-7b-hf",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
}


def parallel_func(func, n_max_nodes=5):
    '''
    A general function to call a function on a list of inputs.
    '''
    def _parallel_func(inputs: list, **kwargs):
        partial_func = partial(func, **kwargs)
        processes = min(len(inputs), n_max_nodes)
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(partial_func, inputs)
        return results
    return _parallel_func


def get_llm_output(message, 
                   model="gpt-4-0125-preview", 
                   max_tokens=2048, 
                   temperature=1, 
                   json_object=False
                   ):
    '''
    A general function to complete a prompt using the specified model.
    '''
    if model not in registered_text_completion_llms:
        warnings.warn(f"Model {model} is not registered. You may still be able to use it.")
    kwargs = {'message': message, 
              'model': model, 
              'max_tokens': max_tokens, 
              'temperature': temperature, 
              'json_object': json_object}
    
    if 'gpt-4' in model:
        kwargs.update({'max_retry': MAX_OPENAI_RETRY, 'sleep_time': OPENAI_SLEEP_TIME})
        return get_gpt_output(**kwargs)
    elif 'claude' in model:
        kwargs.update({'max_retry': MAX_CLAUDE_RETRY, 'sleep_time': CLAUDE_SLEEP_TIME})
        return complete_text_claude(**kwargs)
    elif 'huggingface' in model:
        return complete_text_hf(**kwargs)
    else:
        raise ValueError(f"Model {model} not recognized.")
    
    
complete_texts_claude = parallel_func(complete_text_claude)
complete_texts_hf = parallel_func(complete_text_hf)
get_gpt_outputs = parallel_func(get_gpt_output)
get_llm_outputs = parallel_func(get_llm_output)
