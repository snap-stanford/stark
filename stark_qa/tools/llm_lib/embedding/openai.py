import re
import torch
import openai
from functools import partial
import time
import multiprocessing


def get_openai_embedding(text: str, 
                         model: str = "text-embedding-ada-002",
                         max_retry: int = 10,
                         sleep_time: int = 0) -> torch.FloatTensor:
    """
    Get the OpenAI embedding for a given text.

    Args:
        text (str): The input text to be embedded.
        model (str): The model to use for embedding. Default is "text-embedding-ada-002".
        max_retry (int): Maximum number of retries in case of an error. Default is 1.
        sleep_time (int): Sleep time between retries in seconds. Default is 0.

    Returns:
        torch.FloatTensor: The embedding of the input text.
    """
    assert isinstance(text, str), f'text must be str, but got {type(text)}'
    assert len(text) > 0, 'text to be embedded should be non-empty'
    
    client = openai.OpenAI()
    
    for _ in range(max_retry):
        try:
            emb = client.embeddings.create(input=[text], model=model)
            return torch.FloatTensor(emb.data[0].embedding).view(1, -1)
        except openai.BadRequestError as e:
            print(f'{e}')
            e = str(e)
            ori_length = len(text.split(' '))
            match = re.search(r'maximum context length is (\d+) tokens, however you requested (\d+) tokens', e)
            if match is not None:
                max_length = int(match.group(1))
                cur_length = int(match.group(2))
                ratio = float(max_length) / cur_length
                for reduce_rate in range(9, 0, -1):
                    shorten_text = text.split(' ')
                    length = int(ratio * ori_length * (reduce_rate * 0.1))
                    shorten_text = ' '.join(shorten_text[:length])
                    try:
                        emb = client.embeddings.create(input=[shorten_text], model=model)
                        print(f'length={length} works! reduce_rate={0.1 * reduce_rate}.')
                        return torch.FloatTensor(emb.data[0].embedding).view(1, -1)
                    except: 
                        continue
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            print(f'{e}, sleep for {sleep_time} seconds')
            time.sleep(sleep_time)
    raise RuntimeError("Failed to get embedding after maximum retries")


def get_openai_embeddings(texts: list, 
                          model: str = "text-embedding-ada-002",
                          n_max_nodes: int = 5) -> torch.FloatTensor:
    """
    Get embeddings for a list of texts using OpenAI's embedding model.

    Args:
        texts (list): List of input texts to be embedded.
        n_max_nodes (int): Maximum number of parallel processes. Default is 5.
        model (str): The model to use for embedding. Default is "text-embedding-ada-002".

    Returns:
        torch.FloatTensor: A tensor containing embeddings for all input texts.
    """
    assert isinstance(texts, list), f'texts must be list, but got {type(texts)}'
    assert all([len(s) > 0 for s in texts]), 'every string in the `texts` list to be embedded should be non-empty'

    processes = min(len(texts), n_max_nodes)
    ada_encoder = partial(get_openai_embedding, model=model)
    
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(ada_encoder, texts)

    results = torch.cat(results, dim=0)
    return results
