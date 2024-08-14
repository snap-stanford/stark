import re
import torch
import voyageai
from functools import partial
import time
import multiprocessing


def get_voyage_embedding(text: str, 
                         model: str = "voyage-large-2-instruct",
                         max_retry: int = 10,
                         sleep_time: int = 60) -> torch.FloatTensor:
    """
    Get the voyage embedding for a given text.

    Args:
        text (str): The input text to be embedded.
        model (str): The model to use for embedding. Default is "voyage-large-2-instruct".
        max_retry (int): Maximum number of retries in case of an error. Default is 1.
        sleep_time (int): Sleep time between retries in seconds. Default is 0.

    Returns:
        torch.FloatTensor: The embedding of the input text.
    """
    assert isinstance(text, str), f'text must be str, but got {type(text)}'
    assert len(text) > 0, 'text to be embedded should be non-empty'
    
    client = voyageai.Client()
    for _ in range(max_retry):
        try:
            emb = client.embed([text], model=model).embeddings
            return torch.FloatTensor(emb).view(1, -1)
        except voyageai.error.InvalidRequestError as e:
            print(f'{e}')
            e = str(e)
            ori_length = len(text.split(' '))
            match = re.search(r'The max allowed tokens per submitted batch is (\d+). Your batch has (\d+) tokens after truncation', e)
            if match is not None:
                max_length = int(match.group(1))
                cur_length = int(match.group(2))
                ratio = float(max_length) / cur_length
                for reduce_rate in range(9, 0, -1):
                    shorten_text = text.split(' ')
                    length = int(ratio * ori_length * (reduce_rate * 0.1))
                    shorten_text = ' '.join(shorten_text[:length])
                    try:
                        emb = client.embed([shorten_text], model=model).embeddings
                        print(f'length={length} works! reduce_rate={0.1 * reduce_rate}.')
                        return torch.FloatTensor(emb).view(1, -1)
                    except: 
                        continue
        except Exception as e:
            print(f'{e}, sleep for {sleep_time} seconds')
            time.sleep(sleep_time)
    raise RuntimeError("Failed to get embedding after maximum retries")


def get_voyage_embeddings(texts: list, 
                          n_max_nodes: int = 128, 
                          model: str = "voyage-large-2-instruct") -> torch.FloatTensor:
    """
    Get embeddings for a list of texts using voyage's embedding model.

    Args:
        texts (list): List of input texts to be embedded.
        n_max_nodes (int): Maximum number of parallel processes. Default is 5.
        model (str): The model to use for embedding. Default is "voyage-large-2-instruct".

    Returns:
        torch.FloatTensor: A tensor containing embeddings for all input texts.
    """
    assert isinstance(texts, list), f'texts must be list, but got {type(texts)}'
    assert all([len(s) > 0 for s in texts]), 'every string in the `texts` list to be embedded should be non-empty'

    processes = min(len(texts), n_max_nodes)
    ada_encoder = partial(get_voyage_embedding, model=model)
    
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(ada_encoder, texts)

    results = torch.cat(results, dim=0)
    return results
