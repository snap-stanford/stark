import torch
from typing import Any, Union, List, Dict, Optional
from stark_qa.tools.llm_lib.embedding import *
from stark_qa.tools.loose_match import loose_match


def get_llm_embeddings(text: Union[str, List[str]],
                       model: str,
                       **encode_kwargs: Any) -> torch.Tensor:
    """
    Get embeddings for the given text using the specified model.

    Args:
        text (Union[str, List[str]]): The input text or list of texts to be embedded.
        model (str): The name of the embedding model to use.

    Returns:
        torch.Tensor: Embeddings with shape (num_texts, embedding_dim).
    """
    if loose_match(model, REGISTERED_TEXT_COMPLETION_LLMS) is False:
        warnings.warn(f"Model {model} is not registered. You may still be able to use it.")
    
    if isinstance(text, str):
        text = [text]
        
    if 'GritLM' in model:
        emb = get_gritlm_embeddings(text, model, **encode_kwargs)
    elif 'LLM2Vec' in model:
        emb = get_llm2vec_embeddings(text, model, **encode_kwargs)
    elif 'voyage' in model:
        emb = get_voyage_embeddings(text, model, **encode_kwargs)
    elif 'text-embedding' in model:
        emb = get_openai_embeddings(text, model, **encode_kwargs)
    return emb.view(len(text), -1)

