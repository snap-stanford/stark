import torch
from typing import List, Union

loaded_llm2vec_models = {}


def get_llm2vec_embeddings(text: Union[str, List[str]], 
                           model_name: str = 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp', 
                           peft_model_name: str = 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse',
                           instruction: str = '',
                           device: str = 'cuda', 
                           max_length: str = 2048,
                           norm=True) -> torch.Tensor:
    """
    Get LLM2Vec embeddings for the given text.

    Args:
        text (Union[str, List[str]]): The input text to be embedded.
        model_name (str): The model to use for embedding. 
        peft_model_name (str): The model to use for PEFT embeddings. 
    
    Returns:
        torch.Tensor: The embedding(s) of the input text(s).
    """
    try:
        from llm2vec import LLM2Vec
    except ImportError:
        raise ImportError("Please install the llm2vec package using `pip install llm2vec`.")

    if peft_model_name in loaded_llm2vec_models:
        l2v = loaded_llm2vec_models[peft_model_name]
    else:
        l2v = LLM2Vec.from_pretrained(
            model_name,
            peft_model_name_or_path=peft_model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            max_length=max_length
        )
        loaded_llm2vec_models[peft_model_name] = l2v
    
    if isinstance(text, str):
        text = [text]

    if len(instruction) > 0:
        text = [[instruction, t] for t in text]
    embeddings = l2v.encode(text, batch_size=len(text))
    if norm:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.view(len(text), -1)
