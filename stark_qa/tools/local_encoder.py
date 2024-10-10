import torch
from typing import List, Union

loaded_llm_models = {}


def get_llm2vec_embeddings(text: Union[str, List[str]], 
                           model_name: str = 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp', 
                           peft_model_name: str = 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse',
                           instruction: str = '',
                           device: str = 'cuda', 
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

    if peft_model_name in loaded_llm_models:
        l2v = loaded_llm_models[peft_model_name]
    else:
        l2v = LLM2Vec.from_pretrained(
            model_name,
            peft_model_name_or_path=peft_model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        loaded_llm_models[peft_model_name] = l2v
    
    if isinstance(text, str):
        text = [text]

    if len(instruction) > 0:
        text = [[instruction, t] for t in text]
    embeddings = l2v.encode(text, batch_size=len(text))
    if norm:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.view(len(text), -1)


def get_gritlm_embeddings(text: Union[str, List[str]],
                          model_name: str = 'GritLM/GritLM-7B', 
                          instruction: str = '',
                          device: str = 'cuda'
                          ) -> torch.Tensor:

        try:
            from gritlm import GritLM
        except ImportError:
            raise ImportError("Please install the gritlm package using `pip install gritlm`.")

        def gritlm_instruction(instruction):
            return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

        """
        Get GritLM embeddings for the given text.
    
        Args:
            text (Union[str, List[str]]): The input text to be embedded.
            instruction (str): The instruction to be used for GritLM.
            model_name (str): The model to use for embedding.

        Returns:
            torch.Tensor: The embedding(s) of the input text(s).
        """

        if model_name in loaded_llm_models:
            gritlm_model = loaded_llm_models[model_name]
        else:
            gritlm_model = GritLM(model_name, torch_dtype=torch.bfloat16)
            loaded_llm_models[model_name] = gritlm_model
        
        if isinstance(text, str):
            text = [text]
    
        embeddings = gritlm_model.encode(text, instruction=gritlm_instruction(instruction))
        embeddings = torch.from_numpy(embeddings)
        return embeddings.view(len(text), -1)

