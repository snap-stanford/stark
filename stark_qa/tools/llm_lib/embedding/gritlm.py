import torch
from typing import List, Union

loaded_gritlm_models = {}


def get_gritlm_embeddings(text: Union[str, List[str]],
                          model_name: str = 'GritLM/GritLM-7B', 
                          instruction: str = '',
                          device: str = 'cuda',
                          max_length: str = 2048,
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

        if model_name in loaded_gritlm_models:
            gritlm_model = loaded_gritlm_models[model_name]
        else:
            gritlm_model = GritLM(model_name, torch_dtype=torch.bfloat16)
            loaded_gritlm_models[model_name] = gritlm_model
        
        if isinstance(text, str):
            text = [text]
    
        embeddings = gritlm_model.encode(text, instruction=gritlm_instruction(instruction),
                                         max_length=max_length)
        embeddings = torch.from_numpy(embeddings)
        return embeddings.view(len(text), -1)

