import os
from typing import Union
from stark_qa.retrieval import STaRKDataset

REGISTERED_DATASETS = ['amazon', 'prime', 'mag']


def load_qa(name: str,
            root: Union[str, None] = None,
            human_generated_eval: bool = False) -> STaRKDataset:
    """
    Load the QA dataset.

    Args:
        name (str): Name of the dataset. One of 'amazon', 'prime', or 'mag'.
        root (Union[str, None]): Root directory to store the dataset. If not provided, the default Hugging Face cache path is used.
        human_generated_eval (bool): Whether to use human-generated evaluation data. Default is False.

    Returns:
        STaRKDataset: The loaded STaRK dataset.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    
    if root is not None:
        if not os.path.isabs(root):
            root = os.path.abspath(root)

    if name in REGISTERED_DATASETS:
        return STaRKDataset(name, root, 
                            human_generated_eval=human_generated_eval)
    else:
        raise ValueError(f"Unknown dataset {name}")
    
    
    