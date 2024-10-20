import os
import os.path as osp
from typing import Union
from stark_qa.skb import *


def load_skb(name: str, 
             root: Union[str, None] = None,
             download_processed: bool = True, 
             **kwargs) -> SKB:
    """
    Load the SKB dataset.

    Args:
        name (str): Name of the dataset. One of 'amazon', 'prime', or 'mag'.
        root (Union[str, None]): Root directory to store the dataset. If None, defaults to the HF cache path.
        download_processed (bool): Whether to download processed data. Default is False. If True, `root` must be provided.
        **kwargs: Additional keyword arguments for the specific dataset class.

    Returns:
        An instance of the specified SKB dataset class.

    Raises:
        ValueError: If the dataset name is not recognized.
        AssertionError: If `root` is not provided when `download_processed` is False.
    """
    assert name in REGISTERED_SKBS, f"Unknown SKB {name}"

    if not download_processed:
        assert root is not None, "root must be provided if download_processed is False"

    if root is None:
        data_root = None
    else:
        root = os.path.abspath(root)
        data_root = osp.join(root, name)

    if name == 'amazon':
        categories = ['Sports_and_Outdoors']
        skb = AmazonSKB(root=data_root,
                        categories=categories,
                        download_processed=download_processed,
                        **kwargs
                        )
    elif name == 'prime':
        skb = PrimeSKB(root=data_root, 
                       download_processed=download_processed,
                       **kwargs)
    
    elif name == 'mag':
        skb = MagSKB(root=data_root, 
                     download_processed=download_processed,
                     **kwargs)
    return skb