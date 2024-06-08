import os.path as osp
from stark_qa.skb import AmazonSKB, PrimeSKB, MagSKB


def load_skb(name: str, 
             root: str = 'data/', 
             download_processed: bool = False, 
             **kwargs):

    data_root = osp.join(root, name)
    if name == 'amazon':
        categories = ['Sports_and_Outdoors']
        kb = AmazonSKB(root=data_root,
                       categories=categories,
                       download_processed=download_processed,
                       **kwargs
                       )
    elif name == 'prime':
        kb = PrimeSKB(root=data_root, 
                      download_processed=download_processed,
                        **kwargs)
    
    elif name == 'mag':
        kb = MagSKB(root=data_root, 
                    download_processed=download_processed,
                    **kwargs)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return kb