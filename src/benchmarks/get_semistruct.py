import os.path as osp
from src.benchmarks.semistruct import AmazonSemiStruct, PrimeKGSemiStruct, MagSemiStruct


def get_semistructured_data(name: str, 
                            root: str = 'data/', 
                            download_processed: bool = False, 
                            **kwargs):

    data_root = osp.join(root, name)
    if name == 'amazon':
        categories = ['Sports_and_Outdoors']
        kb = AmazonSemiStruct(root=data_root,
                              categories=categories,
                              download_processed=download_processed,
                              **kwargs
                              )
    elif name == 'primekg':
        kb = PrimeKGSemiStruct(root=data_root, 
                                download_processed=download_processed,
                                **kwargs)
    
    elif name == 'mag':
        kb = MagSemiStruct(root=data_root, 
                            download_processed=download_processed,
                            **kwargs)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return kb