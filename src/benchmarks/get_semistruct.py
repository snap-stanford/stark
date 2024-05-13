import os.path as osp
from src.benchmarks.semistruct import AmazonSemiStruct, PrimeKGSemiStruct, MagSemiStruct


def get_semistructured_data(name, root='data/', download_processed=True, **kwargs):
    data_root = osp.join(root, name)
    if name == 'amazon':
        categories = ['Sports_and_Outdoors']
        kb = AmazonSemiStruct(root=data_root,
                              categories=categories,
                              download_processed=download_processed,
                              **kwargs
                              )
    if name == 'primekg':
        kb = PrimeKGSemiStruct(root=data_root, 
                                download_processed=download_processed,
                                **kwargs)
    
    if name == 'mag':
        kb = MagSemiStruct(root=data_root, 
                            download_processed=download_processed,
                            **kwargs)
    return kb