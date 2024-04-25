import os.path as osp
from src.benchmarks.semistruct import AmazonSemiStruct, PrimeKGSemiStruct, MagSemiStruct


def get_semistructured_data(name, root='data/', download_processed=True):
    data_root = osp.join(root, name)
    if name == 'amazon':
        categories = ['Sports_and_Outdoors']
        database = AmazonSemiStruct(root=data_root,
                                    categories=categories,
                                    meta_link_types=['brand'],
                                    indirected=True,
                                    download_processed=download_processed
                                    )
    if name == 'primekg':
        database = PrimeKGSemiStruct(root=data_root, 
                                     download_processed=download_processed)
    
    if name == 'mag':
        database = MagSemiStruct(root=data_root, 
                                 download_processed=download_processed)
    return database
