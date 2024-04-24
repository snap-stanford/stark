import os.path as osp
from src.benchmarks.semistruct import AmazonSemiStruct, PrimeKGSemiStruct, MagSemiStruct


def get_semistructured_data(name, root='data/'):
    data_root = osp.join(root, name)
    if name == 'amazon':
        categories = ['Sports_and_Outdoors']
        raw_data_dir = osp.join(data_root, 'raw')
        save_path = osp.join(data_root, 'processed')
        database = AmazonSemiStruct(categories, 
                                    raw_data_dir=raw_data_dir,
                                    save_path=save_path,
                                    meta_link_types=['brand'],
                                    indirected=True,
                                    )
    if name == 'primekg':
        save_path = osp.join(data_root, 'processed')
        database = PrimeKGSemiStruct(root=data_root,
                                     save_path=save_path)
    
    if name == 'mag':
        save_path = osp.join(data_root, 'processed')
        database = MagSemiStruct(root=data_root, 
                                 schema_dir=osp.join(data_root, 'schema'),
                                 save_path=save_path)
    return database
