import os.path as osp
from src.benchmarks.semistruct import AmazonSemiStruct, PrimeKGSemiStruct, MagSemiStruct


def get_semistructured_data(name, root='/dfs/project/kgrlm/benchmark/'):
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
        metadata_path = osp.join(data_root, 'primekg_metadata_extended.pkl')
        kg_path = osp.join(root, 'kg.csv')
        database = PrimeKGSemiStruct(save_path, metadata_path, kg_path)
    
    if name == 'mag':
        save_path = osp.join(data_root, 'processed')
        raw_data_dir = osp.join(data_root, 'raw')
        database = MagSemiStruct(root='/dfs/project/kgrlm/data', 
                                 schema_dir=osp.join(raw_data_dir, 'schema'),
                                 save_path=save_path)
    return database
