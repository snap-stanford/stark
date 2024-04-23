import os.path as osp
from src.benchmarks.qa_datasets import AmazonSTaRKDataset, PrimeKGSTaRKDataset, MAGSTaRKDataset, STaRKDataset


def get_qa_dataset(name, root='/dfs/project/kgrlm/benchmark/'):
    qa_root = osp.join(root, name)
    if name == 'amazon':
        split_dir = osp.join(qa_root, 'split')
        hybrid_qa_dir = osp.join(qa_root, 'hybrid_qa')
        dataset = AmazonSTaRKDataset(hybrid_qa_dir, split_dir)
    if name == 'primekg':
        split_dir = osp.join(qa_root, 'split')
        hybrid_qa_dir = osp.join(qa_root, 'hybrid_qa')
        dataset = PrimeKGSTaRKDataset(hybrid_qa_dir, split_dir)
    if name == 'mag':
        split_dir = osp.join(qa_root, 'split')
        hybrid_qa_dir = osp.join(qa_root, 'hybrid_qa')
        dataset = MAGSTaRKDataset(hybrid_qa_dir, split_dir)
    else:
        try:
            print('loading HybridQA dataset from external data')
            split_dir = osp.join(qa_root, 'split')
            hybrid_qa_dir = osp.join(qa_root, 'hybrid_qa')
            dataset = STaRKDataset(hybrid_qa_dir, split_dir)
        except Exception as e:
            print('Please check dataset name, path, or format\n')
            raise e
    return dataset
    
    