import os.path as osp
import json
import torch
from src.benchmarks.qa_datasets.hybrid_qa import STaRKDataset


class PrimeKGSTaRKDataset(STaRKDataset):
    
    def __init__(self, hybrid_qa_dir, split_dir):
        super().__init__(hybrid_qa_dir, split_dir)