import copy
import os.path as osp

import pandas as pd
from typing import Union
import torch

from stark_qa.tools.download_hf import download_hf_folder


STARK_QA_DATASET = {
    "repo": "snap-stanford/stark",
    "folder": "qa"
}

class STaRKDataset:
    def __init__(self, 
                 name: str, 
                 root: Union[str, None] = None, 
                 human_generated_eval: bool = False):
        """
        Initialize the STaRK dataset.

        Args:
            name (str): Name of the dataset.
            root (Union[str, None]): Root directory to store the dataset. If None, default HF cache paths will be used.
            human_generated_eval (bool): Whether to use human-generated evaluation data.
        """
        self.name = name
        self.root = root
        self.dataset_root = osp.join(self.root, name) if self.root is not None else None
        self._download()
        self.split_dir = osp.join(self.dataset_root, 'split')
        self.query_dir = osp.join(self.dataset_root, 'stark_qa')
        self.human_generated_eval = human_generated_eval

        self.qa_csv_path = osp.join(
            self.query_dir, 
            'stark_qa_human_generated_eval.csv' if human_generated_eval else 'stark_qa.csv'
        )
        
        self.data = pd.read_csv(self.qa_csv_path)
        self.indices = sorted(self.data['id'].tolist())
        self.split_indices = self.get_idx_split()

    def __len__(self) -> int:
        """
        Return the number of queries in the dataset.

        Returns:
            int: Number of queries.
        """
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Get the query, id, answer ids, and meta information for a given index.

        Args:
            idx (int): Index of the query.

        Returns:
            tuple: Query, query id, answer ids, and meta information.
        """
        q_id = self.indices[idx]
        row = self.data[self.data['id'] == q_id].iloc[0]
        query = row['query']
        answer_ids = eval(row['answer_ids'])
        meta_info = None  # Replace with actual meta information if available
        return query, q_id, answer_ids, meta_info

    def _download(self):
        """
        Download the dataset from the Hugging Face repository.
        """
        self.dataset_root = download_hf_folder(
            STARK_QA_DATASET["repo"],
            osp.join(STARK_QA_DATASET["folder"], self.name),
            repo_type="dataset",
            save_as_folder=self.dataset_root,
        )

    def get_idx_split(self, test_ratio: float = 1.0) -> dict:
        """
        Return the indices of train/val/test split in a dictionary.

        Args:
            test_ratio (float): Ratio of test data to include.

        Returns:
            dict: Dictionary with split indices for train, val, and test sets.
        """
        if self.human_generated_eval:
            return {'human_generated_eval': torch.LongTensor(self.indices)}

        split_idx = {}
        for split in ['train', 'val', 'test', 'test-0.1']:
            indices_file = osp.join(self.split_dir, f'{split}.index')
            with open(indices_file, 'r') as f:
                indices = f.read().strip().split('\n')
            query_ids = [int(idx) for idx in indices]
            split_idx[split] = torch.LongTensor([self.indices.index(query_id) for query_id in query_ids])

        if test_ratio < 1.0:
            split_idx['test'] = split_idx['test'][:int(len(split_idx['test']) * test_ratio)]
        return split_idx

    def get_query_by_qid(self, q_id: int) -> str:
        """
        Return the query by query id.

        Args:
            q_id (int): Query id.

        Returns:
            str: Query string.
        """
        row = self.data[self.data['id'] == q_id].iloc[0]
        return row['query']

    def get_subset(self, split: str):
        """
        Return a subset of the dataset.

        Args:
            split (str): Split type ('train', 'val', 'test', 'test-0.1').

        Returns:
            STaRKDataset: Subset of the dataset.
        """
        assert split in ['train', 'val', 'test', 'test-0.1'], "Invalid split specified."
        indices_file = osp.join(self.split_dir, f'{split}.index')
        with open(indices_file, 'r') as f:
            indices = f.read().strip().split('\n')
        subset = copy.deepcopy(self)
        subset.indices = [int(idx) for idx in indices]
        return subset
