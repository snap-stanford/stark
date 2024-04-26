import copy
import os.path as osp
import torch
import pandas as pd


class STaRKDataset:
    def __init__(self, query_dir, split_dir):
        self.query_dir = query_dir
        self.split_dir = split_dir
        self.qa_csv_path = osp.join(query_dir, 'stark_qa.csv')
        self.data = pd.read_csv(self.qa_csv_path)

        self.indices = list(self.data['id'])
        self.indices.sort()
        self.split_indices = self.get_idx_split()
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        q_id = self.indices[idx]
        meta_info = None
        row = self.data[self.data['id'] == q_id].iloc[0]
        query = row['query']
        answer_ids = eval(row['answer_ids'])
        
        return query, q_id, answer_ids, meta_info

    def get_idx_split(self, test_ratio=1.0):
        '''
        Return the indices of train/val/test split in a dictionary.
        '''
        split_idx = {}
        for split in ['train', 'val', 'test']:
            # `{split}.index`stores query ids, not the index in the dataset
            indices_file = osp.join(self.split_dir, f'{split}.index') 
            indices = open(indices_file, 'r').read().strip().split('\n')
            query_ids = [int(idx) for idx in indices]
            split_idx[split] = torch.LongTensor([self.indices.index(query_id) for query_id in query_ids])
        if test_ratio < 1.0:
            split_idx['test'] = split_idx['test'][:int(len(split_idx['test']) * test_ratio)]
        return split_idx

    def get_query_by_qid(self, q_id):
        '''
        Return the query by query id.
        '''
        row = self.data[self.data['id'] == q_id].iloc[0]
        return row['query']
        
    def get_subset(self, split):
        '''
        Return a subset of the dataset.
        '''
        assert split in ['train', 'val', 'test']
        indices_file = osp.join(self.split_dir, f'{split}.index') 
        indices = open(indices_file, 'r').read().strip().split('\n')
        subset = copy.deepcopy(self)
        subset.indices = [int(idx) for idx in indices]
        return subset
    