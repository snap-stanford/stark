import os
import os.path as osp
import torch
import json


class STaRKDataset:
    def __init__(self, query_dir, split_dir):
        self.query_dir = query_dir
        self.split_dir = split_dir
        
        file_names = [path.split('/')[-1].split('.')[0] for path in os.listdir(query_dir)]
        splitted_paths = [((splitted := path.split("_"))[0], splitted[-1]) 
                          for path in file_names]
        
        self.prefix = list(set([s[0] for s in splitted_paths]))[0]
        self.indices = list(set([int(s[1]) for s in splitted_paths]))
        self.indices.sort()
        self.split_indices = self.get_idx_split()
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        query_id = self.indices[idx]
        meta_info = json.load(
            open(osp.join(self.query_dir, f'{self.prefix}_{query_id}.json'), 'r')
            )
        query = meta_info.pop('query')
        answer_ids = meta_info.pop('answer_ids')
        
        return query, query_id, answer_ids, meta_info

    def get_query_by_qid(self, q_id):
        meta_info = json.load(
            open(osp.join(self.query_dir, f'{self.prefix}_{q_id}.json'), 'r')
            )
        query = meta_info.pop('query')
        return query, meta_info
        
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

    def get_subset(self, split):
        '''
        Return a subset of the dataset.
        '''
        assert split in ['train', 'val', 'test']
        indices_file = osp.join(self.split_dir, f'{split}.index') 
        indices = open(indices_file, 'r').read().strip().split('\n')
        self.indices = [int(idx) for idx in indices]
        return self
    