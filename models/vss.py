import os.path as osp
import torch
from typing import Any, Union, List, Dict
from models.model import ModelForSTaRKQA
from tqdm import tqdm
from stark_qa.evaluator import Evaluator

class VSS(ModelForSTaRKQA):
    
    def __init__(self, 
                 skb, 
                 query_emb_dir: str, 
                 candidates_emb_dir: str, 
                 emb_model: str = 'text-embedding-ada-002',
                 device: str = 'cuda'):
        """
        Vector Similarity Search

        Args:
            skb (SemiStruct): Knowledge base.
            query_emb_dir (str): Directory to query embeddings.
            candidates_emb_dir (str): Directory to candidate embeddings.
            emb_model (str): Embedding model name.
        """
        super(VSS, self).__init__(skb, query_emb_dir=query_emb_dir)
        self.emb_model = emb_model
        self.candidates_emb_dir = candidates_emb_dir
        self.device = device
        self.evaluator = Evaluator(self.candidate_ids, device)

        candidate_emb_path = osp.join(candidates_emb_dir, 'candidate_emb_dict.pt')
        if osp.exists(candidate_emb_path):
            candidate_emb_dict = torch.load(candidate_emb_path)
            print(f'Loaded candidate_emb_dict from {candidate_emb_path}!')
        else:
            # deprecated, load embeddings one by one
            print('Loading candidate embeddings...')
            candidate_emb_dict = {}
            for idx in tqdm(self.candidate_ids):
                candidate_emb_dict[idx] = torch.load(osp.join(candidates_emb_dir, f'{idx}.pt'))
            torch.save(candidate_emb_dict, candidate_emb_path)
            print(f'Saved candidate_emb_dict to {candidate_emb_path}!')

        assert len(candidate_emb_dict) == len(self.candidate_ids)
        candidate_embs = [candidate_emb_dict[idx].view(1, -1) for idx in self.candidate_ids]
        self.candidate_embs = torch.cat(candidate_embs, dim=0).to(device)
    
    def forward(self, 
                query: Union[str, List[str]], 
                query_id: Union[int, List[int]], 
                **kwargs: Any) -> dict:
        """
        Forward pass to compute similarity scores for the given query.

        Args:
            query (str): Query string.
            query_id (int): Query index.

        Returns:
            pred_dict (dict): A dictionary of candidate ids and their corresponding similarity scores.
        """
        query_emb = self.get_query_emb(query, query_id, emb_model=self.emb_model)
        similarity = torch.matmul(query_emb.to(self.device), self.candidate_embs.T).cpu()
        if isinstance(query, str):
            return dict(zip(self.candidate_ids, similarity.view(-1)))
        else:
            return torch.LongTensor(self.candidate_ids), similarity.t()
