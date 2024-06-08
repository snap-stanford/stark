import os
import os.path as osp
from typing import Any, Union, List, Dict

import torch
import torch.nn as nn
from stark_qa.tools.api import get_openai_embedding
from stark_qa.evaluator import Evaluator


class ModelForSTaRKQA(nn.Module):
    
    def __init__(self, skb):
        """
        Initializes the model with the given knowledge base.
        
        Args:
            skb: Knowledge base containing candidate information.
        """
        super(ModelForSTaRKQA, self).__init__()
        self.skb = skb

        self.candidate_ids = skb.candidate_ids
        self.num_candidates = skb.num_candidates
        self.query_emb_dict = {}
        self.evaluator = Evaluator(self.candidate_ids)
    
    def forward(self, 
                query: Union[str, List[str]], 
                candidates: List[int] = None,
                query_id: Union[int, List[int]] = None,
                **kwargs: Any) -> Dict[str, Any]:
        """
        Forward pass to compute predictions for the given query.
        
        Args:
            query (Union[str, list]): Query string or a list of query strings.
            candidates (Union[list, None]): A list of candidate ids (optional).
            query_id (Union[int, list, None]): Query index (optional).
            
        Returns:
            pred_dict (dict): A dictionary of predicted scores or answer ids.
        """
        raise NotImplementedError
    
    def get_query_emb(self, 
                       query: str, 
                       query_id: int, 
                       emb_model: str = 'text-embedding-ada-002') -> torch.Tensor:
        """
        Retrieves or computes the embedding for the given query.
        
        Args:
            query (str): Query string.
            query_id (int): Query index.
            emb_model (str): Embedding model to use.
            
        Returns:
            query_emb (torch.Tensor): Query embedding.
        """
        if query_id is None:
            query_emb = get_openai_embedding(query, model=emb_model)
        elif len(self.query_emb_dict) > 0:
            query_emb = self.query_emb_dict[query_id]
        else:
            query_emb_dic_path = osp.join(self.query_emb_dir, 'query_emb_dict.pt')
            if os.path.exists(query_emb_dic_path):
                print(f'Load query embeddings from {query_emb_dic_path}')
                self.query_emb_dict = torch.load(query_emb_dic_path)
                query_emb = self.query_emb_dict[query_id]
            else:
                query_emb_dir = osp.join(self.query_emb_dir, 'query_embs')
                if not os.path.exists(query_emb_dir):
                    os.makedirs(query_emb_dir)
                query_emb_path = osp.join(query_emb_dir, f'query_{query_id}.pt')
                query_emb = get_openai_embedding(query, model=emb_model)
                torch.save(query_emb, query_emb_path)
        return query_emb
    
    def evaluate(self, 
                 pred_dict: Dict[int, float], 
                 answer_ids: torch.LongTensor, 
                 metrics: List[str] = ['mrr', 'hit@3', 'recall@20'], 
                 **kwargs: Any) -> Dict[str, float]:
        """
        Evaluates the predictions using the specified metrics.
        
        Args:
            pred_dict (Dict[int, float]): Predicted answer ids or scores.
            answer_ids (torch.LongTensor): Ground truth answer ids.
            metrics (List[str]): A list of metrics to be evaluated, including 'mrr', 'hit@k', 'recall@k', 
                                 'precision@k', 'map@k', 'ndcg@k'.
                             
        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        return self.evaluator(pred_dict, answer_ids, metrics)
