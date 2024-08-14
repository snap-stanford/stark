import os
import os.path as osp
from typing import Any, Union, List, Dict

import torch
import torch.nn as nn
from stark_qa.tools.api import get_api_embedding, get_api_embeddings
from stark_qa.evaluator import Evaluator


class ModelForSTaRKQA(nn.Module):
    
    def __init__(self, skb, query_emb_dir='.'):
        """
        Initializes the model with the given knowledge base.
        
        Args:
            skb: Knowledge base containing candidate information.
        """
        super(ModelForSTaRKQA, self).__init__()
        self.skb = skb

        self.candidate_ids = skb.candidate_ids
        self.num_candidates = skb.num_candidates
        self.query_emb_dir = query_emb_dir

        query_emb_path = osp.join(self.query_emb_dir, 'query_emb_dict.pt')
        if os.path.exists(query_emb_path):
            print(f'Load query embeddings from {query_emb_path}')
            self.query_emb_dict = torch.load(query_emb_path)
        else:
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
                       query: Union[str, List[str]], 
                       query_id: Union[int, List[int]], 
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
            if isinstance(query, int):
                query_emb = get_api_embedding(emb_model, text=query)
            else:
                query_emb = torch.concat([get_api_embeddings(emb_model, text=q) for q in query], dim=0)
        
        if len(self.query_emb_dict) > 0:
            if isinstance(query_id, int):
                query_emb = self.query_emb_dict[query_id]
            else:
                query_emb = torch.concat([self.query_emb_dict[qid] for qid in query_id], dim=0)
        else:
            query_emb = get_api_embeddings(emb_model, texts=query)
            if isinstance(query_id, int):
                query_id = [query_id]
            for qid, emb in zip(query_id, query_emb):
                self.query_emb_dict[qid] = emb.view(1, -1)
            torch.save(self.query_emb_dict, osp.join(self.query_emb_dir, 'query_emb_dict.pt'))
            
        if isinstance(query, str):
            query_emb = query_emb.view(1, -1)
        else:
            query_emb = query_emb.view(len(query), -1)
        return query_emb
    
    def evaluate(self, 
                 pred_dict: Dict[int, float], 
                 answer_ids: Union[torch.LongTensor, List[Any]], 
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
    
    def evaluate_batch(self, 
                pred_ids: List[int],
                pred: torch.Tensor, 
                answer_ids: Union[torch.LongTensor, List[Any]], 
                metrics: List[str] = ['mrr', 'hit@3', 'recall@20'], 
                **kwargs: Any) -> Dict[str, float]:
        return self.evaluator.evaluate_batch(pred_ids, pred, answer_ids, metrics)
