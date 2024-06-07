import os
import os.path as osp
from typing import Any, Union, List, Dict

import torch
import torch.nn as nn
from torchmetrics.functional import (
    retrieval_hit_rate, retrieval_reciprocal_rank, retrieval_recall, 
    retrieval_precision, retrieval_average_precision, retrieval_normalized_dcg, 
    retrieval_r_precision
)
from src.tools.api import get_openai_embedding


class ModelForSemiStructQA(nn.Module):
    
    def __init__(self, kb):
        """
        Initializes the model with the given knowledge base.
        
        Args:
            kb: Knowledge base containing candidate information.
        """
        super(ModelForSemiStructQA, self).__init__()
        self.kb = kb
        self.candidate_ids = kb.candidate_ids
        self.num_candidates = kb.num_candidates
        self.query_emb_dict = {}
    
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
    
    def _get_query_emb(self, query: str, query_id: int, 
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
            pred_dict (dict): Predicted answer ids or scores.
            answer_ids (torch.LongTensor): Ground truth answer ids.
            metrics (list): A list of metrics to be evaluated, including 'mrr', 'hit@k', 'recall@k', 
                            'precision@k', 'map@k', 'ndcg@k'.
                            
        Returns:
            eval_metrics (dict): A dictionary of evaluation metrics.
        """
        # Convert prediction dictionary to tensor
        pred_ids = torch.LongTensor(list(pred_dict.keys())).view(-1)
        pred = torch.FloatTensor(list(pred_dict.values())).view(-1)
        answer_ids = answer_ids.view(-1)

        # Initialize all predictions to a very low value
        all_pred = torch.ones(max(self.candidate_ids) + 1, dtype=torch.float) * (min(pred) - 1)
        all_pred[pred_ids] = pred
        all_pred = all_pred[self.candidate_ids]

        # Initialize ground truth boolean tensor
        bool_gd = torch.zeros(max(self.candidate_ids) + 1, dtype=torch.bool)
        bool_gd[answer_ids] = True
        bool_gd = bool_gd[self.candidate_ids]

        # Compute evaluation metrics
        eval_metrics = {}
        for metric in metrics:
            k = int(metric.split('@')[-1]) if '@' in metric else None
            if metric == 'mrr':
                result = retrieval_reciprocal_rank(all_pred, bool_gd)
            elif metric == 'rprecision':
                result = retrieval_r_precision(all_pred, bool_gd)
            elif 'hit' in metric:
                result = retrieval_hit_rate(all_pred, bool_gd, top_k=k)
            elif 'recall' in metric:
                result = retrieval_recall(all_pred, bool_gd, top_k=k)
            elif 'precision' in metric:
                result = retrieval_precision(all_pred, bool_gd, top_k=k)
            elif 'map' in metric:
                result = retrieval_average_precision(all_pred, bool_gd, top_k=k)
            elif 'ndcg' in metric:
                result = retrieval_normalized_dcg(all_pred, bool_gd, top_k=k)
            eval_metrics[metric] = float(result)

        return eval_metrics
