import torch.nn as nn
from typing import Any, Union
import torch
import os
import os.path as osp
from torchmetrics.functional.retrieval import retrieval_hit_rate, \
                                              retrieval_reciprocal_rank, \
                                              retrieval_recall, retrieval_precision, \
                                              retrieval_average_precision, \
                                              retrieval_normalized_dcg, \
                                              retrieval_r_precision
from src.tools.api import get_openai_embedding


class ModelForSemiStructQA(nn.Module):
    
    def __init__(self, kb):
        super(ModelForSemiStructQA, self).__init__()
        self.kb = kb
        self.candidate_ids = kb.candidate_ids
        self.num_candidates = kb.num_candidates
        self.query_emb_dict = {}
    
    def forward(self, 
                query: Union[str, list], 
                candidates=None,
                query_id=None,
                **kwargs: Any):
        '''
        Args:
            query (Union[str, list]): query string or a list of query strings
            candidates (Union[list, None]): a list of candidate ids (optional)
            query_id (Union[int, list, None]): query index (optional)
            
        Returns:
            pred_dict (dict): a dictionary of predicted scores or answer ids
        '''
        raise NotImplementedError
    
    def _get_query_emb(self, query: str, query_id: int, 
                       emb_model: str = 'text-embedding-ada-002'):
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
                 pred_dict: dict, 
                 answer_ids: torch.LongTensor, 
                 metrics=['mrr', 'hit@3', 'recall@20'], 
                 **kwargs: Any):
        '''
        Args:
            pred_dict (torch.Tensor): predicted answer ids or scores
            answer_ids (torch.LongTensor): ground truth answer ids
            metrics (list): a list of metrics to be evaluated, 
                including 'mrr', 'hit@k', 'recall@k', 'precision@k', 'map@k', 'ndcg@k'
        Returns:
            eval_metrics (dict): a dictionary of evaluation metrics
        '''
        pred_ids = torch.LongTensor(list(pred_dict.keys())).view(-1)
        pred = torch.FloatTensor(list(pred_dict.values())).view(-1)
        answer_ids = answer_ids.view(-1)

        all_pred = torch.ones(max(self.candidate_ids) + 1, dtype=torch.float) * min(pred) - 1
        all_pred[pred_ids] = pred
        all_pred = all_pred[self.candidate_ids]

        bool_gd = torch.zeros(max(self.candidate_ids) + 1, dtype=torch.bool)
        bool_gd[answer_ids] = True
        bool_gd = bool_gd[self.candidate_ids]

        eval_metrics = {}
        for metric in metrics:
            k = int(metric.split('@')[-1]) if '@' in metric else None
            if 'mrr' == metric:
                result = retrieval_reciprocal_rank(all_pred, bool_gd)
            elif 'rprecision' == metric:
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