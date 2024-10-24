from typing import List, Dict, Any
import torch
from torchmetrics.functional import (
    retrieval_hit_rate, retrieval_reciprocal_rank, retrieval_recall, 
    retrieval_precision, retrieval_average_precision, retrieval_normalized_dcg, 
    retrieval_r_precision
)

class Evaluator:
    
    def __init__(self, candidate_ids: List[int], device: str = 'cpu'):
        """
        Initializes the evaluator with the given candidate IDs.
        
        Args:
            candidate_ids (List[int]): List of candidate IDs.
        """
        self.candidate_ids = candidate_ids
        self.device = device

    def __call__(self, 
                 pred_dict: Dict[int, float], 
                 answer_ids: torch.LongTensor, 
                 metrics: List[str] = ['mrr', 'hit@3', 'recall@20']) -> Dict[str, float]:
        """
        Evaluates the predictions using the specified metrics.
        
        Args:
            pred_dict (Dict[int, float]): Dictionary of predicted scores.
            answer_ids (torch.LongTensor): Ground truth answer IDs.
            metrics (List[str]): List of metrics to be evaluated, including 'mrr', 'hit@k', 'recall@k', 
                                 'precision@k', 'map@k', 'ndcg@k'.
                             
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        return self.evaluate(pred_dict, answer_ids, metrics)
    
    def evaluate(self, 
                 pred_dict: Dict[int, float], 
                 answer_ids: torch.LongTensor, 
                 metrics: List[str] = ['mrr', 'hit@3', 'recall@20']) -> Dict[str, float]:
        """
        Evaluates the predictions using the specified metrics.
        
        Args:
            pred_dict (Dict[int, float]): Dictionary of predicted scores.
            answer_ids (torch.LongTensor): Ground truth answer IDs.
            metrics (List[str]): A list of metrics to be evaluated, including 'mrr', 'hit@k', 'recall@k', 
                                 'precision@k', 'map@k', 'ndcg@k'.
                             
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
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
    
    def evaluate_batch(self, 
                       pred_ids,
                       pred,
                       answer_ids: List[Any], 
                       metrics: List[str] = ['mrr', 'hit@3', 'recall@20']) -> Dict[str, float]:

        all_pred = torch.ones((max(self.candidate_ids) + 1, pred.shape[1]), dtype=torch.float) * (pred.min() - 1)
        all_pred[pred_ids, :] = pred
        all_pred = all_pred[self.candidate_ids].t().to(self.device)

        bool_gd = torch.zeros((max(self.candidate_ids) + 1, pred.shape[1]), dtype=torch.bool)
        bool_gd[torch.concat(answer_ids), torch.repeat_interleave(torch.arange(len(answer_ids)), torch.tensor(list(map(len, answer_ids))))] = True
        bool_gd = bool_gd[self.candidate_ids].t().to(self.device)

        results = []
        for i in range(len(answer_ids)):
            eval_metrics = {}
            for metric in metrics:
                k = int(metric.split('@')[-1]) if '@' in metric else None
                if metric == 'mrr':
                    result = retrieval_reciprocal_rank(all_pred[i], bool_gd[i])
                elif metric == 'rprecision':
                    result = retrieval_r_precision(all_pred[i], bool_gd[i])
                elif 'hit' in metric:
                    result = retrieval_hit_rate(all_pred[i], bool_gd[i], top_k=k)
                elif 'recall' in metric:
                    result = retrieval_recall(all_pred[i], bool_gd[i], top_k=k)
                elif 'precision' in metric:
                    result = retrieval_precision(all_pred[i], bool_gd[i], top_k=k)
                elif 'map' in metric:
                    result = retrieval_average_precision(all_pred[i], bool_gd[i], top_k=k)
                elif 'ndcg' in metric:
                    result = retrieval_normalized_dcg(all_pred[i], bool_gd[i], top_k=k)
                eval_metrics[metric] = float(result)
            results.append(eval_metrics)
        return results
