import os
import os.path as osp
from typing import Any, Union, List, Dict, Optional

import torch
import torch.nn as nn
from stark_qa.evaluator import Evaluator
from stark_qa.tools.llm_lib.get_llm_embeddings import get_llm_embeddings


class ModelForSTaRKQA(nn.Module):
    def __init__(self, skb: Any, query_emb_dir: str = '.') -> None:
        """
        Initialize the model with the given knowledge base and query embedding directory.

        Args:
            skb (Any): The knowledge base containing candidate information.
            query_emb_dir (str, optional): Directory where query embeddings are stored. Defaults to '.'.
        """
        super(ModelForSTaRKQA, self).__init__()
        self.skb = skb

        self.candidate_ids: List[int] = skb.candidate_ids
        self.num_candidates: int = skb.num_candidates
        self.query_emb_dir: str = query_emb_dir

        # Load query embeddings if they exist
        query_emb_path = osp.join(self.query_emb_dir, 'query_emb_dict.pt')
        if os.path.exists(query_emb_path):
            print(f'Loading query embeddings from {query_emb_path}')
            self.query_emb_dict: Dict[int, torch.Tensor] = torch.load(query_emb_path)
        else:
            self.query_emb_dict = {}

        # Initialize the evaluator with candidate IDs
        self.evaluator = Evaluator(self.candidate_ids)

    def forward(
        self,
        query: Union[str, List[str]],
        candidates: Optional[List[int]] = None,
        query_id: Optional[Union[int, List[int]]] = None,
        **kwargs: Any
    ) -> Dict[int, float]:
        """
        Compute predictions for the given query.

        Args:
            query (Union[str, List[str]]): The input query or list of queries.
            candidates (Optional[List[int]], optional): List of candidate IDs to consider. Defaults to None.
            query_id (Optional[Union[int, List[int]]], optional): Query ID or list of query IDs. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[int, float]: A dictionary mapping candidate IDs to predicted scores.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_query_emb(
        self,
        query: Union[str, List[str]],
        query_id: Optional[Union[int, List[int]]] = None,
        emb_model: str = 'text-embedding-ada-002',
        **encode_kwargs: Any
    ) -> torch.Tensor:
        """
        Retrieve or compute the embeddings for the given queries.

        Args:
            query (Union[str, List[str]]): Query string or a list of query strings.
            query_id (Optional[Union[int, List[int]]], optional): Query ID or list of query IDs. Defaults to None.
            emb_model (str, optional): Embedding model to use. Defaults to 'text-embedding-ada-002'.
            **encode_kwargs: Additional keyword arguments for the embedding function.

        Returns:
            torch.Tensor: Query embeddings with shape (num_queries, embedding_dim).
        """
        if isinstance(query_id, int):
            query_id = [query_id]
        if isinstance(query, str):
            query = [query]

        if query_id is None:
            # No query IDs provided; compute embeddings directly
            query_emb = get_llm_embeddings(query, emb_model, **encode_kwargs)
        else:
            # Check for missing embeddings in the cache
            missing_ids = [qid for qid in query_id if qid not in self.query_emb_dict]
            if missing_ids:
                # Compute embeddings for missing query IDs
                missing_queries = [q for q, qid in zip(query, query_id) if qid in missing_ids]
                new_embs = get_llm_embeddings(missing_queries, emb_model, **encode_kwargs)
                for qid, emb in zip(missing_ids, new_embs):
                    self.query_emb_dict[qid] = emb.view(1, -1)
                # Save updated embeddings to the cache
                torch.save(self.query_emb_dict, osp.join(self.query_emb_dir, 'query_emb_dict.pt'))

            # Retrieve embeddings in the order of query IDs
            query_emb_list = [self.query_emb_dict[qid] for qid in query_id]
            query_emb = torch.cat(query_emb_list, dim=0)

        query_emb = query_emb.view(len(query), -1)
        return query_emb

    def evaluate(
        self,
        pred_dict: Dict[int, float],
        answer_ids: Union[torch.LongTensor, List[int]],
        metrics: List[str] = ['mrr', 'hit@3', 'recall@20'],
        **kwargs: Any
    ) -> Dict[str, float]:
        """
        Evaluate the predictions using the specified metrics.

        Args:
            pred_dict (Dict[int, float]): Dictionary mapping candidate IDs to predicted scores.
            answer_ids (Union[torch.LongTensor, List[int]]): Ground truth answer IDs.
            metrics (List[str], optional): List of metrics to compute. Defaults to ['mrr', 'hit@3', 'recall@20'].
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, float]: A dictionary containing the computed metrics.
        """
        return self.evaluator(pred_dict, answer_ids, metrics)

    def evaluate_batch(
        self,
        pred_ids: List[int],
        pred: torch.Tensor,
        answer_ids: Union[torch.LongTensor, List[int]],
        metrics: List[str] = ['mrr', 'hit@3', 'recall@20'],
        **kwargs: Any
    ) -> Dict[str, float]:
        """
        Evaluate batch predictions using the specified metrics.

        Args:
            pred_ids (List[int]): List of predicted candidate IDs.
            pred (torch.Tensor): Tensor of predicted scores corresponding to `pred_ids`.
            answer_ids (Union[torch.LongTensor, List[int]]): Ground truth answer IDs.
            metrics (List[str], optional): List of metrics to compute. Defaults to ['mrr', 'hit@3', 'recall@20'].
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, float]: A dictionary containing the computed metrics.
        """
        return self.evaluator.evaluate_batch(pred_ids, pred, answer_ids, metrics)
