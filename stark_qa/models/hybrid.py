"""
HybridRetriever: Combines BM25 (sparse) and VSS (dense) retrieval 
using Reciprocal Rank Fusion (RRF).

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms 
Condorcet and individual Rank Learning Methods" (SIGIR 2009)
"""

import torch
from typing import Any, Dict, List, Optional, Union

from stark_qa.models.base import ModelForSTaRKQA
from stark_qa.models.bm25 import BM25
from stark_qa.models.vss import VSS


class HybridRetriever(ModelForSTaRKQA):
    """
    Hybrid retrieval combining BM25 (sparse) and VSS (dense) using RRF.
    
    Args:
        skb: Semi-structured knowledge base.
        query_emb_dir: Directory for query embeddings.
        candidates_emb_dir: Directory for candidate embeddings.
        emb_model: Embedding model name for VSS.
        alpha: Weight for VSS (0-1). Default: 0.5.
        rrf_k: RRF constant. Default: 60.
        fusion_method: 'rrf' or 'weighted'. Default: 'rrf'.
        bm25_top_k: BM25 candidates to retrieve. Default: 100.
        device: Compute device. Default: 'cuda'.
    """
    
    def __init__(
        self,
        skb: Any,
        query_emb_dir: str,
        candidates_emb_dir: str,
        emb_model: str = 'text-embedding-ada-002',
        alpha: float = 0.5,
        rrf_k: int = 60,
        fusion_method: str = 'rrf',
        bm25_top_k: int = 100,
        device: str = 'cuda'
    ) -> None:
        super().__init__(skb, query_emb_dir=query_emb_dir)
        
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if fusion_method not in ['rrf', 'weighted']:
            raise ValueError(f"fusion_method must be 'rrf' or 'weighted'")
        
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.fusion_method = fusion_method
        self.bm25_top_k = bm25_top_k
        self.emb_model = emb_model
        
        self.bm25 = BM25(skb)
        self.vss = VSS(
            skb, 
            query_emb_dir=query_emb_dir,
            candidates_emb_dir=candidates_emb_dir,
            emb_model=emb_model,
            device=device
        )

    def forward(
        self,
        query: Union[str, List[str]],
        query_id: Optional[Union[int, List[int]]] = None,
        **kwargs: Any
    ) -> Dict[int, float]:
        """Compute hybrid scores combining BM25 and VSS."""
        if not isinstance(query, str):
            raise NotImplementedError("Batch queries not supported")
        
        bm25_scores = self.bm25.forward(query, query_id, k=self.bm25_top_k, **kwargs)
        vss_scores = self.vss.forward(query, query_id, **kwargs)
        
        if self.fusion_method == 'rrf':
            return self._rrf_fusion(bm25_scores, vss_scores)
        return self._weighted_fusion(bm25_scores, vss_scores)

    def _rrf_fusion(
        self, 
        bm25_scores: Dict[int, float], 
        vss_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """Combine using Reciprocal Rank Fusion."""
        bm25_ranks = self._scores_to_ranks(bm25_scores)
        vss_ranks = self._scores_to_ranks(vss_scores)
        all_candidates = set(bm25_scores.keys()) | set(vss_scores.keys())
        
        hybrid_scores = {}
        for cand_id in all_candidates:
            score = 0.0
            if cand_id in bm25_ranks:
                score += (1 - self.alpha) / (self.rrf_k + bm25_ranks[cand_id])
            if cand_id in vss_ranks:
                score += self.alpha / (self.rrf_k + vss_ranks[cand_id])
            hybrid_scores[cand_id] = score
        
        return hybrid_scores

    def _weighted_fusion(
        self, 
        bm25_scores: Dict[int, float], 
        vss_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """Combine using weighted linear combination."""
        norm_bm25 = self._normalize_scores(bm25_scores)
        norm_vss = self._normalize_scores(vss_scores)
        all_candidates = set(bm25_scores.keys()) | set(vss_scores.keys())
        
        return {
            cand_id: (1 - self.alpha) * norm_bm25.get(cand_id, 0.0) 
                     + self.alpha * norm_vss.get(cand_id, 0.0)
            for cand_id in all_candidates
        }

    def _scores_to_ranks(self, scores: Dict[int, float]) -> Dict[int, int]:
        """Convert scores to ranks (1-indexed)."""
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return {cand_id: rank + 1 for rank, cand_id in enumerate(sorted_ids)}

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return {}
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 0.5 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
