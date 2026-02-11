"""
GraphRetriever: Graph-Aware Retrieval for Semi-Structured Knowledge Bases

This model enhances retrieval by leveraging the relational structure of the knowledge base.
It combines semantic similarity (VSS) with graph-based proximity scoring, where nodes
connected to highly-relevant nodes receive boosted scores.

Reference: Graph-based retrieval is a well-established approach for semi-structured data,
leveraging both content similarity and structural relationships.

Algorithm:
1. Initial retrieval using VSS to get semantic similarity scores
2. Graph propagation: Boost scores of nodes connected to high-scoring nodes
3. Final scoring: Combine semantic and graph-based scores


"""

import torch
from typing import Any, Dict, List, Optional, Union

from stark_qa.models.base import ModelForSTaRKQA
from stark_qa.models.vss import VSS


class GraphRetriever(ModelForSTaRKQA):
    """
    Graph-aware retrieval combining semantic similarity with graph structure.
    
    Args:
        skb: Semi-structured knowledge base with graph structure.
        query_emb_dir: Directory for query embeddings.
        candidates_emb_dir: Directory for candidate embeddings.
        emb_model: Embedding model name for VSS.
        graph_weight: Weight for graph-based scoring (0-1). Default: 0.3.
        propagation_hops: Number of hops for graph propagation. Default: 2.
        propagation_decay: Decay factor for each hop. Default: 0.5.
        top_k_initial: Top-K candidates from VSS for graph propagation. Default: 200.
        device: Compute device. Default: 'cuda'.
    """
    
    def __init__(
        self,
        skb: Any,
        query_emb_dir: str,
        candidates_emb_dir: str,
        emb_model: str = 'text-embedding-ada-002',
        graph_weight: float = 0.3,
        propagation_hops: int = 2,
        propagation_decay: float = 0.5,
        top_k_initial: int = 200,
        device: str = 'cuda'
    ) -> None:
        super().__init__(skb, query_emb_dir=query_emb_dir)
        
        if not 0 <= graph_weight <= 1:
            raise ValueError(f"graph_weight must be in [0, 1], got {graph_weight}")
        if propagation_hops < 1:
            raise ValueError(f"propagation_hops must be >= 1, got {propagation_hops}")
        if not 0 < propagation_decay <= 1:
            raise ValueError(f"propagation_decay must be in (0, 1], got {propagation_decay}")
        
        self.graph_weight = graph_weight
        self.propagation_hops = propagation_hops
        self.propagation_decay = propagation_decay
        self.top_k_initial = top_k_initial
        self.emb_model = emb_model
        self.device = device
        
        # Initialize VSS for semantic similarity
        self.vss = VSS(
            skb,
            query_emb_dir=query_emb_dir,
            candidates_emb_dir=candidates_emb_dir,
            emb_model=emb_model,
            device=device
        )
        
        # Build graph adjacency for candidate nodes only
        self._build_candidate_graph()

    def _build_candidate_graph(self):
        """Build adjacency structure for candidate nodes."""
        candidate_set = set(self.candidate_ids)
        num_nodes = self.skb.num_nodes()
        
        # Filter edges to only include candidate nodes
        edge_index = self.skb.edge_index
        row, col = edge_index
        
        # Keep edges where both nodes are candidates
        candidate_mask = torch.zeros(num_nodes, dtype=torch.bool)
        candidate_mask[torch.tensor(self.candidate_ids)] = True
        
        edge_mask = candidate_mask[row] & candidate_mask[col]
        candidate_edges = edge_index[:, edge_mask]
        
        # Map node IDs to candidate indices
        candidate_id_to_idx = {cand_id: idx for idx, cand_id in enumerate(self.candidate_ids)}
        
        # Remap edge indices to candidate indices
        if candidate_edges.numel() > 0:
            row_mapped = torch.tensor([candidate_id_to_idx[int(nid)] for nid in candidate_edges[0]])
            col_mapped = torch.tensor([candidate_id_to_idx[int(nid)] for nid in candidate_edges[1]])
            self.candidate_edge_index = torch.stack([row_mapped, col_mapped])
        else:
            # Empty graph
            self.candidate_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Build sparse adjacency matrix for candidates
        num_candidates = len(self.candidate_ids)
        if self.candidate_edge_index.numel() > 0:
            edge_weights = torch.ones(self.candidate_edge_index.shape[1])
            self.candidate_adj = torch.sparse.FloatTensor(
                self.candidate_edge_index,
                edge_weights,
                torch.Size([num_candidates, num_candidates])
            ).coalesce()
        else:
            # Empty sparse matrix
            indices = torch.zeros((2, 0), dtype=torch.long)
            values = torch.zeros(0)
            self.candidate_adj = torch.sparse.FloatTensor(
                indices, values, torch.Size([num_candidates, num_candidates])
            )

    def forward(
        self,
        query: Union[str, List[str]],
        query_id: Optional[Union[int, List[int]]] = None,
        **kwargs: Any
    ) -> Dict[int, float]:
        """Compute graph-aware retrieval scores."""
        if not isinstance(query, str):
            raise NotImplementedError("Batch queries not supported")
        
        # Step 1: Get initial semantic similarity scores from VSS
        vss_scores = self.vss.forward(query, query_id, **kwargs)
        
        # Step 2: Graph-based score propagation
        graph_scores = self._graph_propagation(vss_scores)
        
        # Step 3: Combine semantic and graph scores
        final_scores = self._combine_scores(vss_scores, graph_scores)
        
        return final_scores

    def _graph_propagation(self, initial_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Propagate scores through the graph structure.
        
        Uses iterative propagation where scores from high-scoring nodes
        boost the scores of their neighbors.
        """
        num_candidates = len(self.candidate_ids)
        if num_candidates == 0:
            return {}
        
        # Initialize score vector
        scores = torch.zeros(num_candidates, dtype=torch.float32)
        for cand_id, score in initial_scores.items():
            if cand_id in self.candidate_ids:
                idx = self.candidate_ids.index(cand_id)
                scores[idx] = score
        
        # Normalize initial scores to [0, 1] for stable propagation
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Iterative propagation
        current_scores = scores.clone()
        propagated_scores = torch.zeros_like(scores)
        
        for hop in range(self.propagation_hops):
            # Propagate through one hop
            if self.candidate_edge_index.shape[1] > 0:
                # Get neighbor scores
                neighbor_scores = torch.sparse.mm(self.candidate_adj, current_scores.unsqueeze(1)).squeeze(1)
                
                # Normalize by degree to avoid bias toward high-degree nodes
                degrees = torch.sparse.sum(self.candidate_adj, dim=1).to_dense()
                degrees = torch.clamp(degrees, min=1.0)  # Avoid division by zero
                neighbor_scores = neighbor_scores / degrees
                
                # Combine with current scores (weighted average)
                decay = self.propagation_decay ** hop
                propagated_scores += decay * neighbor_scores
                
                # Update current scores for next iteration
                current_scores = neighbor_scores
        
        # Convert back to dictionary
        graph_scores = {}
        for idx, cand_id in enumerate(self.candidate_ids):
            graph_scores[cand_id] = float(propagated_scores[idx])
        
        return graph_scores

    def _combine_scores(
        self,
        semantic_scores: Dict[int, float],
        graph_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """Combine semantic and graph-based scores."""
        all_candidates = set(semantic_scores.keys()) | set(graph_scores.keys())
        
        # Normalize both score distributions
        norm_semantic = self._normalize_scores(semantic_scores)
        norm_graph = self._normalize_scores(graph_scores)
        
        # Weighted combination
        combined = {}
        for cand_id in all_candidates:
            sem_score = norm_semantic.get(cand_id, 0.0)
            graph_score = norm_graph.get(cand_id, 0.0)
            combined[cand_id] = (1 - self.graph_weight) * sem_score + self.graph_weight * graph_score
        
        return combined

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return {}
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 0.5 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

