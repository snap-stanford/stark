"""
Unit tests for GraphRetriever model.
"""

import torch
import pytest
from unittest.mock import Mock, MagicMock

from stark_qa.models.graph_retriever import GraphRetriever


class MockSKB:
    """Mock knowledge base for testing."""
    
    def __init__(self, num_nodes=10, num_candidates=5):
        self.node_info = {i: {} for i in range(num_nodes)}
        # Create a simple chain graph: 0-1-2-3-4
        edge_list = [(i, i+1) for i in range(num_nodes-1)]
        edge_list += [(i+1, i) for i in range(num_nodes-1)]  # Make undirected
        self.edge_index = torch.tensor(edge_list).t().contiguous()
        self.node_types = torch.zeros(num_nodes, dtype=torch.long)
        self.edge_types = torch.zeros(len(edge_list), dtype=torch.long)
        self.edge_type_dict = {0: 'related'}
        self.node_type_dict = {0: 'node'}
        self.candidate_ids = list(range(num_candidates))
        self.num_candidates = num_candidates
    
    def num_nodes(self):
        return len(self.node_info)
    
    def num_edges(self):
        return self.edge_index.shape[1]
    
    def get_doc_info(self, idx, add_rel=False, compact=False):
        return f"Document {idx}"


class MockVSS:
    """Mock VSS model for testing."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def forward(self, query, query_id=None, **kwargs):
        # Return mock scores: higher scores for lower IDs
        return {i: 1.0 / (i + 1) for i in range(5)}


def test_graph_retriever_initialization():
    """Test GraphRetriever initialization."""
    skb = MockSKB()
    
    retriever = GraphRetriever(
        skb=skb,
        query_emb_dir='test_query_dir',
        candidates_emb_dir='test_cand_dir',
        emb_model='test-model',
        graph_weight=0.3,
        propagation_hops=2,
        propagation_decay=0.5,
        top_k_initial=200,
        device='cpu'
    )
    
    assert retriever.graph_weight == 0.3
    assert retriever.propagation_hops == 2
    assert retriever.propagation_decay == 0.5
    assert retriever.top_k_initial == 200


def test_graph_retriever_invalid_params():
    """Test GraphRetriever with invalid parameters."""
    skb = MockSKB()
    
    # Test invalid graph_weight
    with pytest.raises(ValueError, match="graph_weight must be in"):
        GraphRetriever(
            skb=skb,
            query_emb_dir='test',
            candidates_emb_dir='test',
            graph_weight=1.5
        )
    
    # Test invalid propagation_hops
    with pytest.raises(ValueError, match="propagation_hops must be"):
        GraphRetriever(
            skb=skb,
            query_emb_dir='test',
            candidates_emb_dir='test',
            propagation_hops=0
        )
    
    # Test invalid propagation_decay
    with pytest.raises(ValueError, match="propagation_decay must be"):
        GraphRetriever(
            skb=skb,
            query_emb_dir='test',
            candidates_emb_dir='test',
            propagation_decay=0.0
        )


def test_normalize_scores():
    """Test score normalization."""
    skb = MockSKB()
    retriever = GraphRetriever(
        skb=skb,
        query_emb_dir='test',
        candidates_emb_dir='test',
        device='cpu'
    )
    
    # Test normal case
    scores = {0: 1.0, 1: 2.0, 2: 3.0}
    normalized = retriever._normalize_scores(scores)
    assert normalized[0] == 0.0
    assert normalized[2] == 1.0
    assert 0.0 <= normalized[1] <= 1.0
    
    # Test equal scores
    scores = {0: 1.0, 1: 1.0, 2: 1.0}
    normalized = retriever._normalize_scores(scores)
    assert all(v == 0.5 for v in normalized.values())
    
    # Test empty scores
    normalized = retriever._normalize_scores({})
    assert normalized == {}


def test_combine_scores():
    """Test score combination."""
    skb = MockSKB()
    retriever = GraphRetriever(
        skb=skb,
        query_emb_dir='test',
        candidates_emb_dir='test',
        graph_weight=0.3,
        device='cpu'
    )
    
    semantic_scores = {0: 1.0, 1: 0.5, 2: 0.0}
    graph_scores = {0: 0.0, 1: 0.5, 2: 1.0}
    
    combined = retriever._combine_scores(semantic_scores, graph_scores)
    
    # Check that all candidates are in the result
    assert set(combined.keys()) == {0, 1, 2}
    
    # Check that scores are in valid range
    assert all(0.0 <= v <= 1.0 for v in combined.values())


if __name__ == '__main__':
    pytest.main([__file__])

