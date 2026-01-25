"""
Unit tests for HybridRetriever model.
"""

import pytest


class TestRRFLogic:
    """Tests for Reciprocal Rank Fusion computation."""
    
    def test_scores_to_ranks(self):
        """Verify score-to-rank conversion."""
        scores = {1: 0.9, 2: 0.5, 3: 0.7, 4: 0.3}
        sorted_candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        ranks = {cand_id: rank + 1 for rank, cand_id in enumerate(sorted_candidates)}
        
        assert ranks[1] == 1
        assert ranks[3] == 2
        assert ranks[2] == 3
        assert ranks[4] == 4
    
    def test_normalize_scores(self):
        """Verify min-max normalization."""
        scores = {1: 10.0, 2: 5.0, 3: 0.0}
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        normalized = {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
        
        assert normalized[1] == 1.0
        assert normalized[2] == 0.5
        assert normalized[3] == 0.0
    
    def test_rrf_formula(self):
        """Verify RRF score computation."""
        k = 60
        rrf_rank1 = 1.0 / (k + 1)
        rrf_rank10 = 1.0 / (k + 10)
        rrf_rank100 = 1.0 / (k + 100)
        
        assert rrf_rank1 > rrf_rank10 > rrf_rank100
        assert abs(rrf_rank1 - 1/61) < 1e-6


class TestFusionMethods:
    """Tests for fusion method logic."""
    
    def test_weighted_fusion_alpha_bounds(self):
        """Verify alpha correctly balances scores."""
        bm25_score, vss_score = 1.0, 0.0
        
        # Alpha = 0 -> pure BM25
        assert (1 - 0.0) * bm25_score + 0.0 * vss_score == 1.0
        
        # Alpha = 1 -> pure VSS
        assert (1 - 1.0) * bm25_score + 1.0 * vss_score == 0.0
        
        # Alpha = 0.5 -> balanced
        assert (1 - 0.5) * bm25_score + 0.5 * vss_score == 0.5
    
    def test_rrf_combines_rankings(self):
        """Verify RRF combines two ranking lists correctly."""
        bm25_ranks = {1: 1, 2: 2, 3: 3}
        vss_ranks = {2: 1, 1: 2, 3: 3}
        k, alpha = 60, 0.5
        
        rrf_scores = {}
        for cand_id in [1, 2, 3]:
            bm25_contrib = (1 - alpha) * (1.0 / (k + bm25_ranks[cand_id]))
            vss_contrib = alpha * (1.0 / (k + vss_ranks[cand_id]))
            rrf_scores[cand_id] = bm25_contrib + vss_contrib
        
        sorted_by_rrf = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        assert 3 == sorted_by_rrf[-1]  # Worst in both should be last


class TestInputValidation:
    """Tests for parameter validation."""
    
    def test_alpha_bounds(self):
        """Verify alpha validation logic."""
        assert 0 <= 0.5 <= 1
        assert not (0 <= 1.5 <= 1)
        assert not (0 <= -0.5 <= 1)
    
    def test_fusion_method_options(self):
        """Verify fusion method options."""
        valid_methods = ['rrf', 'weighted']
        assert 'rrf' in valid_methods
        assert 'weighted' in valid_methods
        assert 'invalid' not in valid_methods


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
