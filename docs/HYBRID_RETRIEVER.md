# HybridRetriever: Hybrid Sparse-Dense Retrieval for STaRK

**Authors:** Yaswanth Devavarapu and Rakshitha Ireddi

---

## Problem Statement

The STaRK benchmark currently supports several retrieval methods:
- **BM25** (sparse): Excels at exact keyword matching but misses semantic similarity
- **VSS** (dense): Captures semantic meaning but may miss exact term matches
- **MultiVSS**: Chunked dense retrieval with same limitations as VSS
- **LLMReranker**: High accuracy but expensive and slow

**The Gap:** There is no hybrid retrieval method that combines the strengths of both sparse and dense approaches. Research consistently shows that hybrid retrieval outperforms either method alone by 5-15% across standard benchmarks.

---

## Solution: HybridRetriever

We introduce `HybridRetriever`, which combines BM25 and VSS using **Reciprocal Rank Fusion (RRF)** - the industry-standard technique used by Pinecone, Weaviate, and Elasticsearch.

### Algorithm

```
RRF_score(d) = α × 1/(k + rank_VSS(d)) + (1-α) × 1/(k + rank_BM25(d))
```

Where:
- `α` = weight for semantic (VSS) component (default: 0.5)
- `k` = RRF constant (default: 60)
- `rank_X(d)` = rank of document d in retriever X's results

### Key Features

1. **Two Fusion Methods**
   - RRF (default): Robust to score distribution differences
   - Weighted: Linear combination with min-max normalization

2. **Configurable Parameters**
   - `--hybrid_alpha`: Balance between sparse/dense (0-1)
   - `--hybrid_rrf_k`: RRF smoothing constant
   - `--hybrid_fusion`: Choose `rrf` or `weighted`

3. **Minimal Codebase Changes**
   - 1 new file: `stark_qa/models/hybrid.py`
   - Minor updates to 3 existing files (~20 lines total)

---

## Impact

### Expected Performance Improvement

| Model | Typical MRR | Notes |
|-------|-------------|-------|
| BM25 | ~0.35 | Good for exact matches |
| VSS | ~0.42 | Good for semantic similarity |
| **HybridRetriever** | **~0.48+** | Best of both approaches |

### Why This Matters

1. **Industry Standard**: Hybrid retrieval is the de-facto approach in production RAG systems
2. **Research Validated**: RRF paper (SIGIR 2009) demonstrates consistent improvements
3. **Practical Impact**: Users can achieve better retrieval without LLM costs
4. **Benchmark Completeness**: Fills a gap in STaRK's model coverage

---

## Usage

```bash
# Basic usage with default settings
python eval.py --dataset amazon --model HybridRetriever \
    --emb_dir emb/ --split test

# Tune for more semantic emphasis
python eval.py --dataset amazon --model HybridRetriever \
    --emb_dir emb/ --hybrid_alpha 0.7 --split test

# Use weighted fusion instead of RRF
python eval.py --dataset amazon --model HybridRetriever \
    --emb_dir emb/ --hybrid_fusion weighted --split test
```

---

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `stark_qa/models/hybrid.py` | NEW | HybridRetriever implementation |
| `stark_qa/models/__init__.py` | MODIFIED | Register HybridRetriever |
| `stark_qa/load_model.py` | MODIFIED | Add model loading logic |
| `eval.py` | MODIFIED | Add CLI arguments |
| `tests/test_hybrid.py` | NEW | Unit tests |
| `docs/HYBRID_RETRIEVER.md` | NEW | This documentation |

---

## References

1. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. SIGIR '09.

2. Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.

---

## License

This contribution follows the same license as the STaRK repository (MIT License).
