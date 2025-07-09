# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STaRK is a large-scale Semi-structured Retrieval Benchmark for LLMs that evaluate retrieval performance on textual and relational knowledge bases. The project includes three domains:
- **Amazon**: E-commerce product recommendations with relational product data
- **Prime**: Academic paper retrieval with citation networks  
- **MAG**: Microsoft Academic Graph with institutional relationships

This is a research benchmark that provides both evaluation datasets and baseline retrieval models.

## Development Environment

### Package Installation
```bash
# Install as package (recommended)
pip install stark-qa

# Or install from source
pip install -r requirements.txt
```

### Python Version
- Requires Python >=3.8 and <3.12
- Uses conda/pip for dependency management

## Key Commands

### Data Loading and Setup
```bash
# Download embeddings for evaluation
python emb_download.py --dataset amazon --emb_dir emb/

# Generate custom embeddings
python emb_generate.py --dataset amazon --mode query --emb_dir emb/ --emb_model text-embedding-ada-002
```

### Running Evaluations
```bash
# Basic VSS evaluation
python eval.py --dataset amazon --model VSS --emb_dir emb/ --output_dir output/ --emb_model text-embedding-ada-002 --split test --save_pred

# LLM reranker evaluation
python eval.py --dataset amazon --model LLMReranker --emb_dir emb/ --output_dir output/ --emb_model text-embedding-ada-002 --split human_generated_eval --llm_model gpt-4-1106-preview --save_pred

# BM25 baseline
python eval.py --dataset amazon --model BM25 --output_dir output/ --split test --save_pred

# LLM reranker with ollama model
python eval.py --dataset amazon --model LLMReranker --emb_dir emb/ --output_dir output/ --emb_model text-embedding-ada-002 --split test --llm_model llama3.2:3b --save_pred
```

### Environment Variables for API Keys
```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export VOYAGE_API_KEY=your_key
```

### Ollama Setup (for local models)
```bash
# Install ollama (macOS)
brew install ollama

# Start ollama service
ollama serve

# Pull a model (example)
ollama pull llama3.2:3b
```

## Code Architecture

### Core Components

1. **Data Loading (`stark_qa/`)**: 
   - `load_qa()`: Loads question-answer datasets
   - `load_skb()`: Loads semi-structured knowledge bases
   - `load_model()`: Instantiates retrieval models

2. **Knowledge Bases (`stark_qa/skb/`)**:
   - `AmazonSKB`, `PrimeSKB`, `MagSKB`: Domain-specific knowledge base implementations
   - `SKB`: Base class for all knowledge bases

3. **Retrieval Models (`stark_qa/models/`)**:
   - `BM25`: Traditional sparse retrieval
   - `VSS`: Vector semantic search
   - `MultiVSS`: Multi-vector semantic search with chunking
   - `LLMReranker`: LLM-based reranking of candidates

4. **Evaluation (`eval.py`)**:
   - Main evaluation script supporting all models and datasets
   - Handles embedding generation, candidate retrieval, and metric computation

### Data Flow

1. **Knowledge Base Loading**: `load_skb()` downloads/loads processed knowledge graphs
2. **Query Processing**: `load_qa()` provides question-answer pairs for evaluation
3. **Embedding Generation**: `emb_generate.py` creates vector representations
4. **Retrieval**: Models retrieve relevant candidates from knowledge bases
5. **Evaluation**: `eval.py` computes metrics and saves results

### Configuration

- `config/default_args.json`: Default parameters for MultiVSS models per dataset
- `pyproject.toml`: Package configuration and dependencies
- Model parameters can be overridden via command line arguments

## Supported Datasets and Models

### Datasets
- `amazon`: E-commerce product search
- `prime`: Academic paper retrieval
- `mag`: Microsoft Academic Graph

### Models
- `BM25`: Traditional keyword matching
- `VSS`: Single-vector semantic search
- `MultiVSS`: Multi-vector search with text chunking
- `LLMReranker`: LLM-based candidate reranking

### Evaluation Splits
- `train`, `val`, `test`: Standard splits
- `test-0.1`: 10% random test sample
- `human_generated_eval`: Human-curated evaluation queries

## Key Files

- `eval.py`: Main evaluation script
- `emb_download.py`: Download pre-computed embeddings
- `emb_generate.py`: Generate custom embeddings
- `stark_qa/`: Core package with data loading and models
- `config/default_args.json`: Default model configurations
- `exploration.ipynb`: Data exploration notebook
- `load_dataset.ipynb`: Dataset loading examples