from .gritlm import get_gritlm_embeddings
from .llm2vec import get_llm2vec_embeddings
from .openai import get_openai_embeddings
from .voyage import get_voyage_embeddings

REGISTERED_EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "voyage-large-2-instruct",
    "GritLM/GritLM-*",
    "McGill-NLP/LLM2Vec-*"
]