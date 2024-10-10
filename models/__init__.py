from .vss import  VSS
from .llm_reranker import LLMReranker
from .multi_vss import MultiVSS
from .bm25 import BM25


def get_model(args, skb, **kwargs):
    model_name = args.model
    if model_name == 'BM25':
        return BM25(skb)
    if model_name == 'Colbertv2':
        try:
            from .colbert import Colbertv2
            return Colbertv2(skb, 
                             dataset_name=args.dataset,
                             save_dir=args.output_dir,
                             download_dir=args.download_dir,
                             human_generated_eval=args.split=='human_generated_eval',
                             **kwargs
                             )
        except ImportError:
            raise ImportError("Please install the colbert package using `pip install colbert-ai`.")
    elif model_name == 'VSS':
        return VSS(
            skb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir,
            device=args.device
        )
    if model_name == 'MultiVSS':
        return MultiVSS(
            skb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir,
            chunk_emb_dir=args.chunk_emb_dir,
            aggregate=args.aggregate,
            chunk_size=args.chunk_size,
            max_k=args.multi_vss_topk,
            device=args.device
        )
    if model_name == 'LLMReranker':
        return LLMReranker(skb, 
                           emb_model=args.emb_model,
                           llm_model=args.llm_model,
                           query_emb_dir=args.query_emb_dir, 
                           candidates_emb_dir=args.node_emb_dir,
                           max_cnt = args.max_retry,
                           max_k=args.llm_topk,
                           device=args.device
                           )
    raise NotImplementedError(f'{model_name} not implemented')