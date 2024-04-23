import os.path as osp
from .vss import  VSS
from .llm_qa import LLMQA
from .multi_vss import MultiVSS

def get_model(args, database):
    model_name = args.model
    if model_name == 'VSS':
        return VSS(
            database,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir
        )
    if model_name == 'MultiVSS':
        return MultiVSS(
            database,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir,
            chunk_emb_dir=args.chunk_emb_dir,
            aggregate=args.aggregate,
            chunk_size=args.chunk_size,
            max_k=args.multi_vss_topk
        )
    if model_name in ['GPTQA', 'ClaudeQA']:
        return LLMQA(database, 
                     model_name=args.llm_version,
                     query_emb_dir=args.query_emb_dir, 
                     candidates_emb_dir=args.node_emb_dir,
                     max_k=args.llm_topk
                     )
    raise NotImplementedError(f'{model_name} not implemented')