import os
import os.path as osp
import random
import sys
import argparse
import pandas as pd

import torch
from tqdm import tqdm

sys.path.append('.')
from stark_qa import load_skb, load_qa
from stark_qa.tools.api import get_openai_embeddings, get_voyage_embeddings


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset and embedding model selection
    parser.add_argument('--dataset', default='amazon', choices=['amazon', 'prime', 'mag'])
    parser.add_argument('--emb_model', default='text-embedding-ada-002', 
                        choices=[
                            'text-embedding-ada-002', 
                            'text-embedding-3-small', 
                            'text-embedding-3-large',
                            'voyage-large-2-instruct'
                            ]
                        )

    # Mode settings
    parser.add_argument('--mode', default='doc', choices=['doc', 'query'])

    # Path settings
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--emb_dir", default="emb/", type=str)

    # text settings
    parser.add_argument('--add_rel', action='store_true', default=False, help='add relation to the text')
    parser.add_argument('--compact', action='store_true', default=False, help='make the text compact when input to the model')

    # Evaluation settings
    parser.add_argument("--human_generated_eval", action="store_true", help="if mode is `query`, then generating query embeddings on human generated evaluation split")

    # Batch and node settings
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--n_max_nodes", default=64, type=int)

    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_args()
    mode_surfix = '_human_generated_eval' if args.human_generated_eval and args.mode == 'query' else ''
    mode_surfix += '_no_rel' if not args.add_rel else ''
    mode_surfix += '_no_compact' if not args.compact else ''
    emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f'{args.mode}{mode_surfix}')
    csv_cache = osp.join(args.data_dir, args.dataset, f'{args.mode}{mode_surfix}.csv')
    print(f'Embedding directory: {emb_dir}')
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_cache), exist_ok=True)

    if args.mode == 'doc':
        kb = load_skb(args.dataset)
        lst = kb.candidate_ids
        emb_path = osp.join(emb_dir, f'candidate_emb_dict.pt')
    if args.mode == 'query':
        qa_dataset = load_qa(args.dataset, human_generated_eval=args.human_generated_eval)
        lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
        emb_path = osp.join(emb_dir, f'query_emb_dict.pt')
    random.shuffle(lst)
            
    if osp.exists(emb_path):
        emb_dict = torch.load(emb_path)
        existing_indices = list(emb_dict.keys())
        print(f'Loaded existing embeddings from {emb_path}. Size: {len(emb_dict)}')
    else:
        emb_dict = {}
        existing_indices = []

    if osp.exists(csv_cache):
        df = pd.read_csv(csv_cache)
        indices = df['index'].tolist()
        texts = df['text'].tolist()
    else:
        texts, indices = [], []
    existing_indices = set(existing_indices + indices)

    for idx in tqdm(lst):
        if idx in existing_indices:
            continue
        if args.mode == 'query':
            text = qa_dataset.get_query_by_qid(idx)
        elif args.mode == 'doc':
            text = kb.get_doc_info(idx, add_rel=args.add_rel, compact=args.compact)
        texts.append(text)
        indices.append(idx)

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame({'index': indices, 'text': texts})
    df.to_csv(csv_cache, index=False)

    print(f'Generating embeddings for {len(texts)} texts...')
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i:i+args.batch_size]
        if 'voyage' in args.emb_model:
            get_emb_func = get_voyage_embeddings
        else:
            get_emb_func = get_openai_embeddings
            
        batch_embs = get_emb_func(
            batch_texts, 
            model=args.emb_model, 
            n_max_nodes=args.n_max_nodes
        ).view(len(batch_texts), -1).cpu()
        batch_indices = indices[i:i+args.batch_size]
        for idx, emb in zip(batch_indices, batch_embs):
            emb_dict[idx] = emb
        
    torch.save(emb_dict, emb_path)
    print(f'Saved {len(emb_dict)} embeddings to {emb_path}!')
