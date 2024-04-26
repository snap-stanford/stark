import sys

import os
import os.path as osp

import torch
import random
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('.')
from src.benchmarks import get_semistructured_data, get_qa_dataset
from src.tools.api import get_openai_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='amazon')
    parser.add_argument('--mode', default='doc', choices=['doc', 'query'])
    parser.add_argument('--emb_model', default='text-embedding-ada-002', 
                        choices=[
                            'text-embedding-ada-002', 
                            'text-embedding-3-small', 
                            'text-embedding-3-large'
                            ]
                        )
    parser.add_argument("--emb_dir", default="emb", type=str)
    parser.add_argument('--add_rel', action='store_true', default=False, 
                        help='add relation to the text')
    parser.add_argument('--compact', action='store_true', default=False, 
                        help='make the text compact when input to the model')
    return parser.parse_args()
    
    

if __name__ == '__main__':
    args = parse_args()
    emb_dir = osp.join(args.emb_dir, args.dataset, args.mode)
    os.makedirs(emb_dir, exist_ok=True)

    if args.mode == 'doc':
        kb = get_semistructured_data(args.dataset)
        lst = kb.candidate_ids
        emb_path = osp.join(emb_dir, f'candidate_emb_dict.pt')
    if args.mode == 'query':
        qa_dataset = get_qa_dataset(args.dataset)
        lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
        emb_path = osp.join(emb_dir, f'query_emb_dict.pt')
    random.shuffle(lst)
            
    if osp.exists(emb_path):
        emb_dict = torch.load(emb_path)
        exisiting_indices = list(emb_dict.keys())
        print(f'Loaded existing embeddings from {emb_path}. Size: {len(emb_dict)}')
    else:
        emb_dict = {}
        exisiting_indices = []

    texts, indices = [], []
    for idx in tqdm(lst):
        if idx in exisiting_indices:
            continue
        if args.mode == 'query':
            text = qa_dataset.get_query_by_qid(idx)
        elif args.mode == 'doc':
            text = kb.get_doc_info(idx, add_rel=args.add_rel, compact=args.compact)
        texts.append(text)
        indices.append(idx)
        
    print(f'Generating embeddings for {len(texts)} texts...')
    embs = get_openai_embeddings(texts, model=args.emb_model).view(len(texts), -1).cpu()
    print('Embedding size:', embs.size())
    
    for idx, emb in zip(indices, embs):
        emb_dict[idx] = emb
    torch.save(emb_dict, emb_path)
    print(f'Saved embeddings to {emb_path}!')
