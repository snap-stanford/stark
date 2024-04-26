import os.path as osp
import os
import argparse
import numpy as np
from tqdm import tqdm
import gdown



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon")
    parser.add_argument("--emb_dir", default="emb", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    query_emb_token = {'amazon': '1-zyI84MMh6r66-faOFSc2rWTeIUw3VZW',
                       'mag': '1HSfUrSKBa7mJbECFbnKPQgd6HSsI8spT',
                       'primekg': '1MshwJttPZsHEM2cKA5T13SIrsLeBEdyU'}
    node_emb_token = {'amazon': '18NU7tw_Tcyp9YobxKubLISBncwLaAiJz',
                      'mag': '1oVdScsDRuEpCFXtWQcTAx7ycvOggWF17',
                      'primekg': '16EJvCMbgkVrQ0BuIBvLBp-BYPaye-Edy'}

    emb_dir = args.emb_dir
    dataset = args.dataset
    query_emb_url = 'https://drive.google.com/uc?id=' + query_emb_token[dataset]
    node_emb_url = 'https://drive.google.com/uc?id=' + node_emb_token[dataset]

    emb_dir = osp.join(emb_dir, dataset)
    query_emb_dir = osp.join(emb_dir, "query")
    node_emb_dir = osp.join(emb_dir, "doc")
    os.makedirs(query_emb_dir, exist_ok=True)
    os.makedirs(node_emb_dir, exist_ok=True)
    query_emb_path = osp.join(query_emb_dir, "query_emb_dict.pt")
    node_emb_path = osp.join(node_emb_dir, "candidate_emb_dict.pt")
    
    gdown.download(query_emb_url, query_emb_path, quiet=False)
    gdown.download(node_emb_url, node_emb_path, quiet=False)

