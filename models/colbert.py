import os
import os.path as osp

import subprocess
import torch
from typing import Any, Union, List, Dict
from tqdm import tqdm
import pandas as pd

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from collections import defaultdict

from models.model import ModelForSTaRKQA
from stark_qa import load_qa


class Colbertv2(ModelForSTaRKQA):
    
    url = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz"
    def __init__(self, 
                 skb, 
                 dataset_name,
                 human_generated_eval,
                 add_rel=False,
                 download_dir='output',
                 save_dir='output/colbertv2.0',
                 nbits=2, k=100
                 ):
        super(Colbertv2, self).__init__(skb)


        self.k = k
        self.nbits = nbits

        query_tsv_name = 'query_hg.tsv' if human_generated_eval else 'query.tsv'
        self.exp_name = dataset_name + '_hg' if human_generated_eval else dataset_name

        self.save_dir = save_dir
        self.download_dir = download_dir
        self.experiments_dir = './experiments'
        
        self.model_ckpt_dir = osp.join(self.download_dir, 'colbertv2.0') 
        self.query_tsv_path = osp.join(self.save_dir, query_tsv_name)
        self.doc_tsv_path = osp.join(self.save_dir, 'doc.tsv')
        self.index_ckpt_path = osp.join(self.save_dir, 'index.faiss')
        self.ranking_path = osp.join(self.save_dir, 'ranking.tsv')

        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.experiments_dir, exist_ok=True)

        qa_dataset = load_qa(dataset_name, human_generated_eval=human_generated_eval)
        self._check_query_csv(qa_dataset, self.query_tsv_path)
        self._check_doc_csv(skb, self.doc_tsv_path, add_rel)
        self._download()
        
        # load from tsv
        self.queries = Queries(self.query_tsv_path)
        self.collection = Collection(self.doc_tsv_path)

        self._prepare_indexer()

        self.score_dict = self.run_all()
    
    def _check_query_csv(self, qa_dataset, query_tsv_path):
        if not osp.exists(query_tsv_path):
            queries = {qa_dataset[i][1]: qa_dataset[i][0].replace('\n', ' ') 
                       for i in range(len(qa_dataset))}
            lines = [f"{qid}\t{q}" for qid, q in queries.items()]
            with open(query_tsv_path, 'w') as file:
                file.write('\n'.join(lines))
        else:
            print(f'Load existing queries from {query_tsv_path}')

    def _check_doc_csv(self, skb, doc_tsv_path, add_rel):
        # It can take quite a while if add_rel is True!
        indices = skb.candidate_ids
        # need to remap indices from 0 to N
        self.docid2pid = {idx: i for i, idx in enumerate(indices)}
        self.pid2docid = {i: idx for i, idx in enumerate(indices)}

        if not osp.exists(doc_tsv_path):
            corpus = {self.docid2pid[idx]: skb.get_doc_info(idx, add_rel=add_rel, compact=True)
                      for idx in tqdm(indices, desc="Gathering docs")}
            
            lines = [f"{idx}\t{doc}" for idx, doc in corpus.items()]
            with open(doc_tsv_path, 'w') as file:
                file.write('\n'.join(lines))
        else:
            print(f'Load existing documents from {doc_tsv_path}')
    
    def _download(self):
        if not osp.exists(osp.join(self.download_dir, 'colbertv2.0')):
            # Download the ColBERTv2 checkpoint
            download_command = f"wget {self.url} -P {self.download_dir}"
            subprocess.run(download_command, shell=True, check=True)

            # Extract the downloaded tar.gz file
            tar_command = f"tar -xvzf {osp.join(self.download_dir, 'colbertv2.0.tar.gz')} -C {self.download_dir}"
            subprocess.run(tar_command, shell=True, check=True)

    def _prepare_indexer(self):
        nranks = torch.cuda.device_count()
        with Run().context(RunConfig(nranks=nranks, experiment=self.exp_name)):
            config = ColBERTConfig(
                nbits=self.nbits,
                root=self.experiments_dir
            )
            indexer = Indexer(checkpoint=self.model_ckpt_dir, config=config)
            indexer.index(name=f"{self.exp_name}.nbits={self.nbits}", 
                          collection=self.doc_tsv_path, overwrite='reuse')

    def run_all(self):
        def find_file_path_by_name(name, path):
            for root, dirs, files in os.walk(path):
                if name in files:
                    return osp.join(root, name)
            return None
        
        exp_root = osp.join(self.experiments_dir, self.exp_name)
        ranking_path = find_file_path_by_name('ranking.tsv', exp_root)
        if ranking_path is None:
            nranks = torch.cuda.device_count()
            with Run().context(RunConfig(nranks=nranks, experiment=self.exp_name)):

                config = ColBERTConfig(
                    root=self.experiments_dir,
                )
                searcher = Searcher(index=f"{self.exp_name}.nbits={self.nbits}", config=config)
                queries = Queries(self.query_tsv_path)
                ranking = searcher.search_all(queries, k=self.k)
                ranking.save('ranking.tsv')
        
        self.ranking_path = find_file_path_by_name('ranking.tsv', exp_root)

        score_dict = defaultdict(dict)
        with open(self.ranking_path) as f:
            for line in f:
                qid, pid, rank, *score = line.strip().split('\t')
                qid, pid, rank = int(qid), int(pid), int(rank)
                if len(score) > 0:
                    assert len(score) == 1
                    score = float(score[0])
                    score_dict[qid][pid] = score
                else:
                    score_dict[qid][pid] = -999

        return score_dict

    def forward(self, 
                query: Union[str, None], 
                query_id: int, 
                **kwargs: Any) -> dict:
        """
        This works by directly extracting ranking from the ranking.tsv file.

        Args:
            query (str): Query string.
            query_id (int): Query index.

        Returns:
            pred_dict (dict): A dictionary of candidate ids and their corresponding similarity scores.
        """
        score_dict = self.score_dict[query_id]
        return {self.pid2docid[pid]: score for pid, score in score_dict.items()}
