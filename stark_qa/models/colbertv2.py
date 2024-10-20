import os
import os.path as osp
import subprocess
from typing import Any, Union, List, Dict, Optional
from collections import defaultdict

import torch
from tqdm import tqdm

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

from stark_qa.models.base import ModelForSTaRKQA
from stark_qa import load_qa


class Colbertv2(ModelForSTaRKQA):
    """
    ColBERTv2 Model for STaRK QA.

    This model integrates the ColBERTv2 dense retrieval model to rank candidates based on their relevance
    to a query from a question-answering dataset.
    """
    
    url = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz"
    
    def __init__(self, 
                 skb: Any, 
                 dataset_name: str, 
                 human_generated_eval: bool, 
                 add_rel: bool = False, 
                 download_dir: str = 'output', 
                 save_dir: str = 'output/colbertv2.0', 
                 nbits: int = 2, 
                 k: int = 100):
        """
        Initialize the ColBERTv2 model with the given knowledge base and parameters.

        Args:
            skb (Any): The knowledge base containing candidate documents.
            dataset_name (str): The name of the dataset being used.
            human_generated_eval (bool): Whether to use human-generated queries for evaluation.
            add_rel (bool, optional): Whether to add relational information to the document. Defaults to False.
            download_dir (str, optional): Directory where the ColBERTv2 model is downloaded. Defaults to 'output'.
            save_dir (str, optional): Directory where the experiment output is saved. Defaults to 'output/colbertv2.0'.
            nbits (int, optional): Number of bits for indexing. Defaults to 2.
            k (int, optional): Number of top candidates to retrieve. Defaults to 100.
        """
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

        # Load the question-answer dataset and check for required files
        qa_dataset = load_qa(dataset_name, human_generated_eval=human_generated_eval)
        self._check_query_csv(qa_dataset, self.query_tsv_path)
        self._check_doc_csv(skb, self.doc_tsv_path, add_rel)

        # Download and set up the ColBERTv2 model
        self._download()

        # Load the queries and documents into ColBERTv2 format
        self.queries = Queries(self.query_tsv_path)
        self.collection = Collection(self.doc_tsv_path)

        # Prepare the indexer and build the index
        self._prepare_indexer()

        # Run the model and store the results
        self.score_dict = self.run_all()
    
    def _check_query_csv(self, qa_dataset: Any, query_tsv_path: str) -> None:
        """
        Check if the query TSV file exists; if not, create it from the QA dataset.

        Args:
            qa_dataset (Any): The question-answer dataset.
            query_tsv_path (str): Path to the query TSV file.
        """
        if not osp.exists(query_tsv_path):
            queries = {qa_dataset[i][1]: qa_dataset[i][0].replace('\n', ' ') 
                       for i in range(len(qa_dataset))}
            lines = [f"{qid}\t{q}" for qid, q in queries.items()]
            with open(query_tsv_path, 'w') as file:
                file.write('\n'.join(lines))
        else:
            print(f'Loaded existing queries from {query_tsv_path}')

    def _check_doc_csv(self, skb: Any, doc_tsv_path: str, add_rel: bool) -> None:
        """
        Check if the document TSV file exists; if not, create it from the knowledge base.

        Args:
            skb (Any): The knowledge base containing candidate documents.
            doc_tsv_path (str): Path to the document TSV file.
            add_rel (bool): Whether to add relational information to the documents.
        """
        indices = skb.candidate_ids
        self.docid2pid = {idx: i for i, idx in enumerate(indices)}
        self.pid2docid = {i: idx for i, idx in enumerate(indices)}

        if not osp.exists(doc_tsv_path):
            corpus = {self.docid2pid[idx]: skb.get_doc_info(idx, add_rel=add_rel, compact=True)
                      for idx in tqdm(indices, desc="Gathering documents")}
            
            lines = [f"{idx}\t{doc}" for idx, doc in corpus.items()]
            with open(doc_tsv_path, 'w') as file:
                file.write('\n'.join(lines))
        else:
            print(f'Loaded existing documents from {doc_tsv_path}')
    
    def _download(self) -> None:
        """
        Download the ColBERTv2 model if not already available.
        """
        if not osp.exists(osp.join(self.download_dir, 'colbertv2.0')):
            # Download the ColBERTv2 checkpoint
            download_command = f"wget {self.url} -P {self.download_dir}"
            subprocess.run(download_command, shell=True, check=True)

            # Extract the downloaded tar.gz file
            tar_command = f"tar -xvzf {osp.join(self.download_dir, 'colbertv2.0.tar.gz')} -C {self.download_dir}"
            subprocess.run(tar_command, shell=True, check=True)

    def _prepare_indexer(self) -> None:
        """
        Prepare the BM25 indexer for the document corpus.
        """
        nranks = torch.cuda.device_count()
        with Run().context(RunConfig(nranks=nranks, experiment=self.exp_name)):
            config = ColBERTConfig(nbits=self.nbits, root=self.experiments_dir)
            indexer = Indexer(checkpoint=self.model_ckpt_dir, config=config)
            indexer.index(name=f"{self.exp_name}.nbits={self.nbits}", collection=self.doc_tsv_path, overwrite='reuse')

    def run_all(self) -> Dict[int, Dict[int, float]]:
        """
        Run the retrieval for all queries and store the rankings.

        Returns:
            Dict[int, Dict[int, float]]: A dictionary mapping query IDs to a dictionary of candidate scores.
        """
        def find_file_path_by_name(name: str, path: str) -> Optional[str]:
            """
            Find the file path by its name in a given directory.

            Args:
                name (str): The name of the file to find.
                path (str): The directory to search.

            Returns:
                Optional[str]: The file path if found, None otherwise.
            """
            for root, dirs, files in os.walk(path):
                if name in files:
                    return osp.join(root, name)
            return None
        
        exp_root = osp.join(self.experiments_dir, self.exp_name)
        ranking_path = find_file_path_by_name('ranking.tsv', exp_root)
        if ranking_path is None:
            nranks = torch.cuda.device_count()
            with Run().context(RunConfig(nranks=nranks, experiment=self.exp_name)):
                config = ColBERTConfig(root=self.experiments_dir)
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
                **kwargs: Any) -> Dict[int, float]:
        """
        Forward pass to retrieve rankings for the given query.

        Args:
            query (str): The query string.
            query_id (int): The query index.

        Returns:
            Dict[int, float]: A dictionary of candidate IDs and their corresponding similarity scores.
        """
        score_dict = self.score_dict[query_id]
        return {self.pid2docid[pid]: score for pid, score in score_dict.items()}
