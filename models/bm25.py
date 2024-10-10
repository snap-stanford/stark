import os.path as osp
import torch
from typing import Any, Union, List, Dict
from models.model import ModelForSTaRKQA
from tqdm import tqdm
import pandas as pd
import bm25s


class BM25(ModelForSTaRKQA):
    
    def __init__(self, skb):

        super(BM25, self).__init__(skb)

        self.indices = skb.candidate_ids
        self.corpus = [skb.get_doc_info(idx) for idx in tqdm(self.indices, desc="Gathering docs")]
        
        # Create the BM25 model and index the corpus
        self.retriever = bm25s.BM25(corpus=self.corpus)
        self.retriever.index(bm25s.tokenize(self.corpus))

        # build hash map from text to index
        self.text_to_index = {hash(text): index for text, index in zip(self.corpus, self.indices)}
        
    def forward(self, 
                query: str, 
                query_id: Union[int, None] = None, 
                k: int = 100, 
                **kwargs: Any) -> dict:
        """
        Forward pass to compute similarity scores for the given query.

        Args:
            query (str): Query string.
            query_id (int): Query index.

        Returns:
            pred_dict (dict): A dictionary of candidate ids and their corresponding similarity scores.
        """
        results, scores = self.retriever.retrieve(bm25s.tokenize(query), k=k)
        indices = [self.text_to_index[hash(result.item())] for result in results[0]]
        scores = scores[0].tolist()
        return dict(zip(indices, scores))
