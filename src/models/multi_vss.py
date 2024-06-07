import os.path as osp
import torch
from typing import Any, Union, List, Dict
from src.models.model import ModelForSemiStructQA
from src.models.vss import VSS
from src.tools.api import get_openai_embeddings
from src.tools.process_text import chunk_text


class MultiVSS(ModelForSemiStructQA):
    
    def __init__(self, 
                 kb,
                 query_emb_dir: str,
                 candidates_emb_dir: str,
                 chunk_emb_dir: str,
                 emb_model: str = 'text-embedding-ada-002',
                 aggregate: str = 'top3_avg',
                 max_k: int = 50,
                 chunk_size: int = 256):
        """
        Multivector Vector Similarity Search

        Args:
            kb (SemiStruct): Knowledge base.
            query_emb_dir (str): Directory to query embeddings.
            candidates_emb_dir (str): Directory to candidate embeddings.
            chunk_emb_dir (str): Directory to chunk embeddings.
            emb_model (str): Embedding model name.
            aggregate (str): Aggregation method for similarity scores ('max', 'avg', 'top{k}_avg').
            max_k (int): Maximum number of top candidates to consider.
            chunk_size (int): Size of chunks for text processing.
        """
        super(MultiVSS, self).__init__(kb)
        self.kb = kb
        self.aggregate = aggregate  # 'max', 'avg', 'top{k}_avg'
        self.max_k = max_k
        self.chunk_size = chunk_size
        self.emb_model = emb_model
        self.query_emb_dir = query_emb_dir
        self.chunk_emb_dir = chunk_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        self.parent_vss = VSS(kb, query_emb_dir, candidates_emb_dir, emb_model=emb_model)

    def forward(self, 
                query: Union[str, List[str]],
                query_id: Union[int, List[int]],
                **kwargs: Any) -> Dict[int, float]:
        """
        Forward pass to compute predictions for the given query using MultiVSS.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list]): Query index.
            
        Returns:
            pred_dict (dict): A dictionary of predicted scores or answer ids.
        """
        query_emb = self._get_query_emb(query, query_id)

        initial_score_dict = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())

        # Get the ids with top k highest scores
        top_k_idx = torch.topk(
            torch.FloatTensor(node_scores),
            min(self.max_k, len(node_scores)),
            dim=-1
        ).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]

        pred_dict = {}
        for node_id in top_k_node_ids:
            doc = self.kb.get_doc_info(node_id, add_rel=True, compact=True)
            chunks = chunk_text(doc, chunk_size=self.chunk_size)
            chunk_path = osp.join(self.chunk_emb_dir, f'{node_id}_size={self.chunk_size}.pt')
            if osp.exists(chunk_path):
                chunk_embs = torch.load(chunk_path)
            else:
                chunk_embs = get_openai_embeddings(chunks, model=self.emb_model)
                torch.save(chunk_embs, chunk_path)
            print(f'chunk_embs.shape: {chunk_embs.shape}')

            similarity = torch.matmul(query_emb.cuda(), chunk_embs.cuda().T).cpu().view(-1)
            if self.aggregate == 'max':
                pred_dict[node_id] = torch.max(similarity).item()
            elif self.aggregate == 'avg':
                pred_dict[node_id] = torch.mean(similarity).item()
            elif 'top' in self.aggregate:
                k = int(self.aggregate.split('_')[0][len('top'):])
                pred_dict[node_id] = torch.mean(
                    torch.topk(similarity, k=min(k, len(chunks)), dim=-1).values
                ).item()

        return pred_dict
