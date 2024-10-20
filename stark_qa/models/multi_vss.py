import os.path as osp
import torch
from typing import Any, Union, List, Dict
from stark_qa.models.base import ModelForSTaRKQA
from stark_qa.models.vss import VSS
from stark_qa.tools.llm_lib.get_llm_embeddings import get_llm_embeddings
from stark_qa.tools.process_text import chunk_text


class MultiVSS(ModelForSTaRKQA):
    """
    Multi-Vector Vector Similarity Search (MultiVSS) model for knowledge base retrieval.
    
    This model combines vector similarity search across multiple chunks of candidate documents, 
    allowing for more granular comparisons. The model can aggregate scores from different 
    chunks in various ways, such as 'max', 'avg', or 'top{k}_avg'.
    """
    
    def __init__(self, 
                 skb: Any,
                 query_emb_dir: str,
                 candidates_emb_dir: str,
                 chunk_emb_dir: str,
                 emb_model: str = 'text-embedding-ada-002',
                 aggregate: str = 'top3_avg',
                 max_k: int = 50,
                 chunk_size: int = 256,
                 device: str = 'cuda') -> None:
        """
        Initialize the MultiVSS model.

        Args:
            skb (Any): Knowledge base containing semi-structured data.
            query_emb_dir (str): Directory where query embeddings are stored.
            candidates_emb_dir (str): Directory where candidate embeddings are stored.
            chunk_emb_dir (str): Directory where chunk embeddings are stored.
            emb_model (str, optional): Embedding model to use for similarity search. Defaults to 'text-embedding-ada-002'.
            aggregate (str, optional): Aggregation method for similarity scores ('max', 'avg', 'top{k}_avg'). Defaults to 'top3_avg'.
            max_k (int, optional): Maximum number of top candidates to consider. Defaults to 50.
            chunk_size (int, optional): Size of text chunks for document processing. Defaults to 256.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        super(MultiVSS, self).__init__(skb)
        self.skb = skb
        self.aggregate = aggregate
        self.max_k = max_k
        self.chunk_size = chunk_size
        self.emb_model = emb_model
        self.query_emb_dir = query_emb_dir
        self.chunk_emb_dir = chunk_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        self.parent_vss = VSS(skb, query_emb_dir, candidates_emb_dir, emb_model=emb_model, device=device)

    def forward(self, 
                query: Union[str, List[str]],
                query_id: Union[int, List[int]],
                **kwargs: Any) -> Dict[int, float]:
        """
        Compute predictions for the given query using MultiVSS.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list]): Query index.
            
        Returns:
            Dict[int, float]: A dictionary of candidate IDs mapped to their predicted similarity scores.
        """
        # Get query embeddings
        query_emb = self.get_query_emb(query, query_id)

        # Get initial ranking using VSS
        initial_score_dict = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())

        # Get the top k highest-scoring candidates
        top_k_idx = torch.topk(
            torch.FloatTensor(node_scores),
            min(self.max_k, len(node_scores)),
            dim=-1
        ).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]

        # Initialize the prediction dictionary
        pred_dict = {}

        # Process each candidate document
        for node_id in top_k_node_ids:
            # Get document information and chunk it
            doc = self.skb.get_doc_info(node_id, add_rel=True, compact=True)
            chunks = chunk_text(doc, chunk_size=self.chunk_size)

            # Check for existing chunk embeddings or generate new ones
            chunk_path = osp.join(self.chunk_emb_dir, f'{node_id}_size={self.chunk_size}.pt')
            if osp.exists(chunk_path):
                chunk_embs = torch.load(chunk_path)
            else:
                chunk_embs = get_llm_embeddings(chunks, model=self.emb_model)
                torch.save(chunk_embs, chunk_path)

            # Calculate similarity between query and document chunks
            similarity = torch.matmul(query_emb.cuda(), chunk_embs.cuda().T).cpu().view(-1)

            # Aggregate similarity scores using the specified method
            if self.aggregate == 'max':
                pred_dict[node_id] = torch.max(similarity).item()
            elif self.aggregate == 'avg':
                pred_dict[node_id] = torch.mean(similarity).item()
            elif 'top' in self.aggregate:
                # Aggregate using top k similarities
                k = int(self.aggregate.split('_')[0][len('top'):])
                pred_dict[node_id] = torch.mean(
                    torch.topk(similarity, k=min(k, len(chunks)), dim=-1).values
                ).item()

        return pred_dict
