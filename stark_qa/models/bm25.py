from typing import Any, Union, List, Dict, Optional

from tqdm import tqdm

from stark_qa.models.base import ModelForSTaRKQA
import bm25s


class BM25(ModelForSTaRKQA):
    """
    BM25 Model for STaRK QA.

    This model uses the BM25 algorithm for information retrieval to rank candidates
    based on their relevance to the query.
    """
    
    def __init__(self, skb: Any) -> None:
        """
        Initialize the BM25 model with the given knowledge base.

        Args:
            skb (Any): The knowledge base containing candidate documents.
        """
        super(BM25, self).__init__(skb)

        # Get the candidate indices and their corresponding documents
        self.indices: List[int] = skb.candidate_ids
        self.corpus: List[str] = [
            skb.get_doc_info(idx) for idx in tqdm(self.indices, desc="Gathering documents")
        ]

        # Create the BM25 retriever and index the corpus
        self.retriever = bm25s.BM25(corpus=self.corpus)
        self.retriever.index(bm25s.tokenize(self.corpus))

        # Build a mapping from positions in the corpus to candidate IDs
        self.position_to_candidate_id: Dict[int, int] = {
            idx: candidate_id for idx, candidate_id in enumerate(self.indices)
        }
            
    def forward(
        self, 
        query: str, 
        query_id: Optional[int] = None, 
        k: int = 100, 
        **kwargs: Any
    ) -> Dict[int, float]:
        """
        Compute similarity scores for the given query using BM25.

        Args:
            query (str): The query string.
            query_id (Optional[int], optional): The query ID. Defaults to None.
            k (int, optional): The number of top candidates to retrieve. Defaults to 100.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[int, float]: A dictionary mapping candidate IDs to their corresponding similarity scores.

        Raises:
            KeyError: If a retrieved document position is not found in the position_to_candidate_id mapping.
        """
        # Tokenize the query
        tokenized_query = bm25s.tokenize(query)
        
        # Retrieve top k documents
        results, scores = self.retriever.retrieve(tokenized_query, k=k)

        # Get positions of retrieved documents
        positions = results[0]  # Positions in the corpus

        # Map positions to candidate IDs
        try:
            candidate_ids = [self.position_to_candidate_id[pos] for pos in positions]
        except KeyError as e:
            raise KeyError(f"Position {e} not found in position_to_candidate_id mapping.")

        # Get the corresponding scores
        scores_list = scores[0].tolist()

        # Return a dictionary mapping candidate IDs to scores
        return dict(zip(candidate_ids, scores_list))
