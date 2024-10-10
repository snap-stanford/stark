import torch
from typing import Any, Union, List, Dict
import re

from models.vss import VSS
from models.model import ModelForSTaRKQA
from stark_qa.tools.api import get_llm_output


def find_floating_number(text: str) -> List[float]:
    """
    Extract floating point numbers from the given text.

    Args:
        text (str): Input text from which to extract numbers.

    Returns:
        List[float]: List of extracted floating point numbers.
    """
    pattern = r'0\.\d+|1\.0'
    matches = re.findall(pattern, text)
    return [round(float(match), 4) for match in matches if float(match) <= 1.1]


class LLMReranker(ModelForSTaRKQA):
    
    def __init__(self,
                 kb,
                 llm_model: str,
                 emb_model: str,
                 query_emb_dir: str,
                 candidates_emb_dir: str,
                 sim_weight: float = 0.1,
                 max_cnt: int = 3,
                 max_k: int = 100,
                 device: str = 'cuda'):
        """
        Initializes the LLMReranker model.

        Args:
            kb (SemiStruct): Knowledge base.
            llm_model (str): Name of the LLM model.
            emb_model (str): Embedding model name.
            query_emb_dir (str): Directory to query embeddings.
            candidates_emb_dir (str): Directory to candidate embeddings.
            sim_weight (float): Weight for similarity score.
            max_cnt (int): Maximum count for retrying LLM response.
            max_k (int): Maximum number of top candidates to consider.
        """
        super(LLMReranker, self).__init__(kb)
        self.max_k = max_k
        self.emb_model = emb_model
        self.llm_model = llm_model
        self.sim_weight = sim_weight
        self.max_cnt = max_cnt

        self.query_emb_dir = query_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        self.parent_vss = VSS(kb, query_emb_dir, candidates_emb_dir, 
                              emb_model=emb_model, device=device)

    def forward(self, 
                query: Union[str, List[str]],
                query_id: Union[int, List[int]] = None,
                **kwargs: Any) -> Dict[int, float]:
        """
        Forward pass to compute predictions for the given query using LLM reranking.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list, None]): Query index (optional).

        Returns:
            pred_dict (dict): A dictionary of predicted scores or answer ids.
        """
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
        cand_len = len(top_k_node_ids)

        pred_dict = {}
        for idx, node_id in enumerate(top_k_node_ids):
            node_type = self.skb.get_node_type_by_id(node_id)
            prompt = (
                f'You are a helpful assistant that examines if a {node_type} '
                f'satisfies the requirements in a given query and assign a score from 0.0 to 1.0. '
                f'If the {node_type} does not satisfy any requirement in the query, the score should be 0.0. '
                f'If there exists explicit and strong evidence supporting that {node_type} '
                f'satisfies all aspects mentioned by the query, the score should be 1.0. If partial evidence or weak '
                f'evidence exists, the score should be between 0.0 and 1.0.\n'
                f'Here is the query:\n\"{query}\"\n'
                f'Here is the information about the {node_type}:\n' +
                self.skb.get_doc_info(node_id, add_rel=True) + '\n\n' +
                f'Please score the {node_type} based on how well it satisfies the query. '
                f'ONLY output the floating point score WITHOUT anything else. '
                f'Output: The numeric score of this {node_type} is: '
            )

            success = False
            for _ in range(self.max_cnt):
                try:
                    answer = get_llm_output(
                        prompt, 
                        self.llm_model, 
                        max_tokens=5
                    )
                    answer = find_floating_number(answer)
                    if len(answer) == 1:
                        answer = answer[0]
                        success = True
                        break
                except Exception as e:
                    print(f'Error: {e}, retrying...')

            if success:
                llm_score = float(answer)
                sim_score = (cand_len - idx) / cand_len
                score = llm_score + self.sim_weight * sim_score
                pred_dict[node_id] = score
            else:
                return initial_score_dict
        return pred_dict
