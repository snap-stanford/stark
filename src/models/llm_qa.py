import torch
from tqdm import tqdm
from typing import Any, Union
import os
import os.path as osp

from src.models.vss import VSS
from src.models.model import ModelForSemiStructQA
from src.tools.api import get_gpt_output, get_ada_embedding, complete_text_claude
import re


def find_floating_number(text):
    pattern = r'0\.\d+|1\.0'
    matches = re.findall(pattern, text)
    return [round(float(match), 4) for match in matches if float(match) <= 1.1]


class LLMQA(ModelForSemiStructQA):
    
    def __init__(self,
                 database, 
                 model_name,
                 query_emb_dir, 
                 candidates_emb_dir,
                 sim_weight=0.1,
                 max_k=100
                 ):
        '''
        Answer the query by GPT model.
        Args:
            database (src.benchmarks.semistruct.SemiStruct): database
            organization (str): openai organization
            api_key (str): openai api_key
            model_name (str): model name
            mode (str): one of ['selection', 'yes/no', 'generation']        
        '''
        
        super(LLMQA, self).__init__(database)
        self.max_k = max_k
        self.model_name = model_name
        self.sim_weight = sim_weight

        self.query_emb_dir = query_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        self.parent_vss = VSS(database, query_emb_dir, candidates_emb_dir)

    def forward(self, 
                query,
                query_id=None,
                **kwargs: Any):
        
        if query_id is None:
            query_emb = get_ada_embedding(query)
        else:
            query_emb_path = osp.join(self.query_emb_dir, f'query_{query_id}.pt')
            if os.path.exists(query_emb_path):
                query_emb = torch.load(query_emb_path).view(1, -1)
            else:
                query_emb = get_ada_embedding(query)
                torch.save(query_emb, query_emb_path)

        initial_score_dict = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())
        # get the ids with top k highest scores
        top_k_idx = torch.topk(torch.FloatTensor(node_scores),
                               min(self.max_k, len(node_scores)),
                               dim=-1).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]
        cand_len = len(top_k_node_ids)
        pred_dict = {}
        for id, node_id in enumerate(top_k_node_ids):
            node_type = self.database.get_node_type_by_id(node_id)
            prompt = (
                f'You are a helpful assistant that examines if a {node_type} satisfies a given query and assign a score from 0.0 to 1.0 based on the degree of satisfaction. If the {node_type} does not satisfy the query, the score should be 0.0. If there exists explicit and strong evidence supporting that {node_type} satisfies the query, the score should be 1.0. If partial evidence or weak evidence exists, the score should be between 0.0 and 1.0.\n'
                f'Here is the query:\n\"{query}\"\n'
                f'Here is the information about the {node_type}:\n' +
                self.database.get_doc_info(node_id, add_rel=True) + '\n\n' +
                f'Please score the {node_type} based on how well it satisfies the query. Only output the floating point score without anything else. '
                f'The numeric score of this {node_type} is: '
                )
            redo_flag = True
            while redo_flag:    
                if 'gpt' in self.model_name:
                    answer = get_gpt_output(prompt, self.model_name)
                elif 'claude' in self.model_name:
                    answer = complete_text_claude(prompt, self.model_name)
                answer = find_floating_number(answer)
                if len(answer) == 1:
                    redo_flag = False
                    answer = answer[0]
                else:
                    print('answer length not 1, redoing...')
            gpt_score = float(answer)
            sim_score = (cand_len - id) / cand_len
            score = gpt_score + self.sim_weight * sim_score
            pred_dict[node_id] = score
        return pred_dict
        
    