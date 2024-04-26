import torch
from src.tools.api import get_openai_embedding


def get_top_k_indices(emb: torch.FloatTensor, 
                      candidate_embs: torch.FloatTensor, 
                      return_similarity=False, k=-1) -> list:
    '''
    Args:
        emb (torch.Tensor): embedding of the query
        candidate_embs (torch.Tensor): embeddings of the candidates
        k (int): number of candidates to return. 
        
        If k <= 0, rank all candidates
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = torch.matmul(emb.to(device), candidate_embs.to(device).T).cpu().view(-1)
    if k > 0:
        indices = torch.topk(sim, 
                             k=min(k, len(sim)), 
                             dim=-1, sorted=True).indices.view(-1).cpu()
    else:
        indices = torch.argsort(sim, dim=-1, descending=True).view(-1).cpu()
    indices = indices.tolist()
    
    if return_similarity:
        return indices, sim[indices]
    return indices


def sentence_emb_similarity(s1, s2):
    '''
    Args:
        s1 (str): sentence 1
        s2 (str): sentence 2
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb1 = get_openai_embedding(s1).to(device)
    emb2 = get_openai_embedding(s2).to(device)
    return torch.matmul(emb1, emb2.T).view(-1).cpu()


def normalize(x: torch.FloatTensor) -> torch.FloatTensor:
    '''
    Args:
        x (torch.Tensor): tensor to normalize
    '''
    return (x - x.min()) / (x.max() - x.min())

