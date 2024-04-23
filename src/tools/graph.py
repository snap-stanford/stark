import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import List, Optional, Tuple, Union
from torch import Tensor


def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
        Added bidirectional flow based on PyG's `k_hop_subgraph`.
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source', 'bidirectional']
    if flow == 'target_to_source':
        row, col = edge_index
    elif flow == 'source_to_target':
        col, row = edge_index
    else:
        col, row = torch.concat([edge_index, edge_index[[1, 0]]], dim=1)    

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    
    if flow == 'bidirectional':
        col, row = edge_index
        
    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        edge_index = relabel_graph(subset, edge_index, num_nodes)

    return subset, edge_index, inv, edge_mask


def relabel_graph(subset, edge_index, num_nodes):
    row, col = edge_index
    node_idx = row.new_full((num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    return edge_index
    