import numpy as np
import torch
from torch_geometric.utils import is_undirected, to_undirected

from stark_qa.tools.graph import k_hop_subgraph
from stark_qa.tools.node import Node, register_node


class SKB:
    def __init__(self, 
                 node_info: dict, 
                 edge_index: torch.LongTensor, 
                 node_type_dict=None, 
                 edge_type_dict=None, 
                 node_types=None, 
                 edge_types=None, 
                 indirected=True, 
                 **kwargs):
        """
        Initialize the SKB dataset for semi-structured data.

        Args:
            node_info (Dict[dict]): A meta dictionary where each key is node ID and each value is a dictionary 
                                    containing information about the corresponding node.
            edge_index (torch.LongTensor): Edge index in the PyG format.
            node_type_dict (dict): Meta dictionary where each key is node ID (if node_types is None) 
                                   or node type (if node_types is not None) and each value dictionary 
                                   contains information about the node of the node type.
            edge_type_dict (dict): Meta dictionary where each key is edge ID (if edge_types is None) 
                                   or edge type (if edge_types is not None) and each value dictionary 
                                   contains information about the edge of the edge type.
            node_types (torch.LongTensor): Node types.
            edge_types (torch.LongTensor): Edge types.
            indirected (bool): Whether to make the graph undirected.
            **kwargs: Additional arguments.
        """
        self.node_info = node_info
        self.edge_index = edge_index
        self.edge_type_dict = edge_type_dict
        self.node_type_dict = node_type_dict
        self.node_types = node_types
        self.edge_types = edge_types
        
        if indirected and not is_undirected(self.edge_index):
            self.edge_index, self.edge_types = to_undirected(
                self.edge_index, self.edge_types, 
                num_nodes=self.num_nodes(), reduce='mean'
            )
            self.edge_types = self.edge_types.long()
        
        self.candidate_ids = self.get_candidate_ids() if hasattr(self, 'candidate_types') else [i for i in range(len(self.node_info))]
        self.num_candidates = len(self.candidate_ids)
        self._build_sparse_adj()    

    def __len__(self):
        """Return the number of nodes."""
        return len(self.node_info)
        
    def __getitem__(self, idx):
        """Get the node by index."""
        idx = int(idx)
        node = Node()
        register_node(node, self.node_info[idx])
        return node

    def get_doc_info(self, idx, add_rel=False, compact=False) -> str:
        """
        Return a text document containing information about the node.

        Args:
            idx (int): Node index.
            add_rel (bool): Whether to add relational information explicitly.
            compact (bool): Whether to compact the text.
        """
        raise NotImplementedError

    def _build_sparse_adj(self):
        """Build the sparse adjacency matrix."""
        self.sparse_adj = torch.sparse.FloatTensor(
            self.edge_index, 
            torch.ones(self.edge_index.shape[1]), 
            torch.Size([self.num_nodes(), self.num_nodes()])
        )
        self.sparse_adj_by_type = {}
        for edge_type in self.rel_type_lst():
            edge_idx = torch.arange(self.num_edges())[self.edge_types == self.edge_type2id(edge_type)]
            self.sparse_adj_by_type[edge_type] = torch.sparse.FloatTensor(
                self.edge_index[:, edge_idx], 
                torch.ones(edge_idx.shape[0]), 
                torch.Size([self.num_nodes(), self.num_nodes()])
            )

    def get_rel_info(self, idx, rel_type=None) -> str:
        """
        Return a text document containing information about the node.

        Args:
            idx (int): Node index.
            rel_type (str, optional): Relation type.
        """
        raise NotImplementedError
    
    def get_candidate_ids(self) -> list:
        """Get the candidate IDs."""
        assert hasattr(self, 'candidate_types')
        candidate_ids = np.concatenate(
            [self.get_node_ids_by_type(candidate_type) for candidate_type in self.candidate_types]
        ).tolist()
        candidate_ids.sort()
        return candidate_ids
    
    def num_nodes(self, node_type_id=None):
        """Return the number of nodes."""
        return len(self.node_types) if node_type_id is None else sum(self.node_types == node_type_id)
    
    def num_edges(self, node_type_id=None):
        """Return the number of edges."""
        return len(self.edge_types) if node_type_id is None else sum(self.edge_types == node_type_id)
    
    def rel_type_lst(self):
        """Return the list of relation types."""
        return list(self.edge_type_dict.values())
    
    def node_type_lst(self):
        """Return the list of node types."""
        return list(self.node_type_dict.values())
    
    def node_attr_dict(self):
        """Return the node attribute dictionary."""
        raise NotImplementedError
    
    def is_rel_type(self, edge_type: str):
        """Check if the edge type is a relation type."""
        return edge_type in self.rel_type_lst()
    
    def edge_type2id(self, edge_type: str) -> int:
        """Get the edge type ID given the edge type."""
        try:
            idx = list(self.edge_type_dict.values()).index(edge_type)
        except ValueError:
            raise ValueError(f"Edge type {edge_type} not found")
        return list(self.edge_type_dict.keys())[idx]
    
    def node_type2id(self, node_type: str) -> int:
        """Get the node type ID given the node type."""
        try:
            idx = list(self.node_type_dict.values()).index(node_type)
        except ValueError:
            raise ValueError(f"Node type {node_type} not found")
        return list(self.node_type_dict.keys())[idx]
    
    def get_node_type_by_id(self, node_id: int) -> str:
        """Get the node type given the node ID."""
        return self.node_type_dict[self.node_types[node_id].item()]
    
    def get_edge_type_by_id(self, edge_id: int) -> str:
        """Get the edge type given the edge ID."""
        return self.edge_type_dict[self.edge_types[edge_id].item()]

    def get_node_ids_by_type(self, node_type: str) -> list:
        """Get the node IDs given the node type."""
        return torch.arange(self.num_nodes())[self.node_types == self.node_type2id(node_type)].tolist()
    
    def get_node_ids_by_value(self, node_type, key, value) -> list:
        """Get the node IDs given the node type and the value of a specific attribute."""
        ids = self.get_node_ids_by_type(node_type)
        indices = [idx for idx in ids if hasattr(self[idx], key) and getattr(self[idx], key) == value]
        return indices
    
    def get_edge_ids_by_type(self, edge_type: str) -> list:
        """Get the edge IDs given the edge type."""
        return torch.arange(self.num_edges())[self.edge_types == self.edge_type2id(edge_type)].tolist()
    
    def sample_paths(self, node_types: list, edge_types: list, start_node_id=None, size=1) -> list:
        """
        Sample paths given the node types and edge types. Use "*" to indicate any edge type.
        """
        assert len(node_types) == len(edge_types) + 1
        for i in range(len(edge_types)):
            if edge_types[i] != "*":
                assert (node_types[i], edge_types[i], node_types[i+1]) in self.get_tuples(), \
                       f"{(node_types[i], edge_types[i], node_types[i+1])} invalid"

        paths = []
        while len(paths) < size:
            p = []
            for i in range(len(node_types)):
                if i == 0:
                    node_idx = start_node_id if start_node_id is not None else np.random.choice(self.get_node_ids_by_type(node_types[i]))
                else:
                    neighbor_nodes = self.get_neighbor_nodes(node_idx, edge_types[i-1])
                    neighbor_nodes = torch.LongTensor(neighbor_nodes)
                    node_type_id = self.node_type2id(node_types[i])
                    neighbor_nodes = neighbor_nodes[self.node_types[neighbor_nodes] == node_type_id].tolist()
                    if len(neighbor_nodes) == 0:
                        if i == 1 and start_node_id is not None:
                            return []
                        else:
                            break
                    node_idx = np.random.choice(neighbor_nodes)
                p.append(node_idx)
                
                if len(p) == len(node_types):
                    paths.append(p)
                
        return paths
    
    def get_all_paths(self, start_node_id: int, node_types: list, edge_types: list, 
                      max_num=None, direction='in-and-out') -> list:
        """
        Get all paths given the node types and edge types. Use "*" to indicate any edge type.
        """
        assert len(node_types) == len(edge_types) + 1

        paths = []
        neighbor_nodes = self.get_neighbor_nodes(start_node_id, edge_types[0])
        neighbor_nodes = torch.LongTensor(neighbor_nodes)
        node_type_id = self.node_type2id(node_types[1])
        neighbor_nodes = neighbor_nodes[self.node_types[neighbor_nodes] == node_type_id].tolist()

        if len(neighbor_nodes) == 0:
            return []
        elif len(node_types) == 2:
            return [[start_node_id, node_idx] for node_idx in neighbor_nodes]
        else:
            for iter_start_node_id in neighbor_nodes:
                subpaths = self.get_all_paths(iter_start_node_id, node_types[1:], edge_types[1:])
                if subpaths:
                    for subpath in subpaths:
                        paths.append([start_node_id] + subpath)
                if max_num is not None and len(paths) > max_num:
                    return paths
        return paths
    
    def get_tuples(self) -> list:
        """Get all possible tuples of node types and edge types."""
        col, row = self.edge_index.tolist()
        edge_types = self.edge_types.tolist()
        col_types, row_types = self.node_types[col].tolist(), self.node_types[row].tolist()
        tuples_by_id = set([(n_i, e, n_j) for n_i, e, n_j in zip(col_types, edge_types, row_types)])
        tuples = [(self.node_type_dict[n_i], self.edge_type_dict[e], self.node_type_dict[n_j]) for n_i, e, n_j in tuples_by_id]
        tuples = list(set(tuples))
        tuples.sort()
        return tuples

    def get_neighbor_nodes(self, idx, edge_type: str = "*") -> list:
        """
        Get the neighbor nodes given the node ID and the edge type.
        
        Args:
            idx (int): Node index.
            edge_type (str): Edge type, use "*" to indicate any edge type.
        """
        if edge_type == "*":
            neighbor_nodes = self.sparse_adj[idx].coalesce().indices().view(-1).tolist()
        else:
            neighbor_nodes = self.sparse_adj_by_type[edge_type][idx].coalesce().indices().view(-1).tolist()
        return neighbor_nodes
    
    def k_hop_neighbor(self, node_idx, num_hops, **kwargs):
        """
        Get the k-hop neighbor subgraph.

        Args:
            node_idx (int): Node index.
            num_hops (int): Number of hops.
            **kwargs: Additional arguments.
        """
        subset, edge_index, _, edge_mask = k_hop_subgraph(
            node_idx, num_hops, self.edge_index, 
            num_nodes=self.num_nodes(), flow='bidirectional', **kwargs
        )
        node_types = self.node_types[subset]
        edge_types = self.edge_types[edge_mask]
        return subset, edge_index, node_types, edge_types
        
