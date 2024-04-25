import os.path as osp
from pyvis.network import Network
import torch
import numpy as np
from src.tools.graph import k_hop_subgraph
from src.tools.node import Node, register_node
from torch_geometric.utils import to_undirected, is_undirected
color_types = ['#97c2fc', 'lightgreen', 'lightpink', 'lightpurple']


class SemiStructureKB:
    def __init__(self, node_info, edge_index, 
                 node_type_dict=None, 
                 edge_type_dict=None, 
                 node_types=None, edge_types=None, 
                 indirected=True, **kwargs):
        """
        A abstract dataset for semistructure data

        Args: 
            node_info (Dict[dict]): A meta dictionary, where each key is node ID and each value is a dictionary 
                                    containing information about the corresponding node. 
                                    The dictionary can be in arbitrary structure (e.g., hierarchical).
            
            node_types (torch.LongTensor): node types
            
            node_type_dict (torch.LongTensor): A meta dictionary, where each key is node ID (if node_types is None) or node type 
                                               (if node_types is not None) and each value dictionary contains information about 
                                               the node of the node type.
            
            edge_index (torch.LongTensor): edge index in the pyg format.
            
            edge_types (torch.LongTensor): edge types
            
            edge_type_dict (List[dict]): A meta dictionary, where each key is edge ID (if edge_types is None) or edge type 
                                    (if edge_types is not None) and each value dictionary contains information about 
                                    the edge of the edge type.
        """
        self.node_info = node_info
        self.edge_index = edge_index
        self.edge_type_dict = edge_type_dict
        self.node_type_dict = node_type_dict
        self.node_types = node_types
        self.edge_types = edge_types
        
        if indirected and not is_undirected(self.edge_index):
            self.edge_index, self.edge_types = to_undirected(self.edge_index, self.edge_types, 
                                                             num_nodes=self.num_nodes(), reduce='mean')
            self.edge_types = self.edge_types.long()
            
        if hasattr(self, 'candidate_types'):
            self.candidate_ids = self.get_candidate_ids()
        else:
            self.candidate_ids = [i for i in range(len(self.node_info))]
        self.num_candidates = len(self.candidate_ids)
        self._build_sparse_adj()    

    def __len__(self) -> int:
        return len(self.node_info)
        
    def __getitem__(self, idx):
        idx = int(idx)
        node = Node()
        register_node(node, self.node_info[idx])
        return node

    def get_doc_info(self, idx, 
                     add_rel=False, compact=False) -> str:
        '''
        Return a text document containing information about the node.    
        Args:
            idx (int): node index
            add_rel (bool): whether to add relational information explicitly
            compact (bool): whether to compact the text
        '''
        raise NotImplementedError

    def _build_sparse_adj(self):
        '''
        Build the sparse adjacency matrix.
        '''
        self.sparse_adj = torch.sparse.FloatTensor(self.edge_index, 
                                                   torch.ones(self.edge_index.shape[1]), 
                                                   torch.Size([self.num_nodes(), self.num_nodes()]))
        self.sparse_adj_by_type = {}
        for edge_type in self.rel_type_lst():
            edge_idx = torch.arange(self.num_edges())[self.edge_types == self.edge_type2id(edge_type)]
            self.sparse_adj_by_type[edge_type] = torch.sparse.FloatTensor(self.edge_index[:, edge_idx], 
                                                                          torch.ones(edge_idx.shape[0]), 
                                                                          torch.Size([self.num_nodes(), self.num_nodes()]))

    def get_rel_info(self, idx, rel_type=None) -> str:
        '''
        Return a text document containing information about the node.    
        Args:
            idx (int): node index
            add_rel (bool): whether to add relational information explicitly
            compact (bool): whether to compact the text
        '''
        raise NotImplementedError
    
    def get_candidate_ids(self) -> list:
        '''
        Get the candidate IDs.
        '''
        assert hasattr(self, 'candidate_types')
        candidate_ids = np.concatenate([self.get_node_ids_by_type(candidate_type) for candidate_type in self.candidate_types]).tolist()
        candidate_ids.sort()
        return candidate_ids
    
    def num_nodes(self, node_type_id=None):
        if node_type_id is None:
            return len(self.node_types)
        else:
            return sum(self.node_types == node_type_id)
    
    def num_edges(self, node_type_id=None):
        if node_type_id is None:
            return len(self.edge_types)
        else:
            return sum(self.edge_types == node_type_id)
    
    def rel_type_lst(self):
        return list(self.edge_type_dict.values())
    
    def node_type_lst(self):
        return list(self.node_type_dict.values())
    
    def node_attr_dict(self):
        raise NotImplementedError
    
    def is_rel_type(self, edge_type: str):
        return edge_type in self.rel_type_lst()
    
    def edge_type2id(self, edge_type: str) -> int:
        '''
        Get the edge type ID given the edge type.
        '''
        try:
            idx = list(self.edge_type_dict.values()).index(edge_type)
        except:
            raise ValueError(f"Edge type {edge_type} not found")
        return list(self.edge_type_dict.keys())[idx]
    
    def node_type2id(self, node_type: str) -> int:
        '''
        Get the node type ID given the node type.
        '''
        try:
            idx = list(self.node_type_dict.values()).index(node_type)
        except:
            raise ValueError(f"Node type {node_type} not found")
        return list(self.node_type_dict.keys())[idx]
    
    def get_node_type_by_id(self, node_id: int) -> str:
        '''
        Get the node type given the node ID.
        '''
        return self.node_type_dict[self.node_types[node_id].item()]
    
    def get_edge_type_by_id(self, edge_id: int) -> str:
        '''
        Get the edge type given the edge ID.
        '''
        return self.edge_type_dict[self.edge_types[edge_id].item()]

    def get_node_ids_by_type(self, node_type: str) -> list:
        '''
        Get the node IDs given the node type.
        '''
        return torch.arange(self.num_nodes())[self.node_types == self.node_type2id(node_type)].tolist() 
    
    def get_node_ids_by_value(self, node_type, key, value) -> list:
        '''
        Get the node IDs given the node type and the value of a specific attribute.
        '''
        ids = self.get_node_ids_by_type(node_type)
        indices = []
        for idx in ids:
            if hasattr(self[idx], key) and getattr(self[idx], key) == value:
                indices.append(idx)
        return indices
    
    def get_edge_ids_by_type(self, edge_type: str) -> list:
        '''
        Get the edge IDs given the edge type.
        '''
        return torch.arange(self.num_edges())[self.edge_types == self.edge_type2id(edge_type)].tolist()
    
    def sample_paths(self, node_types: list, edge_types: list, start_node_id=None, size=1) -> list:
        '''
        Sample paths give the node types and edge types.
        Use "*" to indicate any edge type.
        '''
        assert len(node_types) == len(edge_types) + 1
        for i in range(len(edge_types)):
            if edge_types[i] == "*":
                continue
            _tuple = (node_types[i], edge_types[i], node_types[i+1])
            assert _tuple in self.get_tuples(), f"{_tuple} invalid"

        paths = []
        while len(paths) < size:
            p = []
            for i in range(len(node_types)):
                if i == 0:
                    node_idx = start_node_id if not start_node_id is None else \
                               np.random.choice(self.get_node_ids_by_type(node_types[i]))
                else:
                    # neighbor_nodes = self.get_neighbor_nodes(node_idx, edge_types[i-1], direction='in-and-out')
                    neighbor_nodes = self.get_neighbor_nodes(node_idx, edge_types[i-1])
                    neighbor_nodes = torch.LongTensor(neighbor_nodes)
                    node_type_id = list(self.node_type_dict.keys())[list(self.node_type_dict.values()).index(node_types[1])]
                    neighbor_nodes = neighbor_nodes[self.node_types[neighbor_nodes] == node_type_id]
                    neighbor_nodes = neighbor_nodes.tolist()
                    if len(neighbor_nodes) == 0:
                        if i == 1 and not start_node_id is None:
                            return []
                        else:
                            break
                    node_idx = np.random.choice(neighbor_nodes)
                p.append(node_idx)
                
                if len(p) == len(node_types):
                    paths.append(p)
                
        return paths
    
    def get_all_paths(self, start_node_id: int, 
                      node_types: list, edge_types: list, 
                      max_num=None, direction='in-and-out') -> list:
        '''
        Sample paths give the node types and edge types.
        Use "*" to indicate any edge type.
        '''
        assert len(node_types) == len(edge_types) + 1

        paths = []
        # neighbor_nodes = self.get_neighbor_nodes(start_node_id, edge_types[0], direction=direction)
        neighbor_nodes = self.get_neighbor_nodes(start_node_id, edge_types[0])
        neighbor_nodes = torch.LongTensor(neighbor_nodes)
        node_type_id = list(self.node_type_dict.keys())[list(self.node_type_dict.values()).index(node_types[1])]

        neighbor_nodes = neighbor_nodes[self.node_types[neighbor_nodes] == node_type_id]
        neighbor_nodes = neighbor_nodes.tolist()

        if len(neighbor_nodes) == 0:
            # print(f'{start_node_id} => No neighbor nodes | len(node_types)={len(node_types)}')
            return []
        elif len(node_types) == 2:
            return [[start_node_id, node_idx] for node_idx in neighbor_nodes]
        else:
            # print(f'Iterating over # {len(neighbor_nodes)} neighbors')
            for iter_start_node_id in neighbor_nodes:
                subpaths = self.get_all_paths(iter_start_node_id, node_types[1:], edge_types[1:])
                if len(subpaths) == 0:
                    continue
                for subpath in subpaths:
                    paths.append([start_node_id] + subpath)
                # print((iter_start_node_id, node_types[1:], edge_types[1:]), '==> subpaths #', len(subpaths), ' | Total #', len(paths)) 
                if not max_num is None and len(paths) > max_num:
                    print('max_num reached')
                    return []
            # print('--------------Finished iterating--------------')
        return paths
    
    def get_tuples(self) -> list:
        '''
        Get all possible tuples of node types and edge types.
        '''
        col, row = self.edge_index.tolist()
        edge_types = self.edge_types.tolist()
        col_types, row_types = self.node_types[col].tolist(), self.node_types[row].tolist()
        tuples_by_id = set([(n_i, e, n_j) for n_i, e, n_j in zip(col_types, edge_types, row_types)])
        tuples = []
        for n_i, e, n_j in tuples_by_id:
            tuples.append((self.node_type_dict[n_i], self.edge_type_dict[e], self.node_type_dict[n_j]))
        tuples = list(set(tuples))
        tuples.sort()
        return tuples

    def get_neighbor_nodes(self, idx, edge_type: str = "*") -> list:
        '''
        Get the neighbor nodes given the node ID and the edge type.
        
        Args:
            idx (int): node index
            edge_type (str): edge type, use "*" to indicate any edge type.
        '''
        if edge_type == "*":
            neighbor_nodes = self.sparse_adj[idx].coalesce().indices().view(-1).tolist()
        else:
            neighbor_nodes = self.sparse_adj_by_type[edge_type][idx].coalesce().indices().view(-1).tolist()
        return neighbor_nodes
    
    def k_hop_neighbor(self, node_idx, num_hops, **kwargs):
        subset, edge_index, _, edge_mask = k_hop_subgraph(node_idx, 
                                                          num_hops, 
                                                          self.edge_index, 
                                                          num_nodes=self.num_nodes(), 
                                                          flow='bidirectional', 
                                                          **kwargs)
        node_types = self.node_types[subset]
        edge_types = self.edge_types[edge_mask]
        return subset, edge_index, node_types, edge_types
        
    def visualize(self, path='.'):
        net = Network()
        for idx in range(self.num_nodes()):
            try:
                net.add_node(idx, label=getattr(self[idx], 
                                                self.node_type_dict[self.node_types[idx].item()])[:1], 
                             color=color_types[self.node_types[idx].item()]
                             )
            except:
                net.add_node(idx, 
                             label=getattr(self[idx], 'title')[:1], 
                             color=color_types[self.node_types[idx].item()]
                             )
                
        for idx in range(self.num_edges()):
            net.add_edge(self.edge_index[0][idx].item(), 
                         self.edge_index[1][idx].item(), 
                         color=color_types[self.edge_types[idx].item()])
        net.toggle_physics(True)
        net.show(osp.join(path, 'nodes.html'), notebook=False)