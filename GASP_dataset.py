import numpy as np
import networkx as nx
import os

from joblib import Parallel, delayed

import torch
# from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph

class GASPData(Dataset):
    def __init__(self, root, num_cpus, cutoff=5, force_process=False, surface_subgraph_hops=None):
        self.cutoff=cutoff
        self.force_process=force_process
        self.num_cpus = num_cpus
        self.hops = surface_subgraph_hops
        
        if not os.path.isdir(root + '/processed'):
            os.mkdir(root + '/processed')
        
        super().__init__(root, None, None)

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_dir))
    
    @property
    def processed_file_names(self):
        return sorted(os.listdir(self.processed_dir))
    
    def len(self):
        return len(self.raw_file_names)

    def process_helper(self, processed_dir, raw_path, i, cutoff):
        try:
            arr = np.load(raw_path, allow_pickle=True)
        except Exception as e:
            print("Error loading file: {}".format(raw_path), flush=True)
            return
        try:
            adj_matrix = arr['adj_matrix'][()]
            nonzero_mask = np.array(adj_matrix[adj_matrix.nonzero()]> cutoff )[0]
        except Exception as e:
            print(raw_path, flush=True)
            raise Exception from e
        
        # Remove values higher than distance cutoff
        rows = adj_matrix.nonzero()[0][nonzero_mask]
        cols = adj_matrix.nonzero()[1][nonzero_mask]
        adj_matrix[rows,cols] = 0
        adj_matrix.eliminate_zeros()
        
        # Build a graph from the sparse adjacency matrix
        G = nx.convert_matrix.from_scipy_sparse_matrix(adj_matrix)
        nx.set_edge_attributes(G, [0,0,0,0,1,0], "bond_type")              # Set all edge types to 'null' one-hot
        nx.set_edge_attributes(G, arr['edge_attributes'].item())           # Set all edge types to value in file
    
        # Calculate degree values
        degrees = np.array([list(dict(G.degree()).values())]).T

        # Get a COO edge_list
        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
        
        # edge_attr to represents the edge weights 
        edge_attr = torch.FloatTensor([[(cutoff - G[edge[0].item()][edge[1].item()]['weight'])/cutoff] + G[edge[0].item()][edge[1].item()]['bond_type'] for edge in edge_index.T])          
        
        # Convert Labels from one-hot to 1D target
        distance_to_ligand = arr['ligand_distance_array']
        y = torch.FloatTensor(distance_to_ligand)
        
        graph = Data(x=torch.FloatTensor(np.concatenate((arr['feature_matrix'], degrees), axis=1)), edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph.atom_index = torch.arange(graph.num_nodes)
        sasa = torch.FloatTensor(arr['SASA_array'])
        graph.surf_mask = sasa > 1e-4

        if self.hops is not None:
            # inducing the subgraph k hops away from the surface
            sub_nodes, _, _, _ = k_hop_subgraph(graph.atom_index[graph.surf_mask], self.hops, graph.edge_index)
            graph = graph.subgraph(sub_nodes)

        torch.save(graph, os.path.join(processed_dir, "data_{}.pt".format(i))) 

    def process(self):
        Parallel(n_jobs=self.num_cpus)(delayed(self.process_helper)(self.processed_dir, raw_path,i,cutoff=self.cutoff) for i, raw_path in enumerate(sorted(self.raw_paths)))
        print("Finished Dataset Processing")

    def get(self,idx):
        # Returns a data object that represents the graph. See the following link for more details:
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
        try:
            return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))), self.raw_file_names[idx]
        except Exception as e:
            print("Failed Loading File {}/data_{}.pt".format(self.processed_dir,idx), flush=True)
            print(self.cutoff)
            self.process_helper(self.processed_dir, self.raw_paths[idx], idx, cutoff=self.cutoff)
            return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))), self.raw_file_names[idx]