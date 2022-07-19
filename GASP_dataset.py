import numpy as np
import networkx as nx
import os

from joblib import Parallel, delayed

import torch
# from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_scipy_sparse_matrix

class GASPData(Dataset):
    def __init__(self, root, num_cpus, cutoff=5, force_process=False):
        self.cutoff=cutoff
        self.force_process=force_process
        self.num_cpus = num_cpus
        
        if not os.path.isdir(root + '/processed'):
            os.mkdir(root + '/processed')
        
        super().__init__(root, None, None)

        # self.proceesed_dir = processed_dir
        # self.processed_dir = "processed_GASP_Dataset/{}".format(mode)

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_dir))
    
    @property
    def processed_file_names(self):
        return sorted(os.listdir(self.processed_dir)) #[filename for filename in self.processed_dir]
    
    def len(self):
        return len(self.raw_file_names)

    def process_helper(self, processed_dir, raw_path, i, cutoff):
        try:
            arr = np.load(raw_path, allow_pickle=True)
        except Exception as e:
            print("Error loading file: {}".format(raw_path), flush=True)
            return
        try:
            adj_matrix = arr['adj_matrix'][()]#.tocsr() #changed this in the new parsing so it comes in CSR
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
        nx.set_edge_attributes(G, [0,0,0,0,0,1,0], "bond_type")         # Set all edge types to 'null' one-hot
        nx.set_edge_attributes(G, arr['edge_attributes'].item())           # Set all edge types to value in file
    
        # Calculate degree values
        degrees = np.array([list(dict(G.degree()).values())]).T

        # Get a COO edge_list
        # edge_index = torch.LongTensor([[int(line.split()[0]), int(line.split()[1])] for line in nx.generate_edgelist(G, data=False)]).T
        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
        # print("0.5")
        # edge_attr to represents the edge weights 
        edge_attr = torch.FloatTensor([[(cutoff - G[edge[0].item()][edge[1].item()]['weight'])/cutoff] + G[edge[0].item()][edge[1].item()]['bond_type'] for edge in edge_index.T])          
        
        # Convert Labels from one-hot to 1D target
        y = torch.LongTensor([0 if label[0] == 1 else 1 for label in arr['class_array']] )
        # print("0.6")
        # print("TIME TO GET OBJ: {}".format(time.time()- start))
        to_save = Data(x=torch.FloatTensor(np.concatenate((arr['feature_matrix'], degrees), axis=1)), edge_index=edge_index, edge_attr=edge_attr, y=y)
        torch.save(to_save, os.path.join(processed_dir, "data_{}.pt".format(i))) 

    def process(self):
        # print(len(self.raw_file_names), len(self.processed_file_names))
        # print(self.raw_paths)
        # self.force_process or
        # if len(self.raw_file_names) != len(self.processed_file_names):
        #     print("Processing Dataset")
        Parallel(n_jobs=self.num_cpus)(delayed(self.process_helper)(self.processed_dir, raw_path,i,cutoff=self.cutoff) for i, raw_path in enumerate(sorted(self.raw_paths)))
        print("Finished Dataset Processing")

    def get(self,idx):
        # print("Getting Graph")
        # Returns a data object that represents the graph. See the following link for more details:
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
        try:
            return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))), self.raw_file_names[idx]
        except Exception as e:
            print("Failed Loading File {}/data_{}.pt".format(self.processed_dir,idx), flush=True)
            print(self.cutoff)
            self.process_helper(self.processed_dir, self.raw_paths[idx], idx, cutoff=self.cutoff)
            return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))), self.raw_file_names[idx]

# from torch_geometric.data import HeteroData

# class GASPData_Hetro(Dataset):
#     def __init__(self, root, num_cpus, cutoff=5, force_process=False):
#         self.cutoff=cutoff
#         self.force_process=force_process
#         self.num_cpus = num_cpus
#         super().__init__(root, None, None)

#         # self.proceesed_dir = processed_dir
#         # self.processed_dir = "processed_GASP_Dataset/{}".format(mode)

#     @property
#     def raw_file_names(self):
#         return sorted(os.listdir(self.raw_dir))
    
#     @property
#     def processed_file_names(self):
#         return sorted(os.listdir(self.processed_dir)) #[filename for filename in self.processed_dir]
    
#     def len(self):
#         return len(self.raw_file_names)

#     def process_helper(self, processed_dir, raw_path, i, cutoff=4):
#         try:
#             arr = np.load(raw_path, allow_pickle=True)
#         except Exception as e:
#             print("Error loading file: {}".format(raw_path), flush=True)
#             return
#         try:
#             adj_matrix = arr['adj_matrix'][()]#.tocsr() #changed this in the new parsing so it comes in CSR
#             nonzero_mask = np.array(adj_matrix[adj_matrix.nonzero()]> cutoff )[0]
#         except Exception as e:
#             print(raw_path, flush=True)
#             raise Exception from e
#         # Remove values higher than distance cutoff
#         rows = adj_matrix.nonzero()[0][nonzero_mask]
#         cols = adj_matrix.nonzero()[1][nonzero_mask]
#         adj_matrix[rows,cols] = 0
#         adj_matrix.eliminate_zeros()
#         # Build a graph from the sparse adjacency matrix
#         G = nx.convert_matrix.from_scipy_sparse_matrix(adj_matrix)
#         nx.set_edge_attributes(G, [0,0,0,0,0,1,0], "bond_type")         # Set all edge types to 'null' one-hot
#         nx.set_edge_attributes(G, arr['edge_attributes'].item())           # Set all edge types to value in file
    
    
#         # Calculate degree values
#         degrees = np.array([list(dict(G.degree()).values())]).T

#         # Get a COO edge_list
#         # edge_index = torch.LongTensor([[int(line.split()[0]), int(line.split()[1])] for line in nx.generate_edgelist(G, data=False)]).T
#         edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
#         # edge_attr to represents the edge weights 
#         edge_attr = torch.FloatTensor([[(cutoff - G[edge[0].item()][edge[1].item()]['weight'])/cutoff] + G[edge[0].item()][edge[1].item()]['bond_type'] for edge in edge_index.T])          
        
#         # Get bond only attributes
#         bond_edge_attr  = torch.FloatTensor([[(cutoff - G[edge[0].item()][edge[1].item()]['weight'])/cutoff] + G[edge[0].item()][edge[1].item()]['bond_type']  for edge in edge_index.T if G[edge[0].item()][edge[1].item()]['bond_type'][-2] != 1])
#         bond_edge_index = torch.FloatTensor([[edge[0], edge[1]] for edge in edge_index.T if G[edge[0].item()][edge[1].item()]['bond_type'][-2] != 1])
        
#         to_save = HeteroData()
#         to_save['atom'].x                                 = torch.FloatTensor(np.concatenate((arr['feature_matrix'], degrees), axis=1))
#         to_save['atom', 'bonded_to', 'atom'].edge_index   = bond_edge_index.T
#         to_save['atom', 'bonded_to', 'atom'].edge_attr    = bond_edge_attr
#         # to_save['atom', 'bonded_to', 'atom'].edge_index, to_save['atom', 'bonded_to', 'atom'].edge_attr = add_self_loops(to_save['atom', 'bonded_to', 'atom'].edge_index, to_save['atom', 'bonded_to', 'atom'].edge_attr, fill_value=[0,0,0,0,0,0,1,0])
#         to_save['atom', 'distance_to', 'atom'].edge_index = edge_index
#         to_save['atom', 'distance_to', 'atom'].edge_attr  = edge_attr
#         to_save['atom'].y                                 = torch.LongTensor([0 if label[0] == 1 else 1 for label in arr['class_array']] )
        
#         torch.save(to_save, os.path.join(processed_dir, "data_{}.pt".format(i))) 

#     def process(self):
#         # print(len(self.raw_file_names), len(self.processed_file_names))
#         # print(self.raw_paths)
#         # self.force_process or
#         # if len(self.raw_file_names) != len(self.processed_file_names):
#         #     print("Processing Dataset")
#         Parallel(n_jobs=self.num_cpus)(delayed(self.process_helper)(self.processed_dir, raw_path,i,cutoff=self.cutoff) for i, raw_path in enumerate(sorted(self.raw_paths)))
#         print("Finished Dataset Processing")

#     def get(self,idx):
#         # print("Getting Graph")
#         # Returns a hetro data object that represents the graph. See the following link for more details:
#         # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
#         try:
#             return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))), self.raw_file_names[idx]
#         except Exception as e:
#             print("Failed Loading File {}/data_{}.pt".format(self.processed_dir,idx), flush=True)
#             print(self.cutoff)
#             self.process_helper(self.processed_dir, self.raw_paths[idx], idx, cutoff=self.cutoff)
#             return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))), self.raw_file_names[idx]
