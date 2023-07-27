import numpy as np
import networkx as nx
import os

from joblib import Parallel, delayed

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph

class GASPData(Dataset):
    def __init__(self, root:str, num_cpus:int, cutoff:int=5, surface_subgraph_hops:int=None, sasa_threshold:float=1e-4):
        """A PyG dataset for protein graphs

        Parameters
        ----------
        root : str
            Path to the directory containing raw files. Processed files will be placed in the root/processed directory.
        num_cpus : int
            Number of cpus to be used during preprocessing.
        cutoff : int, optional
            Maximum length for edges in the protein graph, by default 5
        surface_subgraph_hops : int, optional
            The maximum number of edge hops away from a solvent exposed atom a node can be to be included in the subgraph. 
            When set to none, the full graph is used, by default None
        sasa_threshold : float, optional
            The minumum ammount of solvent accessible surface area to be considered a surface atom, by default 1e-4
        """        
        self.cutoff = cutoff
        self.num_cpus = num_cpus
        self.hops = surface_subgraph_hops
        self.sasa_thresold = sasa_threshold
        
        if not os.path.isdir(root + '/processed'):
            os.mkdir(root + '/processed')
        
        super().__init__(root, None, None)

    @property
    def raw_file_names(self):
        """:obj:`list` of :obj:`str`: List of file names in the raw directory. 
        
        The list is returned in sorted order, the same order that the directory is processed in.
        """
        return sorted(os.listdir(self.raw_dir))
    
    @property
    def processed_file_names(self):
        """:obj:`list` of :obj:`str`: List of file names in the processed directory. 
        
        The list is returned in sorted order, the same order that the directory is processed in.
        """
        return sorted(os.listdir(self.processed_dir))
    
    def len(self):
        """int: Returns the number of raw files associated with the dataset."""        
        return len(self.raw_file_names)

    def process_helper(self, processed_dir:str, raw_path:str, i:int, cutoff:float):
        """A helper function to process graphs in parallel. This method primarily transforms the data 
        from a numpy record array into a pytorch_geometric.data.Data object. Additionally, it removes 
        edges greater than cutoff, generates continuous labels for the data, and optionally creates the 
        induced subgraph.

        Parameters
        ----------
        processed_dir : str
            Path to output the data.
        raw_path : str
            The full path to the raw data file.
        i : int
            The integer index of this data point. Processed data points are saved as data_i.pt
        cutoff : float
            Maximum distance to be considered an edge in the graph

        Raises
        ------
        Exception
            Reraises exceptions from numpy when processing the numpy record array additionally
            printing the path to the relevant file into stdout.
        """        
        try:
            arr = np.load(raw_path, allow_pickle=True)                              # Load the unprocessed numpy record array.
        except Exception as e:
            print("Error loading file: {}".format(raw_path), flush=True)            # If there is an issue loading the file, we'll skip it 
            return
        try:
            adj_matrix = arr['adj_matrix'][()]
            nonzero_mask = np.array(adj_matrix[adj_matrix.nonzero()] > cutoff )[0]
        except Exception as e:
            print(raw_path, flush=True)                                             # If there is an issue loading the adj matrix
            raise Exception from e                                                  # print the file path and reraise the exception
        
        # Remove values higher than distance cutoff
        rows = adj_matrix.nonzero()[0][nonzero_mask]
        cols = adj_matrix.nonzero()[1][nonzero_mask]
        adj_matrix[rows,cols] = 0
        adj_matrix.eliminate_zeros()
        
        # Build a graph from the sparse adjacency matrix
        G = nx.convert_matrix.from_scipy_sparse_matrix(adj_matrix)
        nx.set_edge_attributes(G, [0,0,0,0,1,0], "bond_type")              # Set all edge types to 'null' one-hot by default
        nx.set_edge_attributes(G, arr['edge_attributes'].item())           # Overwrite edge types to value in file if available
    
        # Calculate degree values
        degrees = np.array([list(dict(G.degree()).values())]).T

        # Get a COO edge_list
        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
        
        # edge_attr to represents the edge weights 
        edge_attr = torch.FloatTensor([[(cutoff - G[edge[0].item()][edge[1].item()]['weight'])/cutoff] + G[edge[0].item()][edge[1].item()]['bond_type'] for edge in edge_index.T])          
        
        # Convert Labels from one-hot to 1D target
        distance_to_ligand = arr['ligand_distance_array']
        y = torch.FloatTensor(distance_to_ligand)

        # Properties for EGNN
        coords = torch.FloatTensor(arr['coords'])
        closest_ligand = torch.Tensor(arr['closest_ligand'])

        # Inialize a pytorch_geometric.data.Data object to store the data point
        graph = Data(x=torch.FloatTensor(np.concatenate((arr['feature_matrix'], degrees), axis=1)),
                    edge_index=edge_index, 
                    edge_attr=edge_attr,
                    y=y, 
                    coords=coords, 
                    closest_ligand=closest_ligand)
        graph.atom_index = torch.arange(graph.num_nodes)
        
        # Mask atoms with solvent accessible surface area lower than the sasa threshold.
        sasa = torch.FloatTensor(arr['SASA_array'])
        graph.surf_mask = sasa > self.sasa_thresold

        # Induce the subgraph k hops away from the surface
        if self.hops is not None:
            sub_nodes, _, _, _ = k_hop_subgraph(graph.atom_index[graph.surf_mask], self.hops, graph.edge_index)
            graph = graph.subgraph(sub_nodes)

        torch.save(graph, os.path.join(processed_dir, "data_{}.pt".format(i))) 

    def process(self):
        """Processes all raw files in the root directory into processed files stored in the processed directory.
        """        
        Parallel(n_jobs=self.num_cpus)(delayed(self.process_helper)(self.processed_dir, raw_path, i, cutoff=self.cutoff) for i, raw_path in enumerate(sorted(self.raw_paths)))
        print("Finished Dataset Processing")

    def get(self,idx:int):
        """Returns a data object that represents the protein graph. See the following link 
        for more details on the object: 
        https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data

        Parameters
        ----------
        idx : int
            Index of the data point to be retrieved 

        Returns
        -------
        pytorch_geometric.data.Data
            A data pytorch geometric data object representing the featurized protein graph.
        """        
        try:
            return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))), self.raw_file_names[idx]
        # If the processed datapoint fails to load, attempt to process it again from raw. If it fails again this will raise and Exception
        except Exception as e:
            print("Failed Loading File {}/data_{}.pt".format(self.processed_dir,idx), flush=True)
            print(self.cutoff)
            self.process_helper(self.processed_dir, self.raw_paths[idx], idx, cutoff=self.cutoff)
            return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))), self.raw_file_names[idx]
