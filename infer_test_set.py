import os
from networkx.generators import directed
import numpy as np
import scipy

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel
from torch_geometric.nn import GCNConv, PointConv, DataParallel, GATv2Conv
from torch_geometric.loader import DataLoader, NeighborLoader, DataListLoader, ClusterData, ClusterLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import dropout_adj, from_scipy_sparse_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef as mcc

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

prepend = str(os.getcwd())

model_path = prepend + "/trained_models/trained_model_1636421443.2471824_epoch_0"
device = 'cpu'
data_path = prepend + "/held_out/"
num_cpus = 4

class KLIFSData(Dataset):
    def __init__(self, paths, mode='train'):
        super(KLIFSData, self).__init__(None, None, None)
        if mode != 'train' and mode != 'test':
            raise ValueError("Expected dataloader mode 'train' or 'test' but got {}.".format(mode))
        self.mode = mode
        self.paths = paths
        self.len = len(paths)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        try:
            arr = np.load(self.paths[idx], allow_pickle=True)
        except Exception as e:
            print("Error loading file: {}".format(self.paths[idx]), flush=True)
            return
        try:
            adj_matrix = arr['adj_matrix'][()]#.tocsr() #changed this in the new parsing so it comes in CSR
            nonzero_mask = np.array(adj_matrix[adj_matrix.nonzero()]> 4 )[0]
        except Exception as e:
            print(self.paths[idx], flush=True)
            raise Exception from e
        # Remove values higher than distance cutoff
        rows = adj_matrix.nonzero()[0][nonzero_mask]
        cols = adj_matrix.nonzero()[1][nonzero_mask]
        adj_matrix[rows,cols] = 0
        adj_matrix.eliminate_zeros()
        # Build a graph from the sparse adjacency matrix
        G = nx.convert_matrix.from_scipy_sparse_matrix(adj_matrix)
        nx.set_edge_attributes(G, [0,0,0,0,0], "edge_type")         # Set all edge types to 'null' one-hot
        nx.set_edge_attributes(G, arr['edge_attributes'].item())           # Set all edge types to value in file
        # Calculate degree values
        degrees = np.array([list(dict(G.degree()).values())]).T
        # Get a COO edge_list
        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
        # edge_attr to represents the edge weights 
        edge_attr = torch.FloatTensor([[(4 - G[edge[0].item()][edge[1].item()]['weight'])/4] + G[edge[0].item()][edge[1].item()]['edge_type'] for edge in edge_index.T])          
        # Convert Labels from one-hot to 1D target
        y = torch.LongTensor([0 if label[0] == 1 else 1 for label in arr['class_array']] )
        # print("0.6")
        return Data(x=torch.FloatTensor(np.concatenate((arr['feature_matrix'], degrees), axis=1)), edge_index=edge_index, edge_attr=edge_attr, y=y)

class GATModel(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        # No need for bias in GAT Convs due to batch norms
        super(GATModel, self).__init__()
        self.preprocess1 = nn.Linear(input_dim, 128, bias=False)
        self.BN0 = BatchNorm(128)
        
        self.GAT_1 = GATv2Conv(128, 8, heads=8, bias=False)
        self.BN1 = BatchNorm(64)
        
        self.preprocess2 = nn.Linear(64, 48)
        self.GAT_2 = GATv2Conv(48, 6, heads=8, bias=False)
        self.BN2 = BatchNorm(48)

        self.preprocess3 = nn.Linear(48, 32)
        self.GAT_3 = GATv2Conv(32, 5, heads=5, bias=False)
        self.BN3 = BatchNorm(25)

        self.postprocess1 = nn.Linear(25, output_dim)

        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)
        
    def forward(self, input):
        x = self.BN0(self.elu(self.preprocess1(input.x)))
        
        x = self.BN1(self.GAT_1(x, input.edge_index))
        
        x = self.elu(self.preprocess2(x))
        x = self.BN2(self.GAT_2(x,input.edge_index))

        x = self.elu(self.preprocess3(x))
        x = self.BN3(self.GAT_3(x,input.edge_index))
        
        x = self.postprocess1(x)
        return x
    
model = GATModel(input_dim=40, output_dim=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

test_set = KLIFSData(np.array([data_path + x for x in os.listdir(data_path)]) , mode='test')
test_dataloader =  DataLoader(test_set, batch_size = 4, shuffle=True, pin_memory=True, num_workers=num_cpus)
 
test_epoch_loss = []
test_epoch_acc = []
test_epoch_mcc = []

model.eval()
with torch.no_grad():
    test_batch_loss = 0.0
    test_batch_acc = 0.0
    test_batch_mcc = 0.0
    for batch in test_dataloader:
        labels = batch.y

        out = model.forward(batch.to(device))
        loss = F.cross_entropy(out, batch.y)
        preds = np.argmax(out.detach().cpu().numpy(), axis=1)
        bl = loss.detach().cpu().item()

        ba = accuracy_score(labels, preds)
        bm = mcc(labels, preds)
            
        test_batch_loss += bl
        test_batch_acc  += ba
        test_batch_mcc  += bm
        print("Training Batch Loss:", bl)
        print("Training Batch Accu:", ba)
        print("Training Batch MCC:", bm)

        # writer.add_scalar('Batch_Loss/test', bl, test_batch_num)
        # writer.add_scalar('Batch_Acc/test',  ba,  test_batch_num)
        # writer.add_scalar('Batch_Acc/MCC',  bm,  test_batch_num)


    test_epoch_loss.append(test_batch_loss/len(test_dataloader))
    test_epoch_acc.append(test_batch_acc/len(test_dataloader))
    test_epoch_mcc.append(test_batch_mcc/len(test_dataloader))
    print("Test Loss: {}".format(test_epoch_loss[-1]))
    print("Test Accu: {}".format(test_epoch_acc[-1]))
    print("Test MCC: {}".format(test_epoch_mcc[-1]))
    # writer.add_scalar('Epoch_Loss/test', test_epoch_loss[-1], test_epoch_num)
    # writer.add_scalar('Epoch_Acc/test',  test_epoch_acc[-1],  test_epoch_num)
    # writer.add_scalar('Epoch_Acc/MCC',  test_epoch_mcc[-1],  test_epoch_num)
