import os
from networkx.generators import directed
import numpy as np
from joblib import Parallel, delayed

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

from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import roc_curve, auc

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

prepend = str(os.getcwd()) + "/trained_models/"

########################## Change Me To Change The Model ##########################
model_name = "trained_model_1640072931.267488_epoch_49"
model_path = prepend + model_name
###################################################################################
# Other Parameters
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

num_cpus = 4
print("The model will be using the following device:", device)
print("The model will be using {} cpus.".format(num_cpus))

class KLIFSData(Dataset):
    def __init__(self, root):
        super().__init__(root, None, None)
        # self.proceesed_dir = processed_dir
        # self.processed_dir = "processed_KLIFS_Dataset/{}".format(mode)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        return  os.listdir(self.processed_dir) #[filename for filename in self.processed_dir]
    
    def len(self):
        return len(self.raw_file_names)

    def process(self):
        # print(len(self.raw_file_names), len(self.processed_file_names))
        def process_helper(processed_dir, raw_path, i):
            try:
                arr = np.load(raw_path, allow_pickle=True)
            except Exception as e:
                print("Error loading file: {}".format(raw_path), flush=True)
                return
            try:
                adj_matrix = arr['adj_matrix'][()]#.tocsr() #changed this in the new parsing so it comes in CSR
                nonzero_mask = np.array(adj_matrix[adj_matrix.nonzero()]> 4 )[0]
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
            nx.set_edge_attributes(G, [0,0,0,0,0], "edge_type")         # Set all edge types to 'null' one-hot
            nx.set_edge_attributes(G, arr['edge_attributes'].item())           # Set all edge types to value in file
        
            # Calculate degree values
            degrees = np.array([list(dict(G.degree()).values())]).T

            # Get a COO edge_list
            # edge_index = torch.LongTensor([[int(line.split()[0]), int(line.split()[1])] for line in nx.generate_edgelist(G, data=False)]).T
            edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
            # print("0.5")
            # edge_attr to represents the edge weights 
            edge_attr = torch.FloatTensor([[(4 - G[edge[0].item()][edge[1].item()]['weight'])/4] + G[edge[0].item()][edge[1].item()]['edge_type'] for edge in edge_index.T])          
            
            # Convert Labels from one-hot to 1D target
            y = torch.LongTensor([0 if label[0] == 1 else 1 for label in arr['class_array']] )
            # print("0.6")
            # print("TIME TO GET OBJ: {}".format(time.time()- start))
            to_save = Data(x=torch.FloatTensor(np.concatenate((arr['feature_matrix'], degrees), axis=1)), edge_index=edge_index, edge_attr=edge_attr, y=y)
            torch.save(to_save, os.path.join(processed_dir, "data_{}.pt".format(i)))
        # print(self.raw_paths)
        if len(self.raw_file_names) != len(self.processed_file_names):
            print("Processing Dataset")
            Parallel(n_jobs=num_cpus)(delayed(process_helper)(self.processed_dir, raw_path,i,) for i, raw_path in enumerate(self.raw_paths))
        print("Finished Dataset Processing")

    def get(self,idx):
        # print("Getting Graph")
        # Returns a data object that represents the graph. See the following link for more details:
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
        load = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return Data(x=load.x, edge_index=load.edge_index, edge_attr=load.edge_attr, y=load.y, name=self.raw_file_names[idx])
    
class GATModelv1(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        # No need for bias in GAT Convs due to batch norms
        super(GATModelv1, self).__init__()
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


class GATModelv2(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        # No need for bias in GAT Convs due to batch norms
        super(GATModelv2, self).__init__()
        self.preprocess1 = nn.Linear(input_dim, 64, bias=False)
        self.BN0 = BatchNorm(64)
        
        self.GAT_1 = GATv2Conv(64, 8, heads=8, bias=False)
        self.BN1 = BatchNorm(64)
        
        self.GAT_2 = GATv2Conv(64, 8, heads=8, bias=False)
        self.BN2 = BatchNorm(64)

        self.GAT_3 = GATv2Conv(64, 8, heads=8, bias=False)
        self.BN3 = BatchNorm(64)

        self.GAT_4 = GATv2Conv(64, 8, heads=8, bias=False)
        self.BN4 = BatchNorm(64)

        self.postprocess1 = nn.Linear(64, output_dim)

        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=0)
         
    def forward(self, input):
        x = self.BN0(self.preprocess1(input.x))
        x = self.elu(x)
        
        block_1_out = self.BN1(self.GAT_1(x, input.edge_index))
        block_1_out = self.elu(torch.add(block_1_out, x))

        block_2_out = self.BN2(self.GAT_2(block_1_out,input.edge_index))
        block_2_out = self.elu(torch.add(block_2_out, block_1_out))              # DenseNet style skip connection
        
        block_3_out = self.BN3(self.GAT_3(block_2_out,input.edge_index))
        block_3_out = self.elu(torch.add(block_3_out, block_2_out))

        block_4_out = self.BN4(self.GAT_4(block_3_out,input.edge_index))
        block_4_out = self.elu(torch.add(block_4_out, block_3_out))

        
        x = self.postprocess1(block_4_out)
        return x

model = GATModelv2(input_dim=43, output_dim=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
 
prepend = str(os.getcwd())
print("Initializing Train Set")
data_set = KLIFSData(prepend + '/test_data_dir')
# Sticking with batch size 1 because it makes it easier to track predictions
test_dataloader =  DataLoader(data_set, batch_size = 1, shuffle=False, pin_memory=True, 
                              num_workers=num_cpus, persistent_workers=True)

test_epoch_loss = []
test_epoch_acc = []
test_epoch_mcc = []

all_probs = torch.Tensor([])
all_labels = torch.Tensor([])

print("Begining Evaluation")
model.eval()
with torch.no_grad():
    test_batch_loss = 0.0
    test_batch_acc = 0.0
    test_batch_mcc = 0.0
    for batch in test_dataloader:
        labels = batch.y

        out = model.forward(batch.to(device))
        probs = F.softmax(out, dim=-1)
        all_probs = torch.cat((all_probs, probs))
        all_labels = torch.cat((all_labels, labels))
        loss = F.nll_loss(torch.log(probs), labels,reduction='sum')         # Cross Entropy
        preds = np.argmax(out.detach().cpu().numpy(), axis=1)
        bl = loss.detach().cpu().item()

        ba = accuracy_score(labels, preds)
        bm = mcc(labels, preds)
            
        test_batch_loss += bl
        test_batch_acc  += ba
        test_batch_mcc  += bm
        # print("Test Batch Loss:", bl)
        # print("Test Batch Accu:", ba)
        # print("Test Batch MCC:", bm)
        np.save(prepend + '/test_metrics/test_probs/' + model_name + '_' + batch.name[0][:-4], probs)
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

# Code for calculating roc curves:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc%20curve

all_probs  =  all_probs.detach().cpu().numpy()
all_labels = all_labels.detach().cpu().numpy()
# all_labels = label_binarize(all_labels.detach().cpu().numpy(), classes=[0,1])
all_labels = np.array([[0,1] if x == 1 else [1,0] for x in all_labels])
print(all_probs.shape)
print(all_labels.shape) 

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 2

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels[:, i],all_probs[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "_roc_auc", roc_auc)
np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "_tpr", tpr)
np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "_fpr", fpr)