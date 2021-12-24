import os
from networkx.generators import directed
import numpy as np
import scipy
import multiprocessing
from joblib import Parallel, delayed

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import dropout_adj, from_scipy_sparse_matrix
import torch_geometric

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef as mcc

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

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
        return torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        

# Model Definition
class GATModel(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        # No need for bias in GAT Convs due to batch norms
        super(GATModel, self).__init__()
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
        
        
        # return self.softmax(x)        # I've seen this done but it doesn't make sense with cross-entropy loss

        return x

# Hyperparameters
num_hops = 2
num_epochs = 50
batch_size = 10
sample_size = 20
learning_rate = 0.005
train_test_split = .9

# Other Parameters
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
# num_cpus = os.cpu_count() # Don't do this, it will see all of the CPU's on the cluster. 
num_cpus = 4
print("The model will be using the following device:", device)
print("The model will be using {} cpus.".format(num_cpus))

# Some notes to help keep everything straight
# - KLIFSData loads the dataset containing multiple graphs
# - dataloader loads the individual graphs from the dataset
# - NighborhoodLoader samples the graphs returned by the dataloader

model = GATModel(input_dim=43, output_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# prepend = str(os.getcwd())
# path = prepend + '/data_atoms_w_atom_feats/'
# path_list = np.array([path + x for x in os.listdir(path)]) 

# np.random.seed(42)
# train_indices = np.random.choice(np.arange(0,len(path_list)), int(train_test_split*len(path_list)), replace=False) 
# val_indices     = set(np.random.choice(train_indices, int(len(train_indices)*.12), replace=False))                          # A subset of our training indices will be our validation indices
# train_indices   = set(train_indices) - val_indices   
# train_indices   = list(train_indices)
# val_indices     = list(val_indices)

# print("Initializing Train Set Dataloader")
# train_set = KLIFSData('./data_dir', path_list[train_indices], mode='train')
# train_dataloader = DataLoader(train_set, batch_size = 20, shuffle=True, pin_memory=True, num_workers=num_cpus)

# print("Initializing Test Set Dataloader")
# val_set = KLIFSData('./data_dir', path_list[val_indices],  mode='test')
# val_dataloader = DataLoader(val_set, batch_size = 2, shuffle=True, pin_memory=True, num_workers=num_cpus)


print(torch_geometric.__version__)

prepend = str(os.getcwd())
print("Initializing Train Set")
data_set = KLIFSData(prepend + '/data_dir')
# if len(data_set.raw_file_names) != len(data_set.processed_file_names):
#     data_set.process(num_workers = num_cpus)
train_size = int(train_test_split*len(data_set))
train_set, val_set = random_split(data_set, [train_size, len(data_set) - train_size])
print("Initializing Data Loaders")
train_dataloader = DataLoader(train_set, batch_size = 10, shuffle=True, pin_memory=True, num_workers=num_cpus)
val_dataloader = DataLoader(val_set, batch_size = 2, shuffle=True, pin_memory=True, num_workers=num_cpus)

# Track Training Statistics
training_epoch_loss = []
training_epoch_acc = []
training_epoch_mcc = []
training_epoch_auc = []

val_epoch_loss = []
val_epoch_acc = []
val_epoch_mcc = []
val_epoch_auc = []


writer = SummaryWriter(log_dir='atom_wise_model_logs/' + str(datetime.now()))
train_batch_num, val_batch_num = 0,0
train_epoch_num, val_epoch_num = 0,0

## NON-TRADITIONAL NEIGHBORHOOD SAMPLING LOOP
print("Begining Training")
for epoch in range(num_epochs):
    # Training Set
    model.train()
    # print("Begining New Epoch")
    if (train_epoch_num==0):
        print("Running {} batches per Epoch".format(len(train_dataloader)))
        epoch_start = time.time()
    training_batch_loss = 0.0
    training_batch_acc = 0.0
    training_batch_mcc = 0.0
    training_batch_auc = 0.0
    # got_batch = time.time()
    for batch in train_dataloader:
        # torch.cuda.empty_cache()
        # print("Got batch {}".format(time.time() - got_batch))
        # start_time = time.time()
        # print("Got Batch")
        # labels = batch.y.numpy()
        # pos_neg_size = max(int(np.sum(labels)*0.4),1)
        # positives = np.random.choice(np.arange(0,len(labels)), size = pos_neg_size, p=labels/np.sum(labels), replace=False)
        # temp = np.zeros(len(labels))
        # temp[labels == 0] = 1
        # negatives = np.random.choice(np.arange(0,len(labels)), size = pos_neg_size, p=temp/np.sum(temp), replace=False)

        # if 2 * pos_neg_size > 10000:
        #     print("Warning, running {} samples.".format(2 * pos_neg_size))

        # It turns out # atoms * 0.001 is ususually around 100, so we'll set that to our batch size
        # train_sampler = NeighborLoader(batch, num_neighbors=[-1,-1, -1], batch_size=500,
        # input_nodes= torch.LongTensor(np.concatenate((positives, negatives))),
        # directed=True, num_workers=num_cpus, persistent_workers=False) 
        # We can set directed to true because the number of gnn layers is equal to the number of layers sampled.
        # Doing this provides some speedup.add()
        
        # I don't like using floats for these but it will reduce memory usage significantly
        # training_sample_loss = 0.0
        # training_sample_acc = 0.0
        # training_sample_mcc = 0.0
        # training_sample_auc = 0.0
        # print("Time to sample {}".format(time.time() - start_time))
        # got_sample = time.time()
        # sample_count = 0
        # for sample in train_sampler:
        # if sample_count % 10 == 0:
        # print(sample_count)
        # sample_count += 1 
        # print("Got sample {}".format(time.time() - got_sample))
        labels = batch.y

        optimizer.zero_grad(set_to_none=True)
        out = model.forward(batch.to(device))

        loss = F.cross_entropy(out, batch.y)
        loss.backward() 
        optimizer.step()

        preds = np.argmax(out.detach().cpu().numpy(), axis=1)

        l = loss.detach().cpu().item()
        # training_sample_loss += l
        # training_sample_acc += accuracy_score(labels, preds)
        # training_sample_mcc += mcc(labels, preds)
        # training_sample_auc += roc_auc_score(labels, preds, labels=[0,1])
        bl = l #training_sample_loss/len(train_sampler)
        ba = accuracy_score(labels, preds)#training_sample_acc/len(train_sampler)
        bm = mcc(labels, preds)#training_sample_mcc/len(train_sampler)
        bc = roc_auc_score(labels, preds, labels=[0,1])#training_sample_auc/len(train_sampler)
        training_batch_loss += bl
        training_batch_acc  += ba
        training_batch_mcc  += bm
        training_batch_auc  += bc
        # if train_batch_num % 10 == 0:
        #     print("Training Batch Loss:", bl)
        #     print("Training Batch Accu:", ba)
        #     print("Training Batch MCC:", bm)
        #     print("Training Batch AUC:", bc)
        #     print("That batch took {} seconds.".format(time.time() - start_time))
        writer.add_scalar('Batch_Loss/Train', bl, train_batch_num)
        writer.add_scalar('Batch_ACC/Train',  ba,  train_batch_num)
        writer.add_scalar('Batch_MCC/Train',  bm,  train_batch_num)
        writer.add_scalar('Batch_AUC/Train',  bc,  train_batch_num)
        train_batch_num += 1
        
        # scheduler.step()    # No one would ever put the scheduler inside the batch loop. It should go in the epoch loop
                            # I'm *trying* this because epochs are just so, so long
    print("******* EPOCH END, EPOCH TIME: {}".format(time.time() - epoch_start))
    
    training_epoch_loss.append(training_batch_loss/len(train_dataloader))
    training_epoch_acc.append(training_batch_acc/len(train_dataloader))
    training_epoch_mcc.append(training_batch_mcc/len(train_dataloader))
    training_epoch_auc.append(training_batch_auc/len(train_dataloader))
    print("Training Epoch {} Loss: {}".format(epoch, training_epoch_loss[-1]))
    print("Training Epoch {} Accu: {}".format(epoch, training_epoch_acc[-1]))
    print("Training Epoch {} MCC: {}".format(epoch, training_epoch_mcc[-1]))
    print("Training Epoch {} AUC: {}".format(epoch, training_epoch_auc[-1]))
    writer.add_scalar('Epoch_Loss/Train', training_epoch_loss[-1], train_epoch_num)
    writer.add_scalar('Epoch_ACC/Train',  training_epoch_acc[-1],  train_epoch_num)
    writer.add_scalar('Epoch_MCC/Train',  training_epoch_mcc[-1],  train_epoch_num)
    writer.add_scalar('Epoch_AUC/Train',  training_epoch_auc[-1],  train_epoch_num)

    
    torch.save(model.state_dict(), "./trained_models/trained_model_{}_epoch_{}".format(str(time.time()), train_epoch_num))
    
    train_epoch_num += 1

    model.eval()
    with torch.no_grad():
        val_batch_loss = 0.0
        val_batch_acc = 0.0
        val_batch_mcc = 0.0
        val_batch_auc = 0.0

        for batch in val_dataloader:
            labels = batch.y

            out = model.forward(batch.to(device))
            loss = F.cross_entropy(out, batch.y)
            preds = np.argmax(out.detach().cpu().numpy(), axis=1)
            bl = loss.detach().cpu().item()

            ba = accuracy_score(labels, preds)
            bm = mcc(labels, preds)
            bc = roc_auc_score(labels, preds, labels=[0,1])
                
            val_batch_loss += bl
            val_batch_acc  += ba
            val_batch_mcc  += bm
            val_batch_auc  += bc
            # print("Validation Batch Loss:", val_batch_loss[-1])
            # print("Validation Batch Accu:", val_batch_acc[-1
            writer.add_scalar('Batch_Loss/Val', bl, val_batch_num)
            writer.add_scalar('Batch_ACC/Val',  ba,  val_batch_num)
            writer.add_scalar('Batch_MCC/Val',  bm,  val_batch_num)
            writer.add_scalar('Batch_AUC/Val',  bc,  val_batch_num)
            val_batch_num += 1


        val_epoch_loss.append(val_batch_loss/len(val_dataloader))
        val_epoch_acc.append(val_batch_acc/len(val_dataloader))
        val_epoch_mcc.append(val_batch_mcc/len(val_dataloader))
        val_epoch_auc.append(val_batch_auc/len(val_dataloader))
        print("Validation Epoch {} Loss: {}".format(epoch, val_epoch_loss[-1]))
        print("Validation Epoch {} Accu: {}".format(epoch, val_epoch_acc[-1]))
        print("Validation Epoch {} MCC: {}".format(epoch, val_epoch_mcc[-1]))
        print("Validation Epoch {} AUC: {}".format(epoch, val_epoch_auc[-1]))
        writer.add_scalar('Epoch_Loss/Val', val_epoch_loss[-1], val_epoch_num)
        writer.add_scalar('Epoch_ACC/Val',  val_epoch_acc[-1],  val_epoch_num)
        writer.add_scalar('Epoch_MCC/Val',  val_epoch_mcc[-1],  val_epoch_num)
        writer.add_scalar('Epoch_AUC/Val',  val_epoch_auc[-1],  val_epoch_num)

        val_epoch_num += 1

writer.close()
# torch.save(model.state_dict(), "./trained_models/trained_model_{}".format(str(time.time())))


## CLUSTER SAMPLING LOOP

# for epoch in range(num_epochs):
#     # Training Set
#     model.train()
#     for batch in train_dataloader:
#         # print("Got batch")
#         train_data = ClusterData(batch, num_parts=20)
#         # print("Clustered data")
#         train_sampler = ClusterLoader(train_data, batch_size = 10, num_workers=4, shuffle=True)
#         # print("Instantiated Loader")
#         for sample in train_sampler:
#             # print("Sampling...")
#             # print("Running Sample")
#             # torch.cuda.empty_cache()
#             labels = sample.y

#             optimizer.zero_grad()
#             out = model.forward(sample.to(device))

#             loss = F.cross_entropy(out, sample.y)
#             loss.backward() 
#             optimizer.step()

#             preds = np.argmax(out.detach().cpu().numpy(), axis=1)

#             l = loss.detach().cpu()
#             training_sample_loss.append(l)
#             training_sample_acc.append(accuracy_score(labels, preds))
            
#         training_batch_loss.append(np.mean(training_sample_loss[-10:]))
#         training_batch_acc.append(np.mean(training_sample_acc[-10:]))
#         print("Training Batch Loss:", training_batch_loss[-1])
#         print("Training Batch Accu:", training_batch_acc[-1])

        
#     training_epoch_loss.append(training_batch_loss[-10:])
#     training_epoch_acc.append(training_batch_acc[-10:])
#     print("Training Epoch {} Loss: {}".format(epoch, training_epoch_loss[-1]))
#     print("Training Epoch {} Accu: {}".format(epoch, training_epoch_acc[-1]))

    # Validation Set
    
    # model.eval()
    # with torch.no_grad():
    #     for batch in val_dataloader:
    #         val_sampler = NeighborLoader(batch, num_neighbors = [30] * num_iterations, batch_size = sample_size, directed=False)
    #         for sample in val_sampler:
    #             # print("Running Sample")
    #             # torch.cuda.empty_cache()
    #             out = model.forward(sample.to(device))
    #             # print(out)
    #             # print(sample.y)
    #             loss = F.cross_entropy(out, sample.y)
                
    #             l = loss.detach().cpu()
    #             val_sample_loss.append(l)
    #             # print("Validation Sample Loss:", l)
                
    #         val_batch_loss.append(np.mean(val_sample_loss[-sample_size:]))
    #         print("Validation Batch Loss:", val_batch_loss[-1])
            
    #     val_epoch_loss.append(val_batch_loss[-batch_size:])
        # print("Validation Epoch {} Loss: {}".format(epoch, val_epoch_loss[-1]))
    
    # To Do: 
    # - Add actual statistics tracking
    # - Add test set
    # - Integrate Tensorboard
            
# for epoch in range(num_epochs):
#     # Training Set
#     model.train()
#     for batch in train_dataloader:
#         train_sampler = NeighborLoader(batch, num_neighbors=-1)
#         for sample in train_sampler:
#             # print("Sampling...")
#             # print("Running Sample")
#             # torch.cuda.empty_cache()
#             labels = sample.y

#             optimizer.zero_grad()
#             out = model.forward(sample.to(device))

#             loss = F.cross_entropy(out, sample.y)
#             loss.backward() 
#             optimizer.step()

#             preds = np.argmax(out.detach().cpu().numpy(), axis=1)

#             l = loss.detach().cpu()
#             training_sample_loss.append(l)
#             training_sample_acc.append(accuracy_score(labels, preds))
            
#         training_batch_loss.append(np.mean(training_sample_loss[-4:]))
#         training_batch_acc.append(np.mean(training_sample_acc[-4:]))
#         print("Training Batch Loss:", training_batch_loss[-1])
#         print("Training Batch Accu:", training_batch_acc[-1])

        
#     training_epoch_loss.append(training_batch_loss[-4:])
#     training_epoch_acc.append(training_batch_acc[-4:])
#     print("Training Epoch {} Loss: {}".format(epoch, training_epoch_loss[-1]))
#     print("Training Epoch {} Accu: {}".format(epoch, training_epoch_acc[-1]))


            

# batch = next(iter(train_dataloader))
# print("Got batch")
# train_data = ClusterData(batch,num_parts=20)
# print("Clustered data")
# train_sampler = ClusterLoader(train_data, batch_size = sample_size)
# print("Instantiated Loader")
# for sample in train_sampler:
#     print("Sampling...")
#     print(sample)