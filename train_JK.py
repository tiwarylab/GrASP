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

from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

from KLIFS_dataset import KLIFSData
from atom_wise_models import Two_Track_JK_GATModel 

job_start_time = time.time()

# LabelSmoothing Loss Source: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
# Available by default in PyTorch 1.10 but there seems to be some conflict between PyTorch 1.10 and PyG.
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

#ref: https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/train_eval.py#L82-L97
def k_fold(dataset, path, fold_number):
    val_names    = np.loadtxt(prepend + "/splits/test_ids_fold"  + str(fold_number), dtype='str')
    train_names   = np.loadtxt(prepend + "/splits/train_ids_fold" + str(fold_number), dtype='str')
    
    train_indices, val_indices = [], []
    
    for idx, name in enumerate(dataset.raw_file_names):
        if name[:4] in val_names: 
            val_indices.append(idx)
        if name[:4] in train_names:
            train_indices.append(idx)

    train_mask = torch.ones(len(dataset), dtype=torch.bool)
    val_mask = torch.ones(len(dataset), dtype=torch.bool)
    train_mask[val_indices] = 0
    val_mask[train_mask] = 0
    
    # Temporary sanity check to make sure I got this right
    assert train_mask.sum() > val_mask.sum()

    return train_mask, val_mask


   

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
num_cpus = 8
print("The model will be using the following device:", device)
print("The model will be using {} cpus.".format(num_cpus))

model = Two_Track_JK_GATModel(input_dim=88, output_dim=2, drop_prob=0.1, left_aggr="max", right_aggr="mean").to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
loss_fn = LabelSmoothingLoss(2, smoothing=0.2, weight=torch.FloatTensor([0.80,1.20]).to(device))


prepend = str(os.getcwd())
print("Initializing Train Set")
data_set = KLIFSData(prepend + '/data_dir', num_cpus, cutoff=5)
# data_set.process()

# Set to one temporarily to avoid doing full cv
for cv_iteration in range(1):
    train_mask, val_mask = k_fold(data_set, prepend, cv_iteration)
    train_set   = data_set[train_mask]
    val_set     = data_set[val_mask]

    # train_size = int(train_test_split*len(data_set))
    # train_set, val_set = random_split(data_set, [train_size, len(data_set) - train_size], generator=torch.Generator().manual_seed(42))
    print("Initializing Data Loaders")
    train_dataloader = DataLoader(train_set, batch_size=2, shuffle=True, pin_memory=True, num_workers=num_cpus)
    val_dataloader = DataLoader(val_set, batch_size=2, shuffle=True, pin_memory=True, num_workers=num_cpus)


    # Track Training Statistics
    training_epoch_loss = []
    training_epoch_acc = []
    training_epoch_mcc = []
    training_epoch_auc = []

    val_epoch_loss = []
    val_epoch_acc = []
    val_epoch_mcc = []
    val_epoch_auc = []


    writer = SummaryWriter(log_dir='atom_wise_model_logs/cv_split_' + str(cv_iteration) + "/" + str(job_start_time))
    train_batch_num, val_batch_num = 0,0
    train_epoch_num, val_epoch_num = 0,0

    print("Begining Training")
    for epoch in range(num_epochs):
        # Training Set
        model.train()
        if (train_epoch_num==0):
            print("Running {} batches per Epoch".format(len(train_dataloader)))
            epoch_start = time.time()
        training_batch_loss = 0.0
        training_batch_acc = 0.0
        training_batch_mcc = 0.0
        training_batch_auc = 0.0
        for batch, _  in train_dataloader:

            labels = batch.y
            x_np = batch.x.numpy()

            optimizer.zero_grad(set_to_none=True)
            out = model.forward(batch.to(device))

            # loss = F.cross_entropy(out, batch.y)
            loss = loss_fn(out,batch.y)
            loss.backward() 
            optimizer.step()

            preds = np.argmax(out.detach().cpu().numpy(), axis=1)

            l = loss.detach().cpu().item()
            
            bl = l 
            ba = accuracy_score(labels, preds)
            bm = mcc(labels, preds)
            bc = roc_auc_score(labels, preds, labels=[0,1])
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
            
        scheduler.step()
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

        if not os.path.isdir("./trained_models/trained_model_{}/".format(str(job_start_time))):
            os.makedirs("./trained_models/trained_model_{}/".format(str(job_start_time)))
        torch.save(model.state_dict(), "./trained_models/trained_model_{}/epoch_{}".format(str(job_start_time), train_epoch_num))
        
        train_epoch_num += 1

        model.eval()
        with torch.no_grad():
            val_batch_loss = 0.0
            val_batch_acc = 0.0
            val_batch_mcc = 0.0
            val_batch_auc = 0.0

            for batch, _ in val_dataloader:
                labels = batch.y

                out = model.forward(batch.to(device))
                # loss = F.cross_entropy(out, batch.y)
                loss = loss_fn(out,batch.y) 
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
