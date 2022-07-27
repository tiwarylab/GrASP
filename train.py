import os
from re import X
from networkx.generators import directed
import numpy as np
import scipy
import multiprocessing
from glob import glob
import sys
import argparse
from joblib import Parallel, delayed

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.nn import DataParallel

from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader, DataListLoader
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

from GASP_dataset import GASPData#, GASPData_noisy_nodes
from atom_wise_models import Hybrid_1g12_self_edges, Hybrid_1g12_self_edges_transformer_style

job_start_time = time.time()
prepend = str(os.getcwd())


# LabelSmoothing Loss Source: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
# Available by default in PyTorch 1.10 but there seems to be some conflict between PyTorch 1.10 and PyG.
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None, device='cpu'):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight.to(device)
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
def k_fold(dataset:GASPData,train_path:str, val_path, i):
    val_names    = np.loadtxt(val_path, dtype=str)
    train_names   = np.loadtxt(train_path, dtype=str)
    
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
    # print(train_mask.sum())
    # print(val_mask.sum())
    assert train_mask.sum() > val_mask.sum()

    return (dataset[train_mask], dataset[val_mask], i)


   
def main(node_noise_variance : float, training_split='cv'):
    # Hyperparameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    class_loss_weight = args.class_loss_weight#[0.8,1.2]
    label_smoothing = args.label_smoothing#0.2
    head_loss_weight = args.head_loss_weight

    if training_split not in ['cv', 'train_full', 'chen', 'coach420', 'holo4k', 'sc6k']:
        raise ValueError("Expected training_split to be one of ['cv', 'train_full', 'chen', 'coach420', 'holo4k', 'sc6k'] but got", training_split)
    
    num_cpus = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('The model will be using {} gpus.'.format(world_size))
    # print("The model will be using {} cpus.".format(num_cpus), flush=True)

    # print("Loss Weighting:", str(class_loss_weight))
    # print("Weighted Cross Entropy Loss Function Weight:", head_loss_weight[0])
    # print("Reconstruction (MSE) Loss Function Weight:  ", head_loss_weight[1])

    # model = Two_Track_GATModel(input_dim=88, output_dim=2, drop_prob=0.1, left_aggr="max", right_aggr="mean")
    # model =   Hybrid_1g8_noisy(input_dim=88, node_noise_variance=node_noise_variance, edge_noise_variance=edge_noise_variance)
    # if str(sys.argv[3]) == "Hybrid_1g12_self_edges":
    #     print("Using Hybrid_1g12_self_edges")
    #     model = Hybrid_1g12_self_edges(input_dim = 88, noise_variance = node_noise_variance, GAT_heads=4)
    # elif str(sys.argv[3]) == "Hybrid_1g12_self_edges_dropped_bn":
    #     print("Using Hybrid_1g12_self_edges_dropped_bn")
    #     model = Hybrid_1g12_self_edges_dropped_bn(input_dim = 88, noise_variance = node_noise_variance, GAT_heads=4)
    # elif str(sys.argv[3]) == "Hybrid_1g12_self_edges_transformer_style":
    #     print("Using Hybrid_1g12_self_edges_transformer_style")
    #     model = Hybrid_1g12_self_edges_transformer_style(input_dim = 88, noise_variance = node_noise_variance, GAT_heads=4)
    
    if args.model == 'hybrid':
            print("Using Hybrid_1g12_self_edges with one-hot self-edge encoding, traditional")
            model = Hybrid_1g12_self_edges(input_dim = 88, noise_variance = node_noise_variance, GAT_heads=4)
    elif args.model == 'transformer':
        print("Using Hybrid_1g12_self_edges with one-hot self-edge encoding, transformer style")
        model = Hybrid_1g12_self_edges_transformer_style(input_dim = 88, noise_variance = node_noise_variance, GAT_heads=4)
    else:
        raise ValueError("Unknown Model Type:", args.model)
    model =  DataParallel(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose=True)
    
    loss_fn = LabelSmoothingLoss(2, smoothing=label_smoothing, weight=torch.FloatTensor(class_loss_weight), device=device)
    
    head_loss_weight = torch.tensor(head_loss_weight).to(device)
    
    data_set = GASPData(prepend + '/scPDB_data_dir', num_cpus, cutoff=5)
    
    do_validation = False
    if training_split == 'cv':
        do_validation = True
        val_paths = []
        train_paths = []
        for fold_number in range(1):
            val_paths.append(prepend + "/splits/test_ids_fold"  + str(fold_number))
            train_paths.append(prepend + "/splits/train_ids_fold" + str(fold_number))
        data_points = zip(train_paths,val_paths)

        gen = (k_fold(data_set, train_path, val_path, i) for i, (train_path, val_path) in enumerate(data_points))

    elif training_split == 'train_full':
        do_validation = False
        gen = zip([data_set], [0], [0])

    else:
        if training_split == 'chen':
            train_names = np.loadtxt(prepend + '/splits/train_ids_chen', dtype=str)
        elif training_split == 'coach420':
            train_names = np.loadtxt(prepend + '/splits/train_ids_coach420', dtype=str)
        elif training_split == 'holo4k':
            train_names = np.loadtxt(prepend + '/splits/train_ids_holo4k', dtype=str)
        elif training_split == 'sc6k':
            train_names = np.loadtxt(prepend + '/splits/train_ids_sc6k', dtype=str)
        train_indices = []
        for idx, name in enumerate(data_set.raw_file_names):
            if name.split('_')[0] in train_names:
                train_indices.append(idx)
        train_mask = torch.zeros(len(data_set), dtype=torch.bool)
        train_mask[train_indices] = 1
        gen = zip([data_set[train_mask]],[data_set[torch.zeros(len(data_set),dtype=torch.bool)]],[0])
        # print(gen)

    # Set to one temporarily to avoid doing full cv
    for train_set, val_set, cv_iteration in gen:
        train_dataloader = DataListLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_cpus)
        if do_validation: val_dataloader = DataListLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_cpus)
        # Track Training Statistics
        training_epoch_loss = []
        training_epoch_acc = []
        training_epoch_mcc = []
        training_epoch_auc = []

        val_epoch_loss = []
        val_epoch_acc = []
        val_epoch_mcc = []
        val_epoch_auc = []

        writer = SummaryWriter(log_dir='atom_wise_model_logs/' + training_split + '/cv_split_' + str(cv_iteration) + "/" + model_id)
        train_batch_num, val_batch_num = 0,0
        train_epoch_num, val_epoch_num = 0,0
        
        for epoch in range(num_epochs):
            # Training Set
            model.train()
            if (train_epoch_num==0):
                print("Running {} batches per Epoch".format(len(train_dataloader)), flush=True)
                epoch_start = time.time()
            training_batch_loss = 0.0
            training_batch_acc = 0.0
            training_batch_mcc = 0.0
            training_batch_auc = 0.0
            for batch in train_dataloader:
                batch = list(map(lambda x: x[0].to(device), batch))
                
                unperturbed_x = torch.cat([data.x.clone().detach().to(device) for data in batch])
                for data in batch:
                    data.x = data.x + (node_noise_variance**0.5)*torch.randn_like(data.x)
                labels  = torch.cat([data.y.clone().detach() for data in batch]).cpu().numpy()
                y       = torch.cat([data.y.to(device) for data in batch])
            
                optimizer.zero_grad(set_to_none=True)
                out, out_recon = model.forward(batch)

                weighted_xent_l, mse_l = head_loss_weight[0] * loss_fn(out,y), head_loss_weight[1] * F.mse_loss(out_recon, unperturbed_x)           

                loss = weighted_xent_l + mse_l
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
            print("Training Epoch {} AUC: {}".format(epoch, training_epoch_auc[-1]), flush=True)
            writer.add_scalar('Epoch_Loss/Train', training_epoch_loss[-1], train_epoch_num)
            writer.add_scalar('Epoch_ACC/Train',  training_epoch_acc[-1],  train_epoch_num)
            writer.add_scalar('Epoch_MCC/Train',  training_epoch_mcc[-1],  train_epoch_num)
            writer.add_scalar('Epoch_AUC/Train',  training_epoch_auc[-1],  train_epoch_num)

            if not os.path.isdir("./trained_models/{}/trained_model_{}/".format(training_split, model_id)):
                os.makedirs("./trained_models/{}/trained_model_{}/".format(training_split, model_id))
            torch.save(model.module.state_dict(), "./trained_models/{}/trained_model_{}/epoch_{}".format(training_split, model_id, train_epoch_num))
            
            train_epoch_num += 1

            if do_validation:
                model.eval()
                with torch.no_grad():
                    val_batch_loss = 0.0
                    val_batch_acc = 0.0
                    val_batch_mcc = 0.0
                    val_batch_auc = 0.0

                    for batch in val_dataloader:
                        batch = list(map(lambda x: x[0].to(device), batch))
                
                        unperturbed_x = torch.cat([data.x.clone().detach().to(device) for data in batch])
                        for data in batch:
                            data.x = data.x + (node_noise_variance**0.5)*torch.randn_like(data.x)
                        labels  = torch.cat([data.y.clone().detach() for data in batch]).cpu().numpy()
                        y       = torch.cat([data.y.to(device) for data in batch])
                    
                        optimizer.zero_grad(set_to_none=True)

                        out, _ = model.forward(batch)
                        # loss = F.cross_entropy(out, batch.y)
                        loss = loss_fn(out,y) 
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN for binding site prediction.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--training_split", default="cv", choices=["cv", "train_full", "chen", "coach420", "holo4k", "sc6k"], help="Training set.")
    parser.add_argument("-v", "--node_noise_variance", type=float, default=0.02, help="NoisyNodes variance.")
    parser.add_argument("-m", "--model", default="hybrid", choices=["hybrid", "transformer"], help="GNN architecture to train.")
    parser.add_argument("-e", "--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.005, help="Adam learning rate.")
    parser.add_argument("-cw", "--class_loss_weight", type=float, nargs=2, default=[1.0, 1.0], help="Loss weight for [negative, positive] classes.")
    parser.add_argument("-ls", "--label_smoothing", type=float, default=0, help="Level of label smoothing.")
    parser.add_argument("-hw", "--head_loss_weight", type=float, nargs=2, default=[.9,.1], help="Weight of the loss functions for the [inference, reconstruction] heads.")
    args = parser.parse_args()
    argstring='_'.join(sys.argv[1:]).replace('-','')
    model_id = argstring + str(job_start_time)

    node_noise_variance = args.node_noise_variance
    training_split = args.training_split

    print("Training with noise with variance", node_noise_variance, "and mean 0 added to nodes.")
    
    main(node_noise_variance,training_split)
