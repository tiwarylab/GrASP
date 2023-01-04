from collections import OrderedDict
import os
import argparse
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm

# import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.nn import GATConv, GATv2Conv

from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score
from sklearn.metrics import matthews_corrcoef as mcc
from torch.utils.tensorboard import SummaryWriter
import torch

from GASP_dataset import GASPData
from atom_wise_models import GASPformer_BN, GASPformer_GN, GASPformer_IN, GASPformer_IN_stats, GASPformer_PN, GASPformer_GNS, GASPformer_AON, GASPformer_no_norm
from simple_models import GAT_model


###################################################################################
''' Some bits that are surrounded in comments like this can be used to temporarily
    modify the code to use the training set.'''
###################################################################################


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


def distance_sigmoid(data, midpoint, slope):
    x = -slope*(data-midpoint)
    sigmoid = torch.sigmoid(x)
    
    return sigmoid

def initialize_model(parser_args):
    model_name = parser_args.model
    weight_groups = parser_args.weight_groups
    group_layers = parser_args.group_layers

    if model_name == 'transformer':
        print("Using GASPformer with BatchNorm")
        model = GASPformer_BN(input_dim = 60, noise_variance=0, GAT_heads=4)
    elif model_name == 'transformer_gn':
        print("Using GASPformer with GraphNorm")
        model = GASPformer_GN(input_dim = 60, noise_variance=0, GAT_heads=4)
    elif model_name == 'transformer_in':
        print("Using GASPformer with InstanceNorm")
        model = GASPformer_IN(input_dim = 60, noise_variance=0, GAT_heads=4)
    elif model_name == 'transformer_in_stats':
        print("Using GASPformer with InstancehNorm")
        model = GASPformer_IN_stats(input_dim = 60, noise_variance=0, GAT_heads=4)
    elif model_name == 'transformer_pn':
        print("Using GASPformer with PairNorm")
        model = GASPformer_PN(input_dim = 60, noise_variance=0, GAT_heads=4)
    elif model_name == 'transformer_gns':
        print("Using GASPformer with GraphNormSigmoid")
        model = GASPformer_GNS(input_dim = 60, noise_variance=0, GAT_heads=4)
    elif model_name == 'transformer_aon':
        print("Using GASPformer with AffineOnlyNorm")
        model = GASPformer_AON(input_dim = 60, noise_variance=0, GAT_heads=4)
    elif model_name == 'transformer_no_norm':
        print("Using GASPformer without norms.")
        model = GASPformer_no_norm(input_dim = 60, noise_variance=0, GAT_heads=4)
    elif model_name == 'gat':
        print("Using GAT")
        model = GAT_model(input_dim=60, noise_variance=0,
         GAT_heads=4, GAT_style=GATConv, weight_groups=weight_groups,
          group_layers=group_layers)
    elif model_name == 'gatv2':
        print("Using GATv2")
        model = GAT_model(input_dim=60, noise_variance=0,
         GAT_heads=4, GAT_style=GATv2Conv, weight_groups=weight_groups,
          group_layers=group_layers)
    else:
        raise ValueError("Unknown Model Type:", model_name)
    return model

prepend = str(os.getcwd())

parser = argparse.ArgumentParser(description="Evaluate site prediction on test sets.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("test_set", choices=["val", "coach420", "coach420_mlig", "holo4k", "holo4k_mlig"], help="Test set.")
parser.add_argument("model_path", help="Path to the model from ./trained_models/")
parser.add_argument("-m", "--model", default="gatv2", choices=["transformer", "transformer_gn", "transformer_in", "transformer_in_stats",
     "transformer_pn", "transformer_gns", "transformer_aon", "transformer_no_norm", "gat", "gatv2"], help="GNN architecture to test.")
parser.add_argument("-sp", "--sigmoid_params", type=float, nargs=2, default=[5, 3], help="Parameters for sigmoid labels [label_midpoint, label_slope].")
parser.add_argument("-wg", "--weight_groups", type=int, default=1, help="Number of weight-sharing groups.")
parser.add_argument("-gl", "--group_layers", type=int, default=12, help="Number of layers per weight-sharing group.")
parser.add_argument("-ao", "--all_atom_prediction", action="store_true", help="Option to perform inference on all atoms as opposed to solvent exposed.")
parser.add_argument("-kh", "--k_hops", type=int, default=1, help="Number of hops for constructing a surface graph.")
args = parser.parse_args()
model_name = args.model_path
model = initialize_model(args)

model_path = prepend + "/trained_models/" + model_name
set_to_use = args.test_set
surface_only = not args.all_atom_prediction
k_hops = args.k_hops

label_midpoint, label_slope = args.sigmoid_params

###################################################################################



# Other Parameters
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

num_cpus = 4
print("The model will be using the following device:", device)
print("The model will be using {} cpus.".format(num_cpus))

# model = GATModelv2(input_dim=43, output_dim=2)
# model = Two_Track_GATModel(input_dim=43, output_dim=2, drop_prob=0.1, left_aggr="max", right_aggr="mean").to(device)

'''
This whole bit is a quick fix that allows us to load incorrectly saved models.
This has been fixed in the training script and as a result we can remove this in the future
'''
state_dict = torch.load(model_path, map_location=device)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] #remove 'module'
#     new_state_dict[name] = v

# model.load_state_dict(new_state_dict)
model.load_state_dict(state_dict)

model.to(device)
 
#########################
if set_to_use == 'val':
    print("Initializing Validation Set")
    path_to_dataset = prepend + '/scPDB_data_dir'
    metric_dir = '/test_metrics/validation'

    data_set = GASPData(path_to_dataset, num_cpus, cutoff=5, surface_subgraph_hops=k_hops)
    train_mask, val_mask = k_fold(data_set, prepend, 0) 
    val_set     = data_set[val_mask]
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_cpus)
else:  
    if set_to_use ==  'coach420':
        print("Initializing coach420 Set")    
        path_to_dataset = prepend + '/benchmark_data_dir/coach420'
        metric_dir = '/test_metrics/coach420'
    elif set_to_use ==  'coach420_mlig':
        print("Initializing coach420 Mlig Set")    
        path_to_dataset = prepend + '/benchmark_data_dir/coach420_mlig'
        metric_dir = '/test_metrics/coach420_mlig'
    elif set_to_use == 'holo4k':
        print("Initializing holo4k Set")    
        path_to_dataset = prepend + '/benchmark_data_dir/holo4k'
        metric_dir = '/test_metrics/holo4k'
    elif set_to_use == 'holo4k_mlig':
        print("Initializing holo4k Mlig Set")    
        path_to_dataset = prepend + '/benchmark_data_dir/holo4k_mlig'
        metric_dir = '/test_metrics/holo4k_mlig'
    else:
        raise ValueError("Expected one of {'val','chen','coach420','holo4k','sc6k'} as set_to_use but got:", str(set_to_use))
    data_set = GASPData(path_to_dataset, num_cpus, cutoff=5, surface_subgraph_hops=k_hops)
    data_set.process()
    val_dataloader = DataLoader(data_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_cpus)

test_epoch_loss = []
test_epoch_acc = []
test_epoch_mcc = []
test_epoch_pr_auc = []

all_probs = torch.Tensor([])
all_labels = torch.Tensor([])

prob_path = prepend + metric_dir + '/probs/' + model_name + '/'
label_path = prepend + metric_dir + '/labels/' + model_name + '/'
surface_path = prepend + metric_dir + '/SASAs/'
index_path = prepend + metric_dir + '/indices/' + model_name + '/'

if not os.path.isdir(prob_path):
    os.makedirs(prob_path)
if not os.path.isdir(label_path):
    os.makedirs(label_path)
if not os.path.isdir(surface_path):
    os.makedirs(surface_path)
if not os.path.isdir(index_path):
    os.makedirs(index_path)

print("Begining Evaluation")
model.eval()
with torch.no_grad():
    test_batch_loss = 0.0
    test_batch_acc = 0.0
    test_batch_mcc = 0.0
    test_batch_pr_auc = 0.0
    # for batch, name in test_dataloader:
    for batch, name in tqdm(val_dataloader, position=0, leave=True):
        batch.y = distance_sigmoid(batch.y, label_midpoint, label_slope)
        batch.y = torch.stack([1-batch.y, batch.y], dim=1)
        labels = batch.y.to(device)
        assembly_name = name[0][:-4]  

        out, _ = model.forward(batch.to(device))

        if surface_only:
            surf_mask = batch.surf_mask.to(device)
            labels = labels[surf_mask]
            out = out[surf_mask]

        probs = F.softmax(out, dim=-1) 
        all_probs = torch.cat((all_probs, probs.detach().cpu()))
        all_labels = torch.cat((all_labels, labels.detach().cpu()))
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(out, labels)        # Cross Entropy
        preds = np.argmax(probs.detach().cpu().numpy(), axis=1) # [1 if x[1] > prediction_threshold else 0 for x in probs]
        bl = loss.detach().cpu().item()
        
        labels = batch.y.detach().cpu()

        if surface_only:
            labels = labels[surf_mask.detach().cpu()]

        hard_labels = np.argmax(labels, axis=1)
        surf_masks = batch.surf_mask.detach().cpu()
        atom_indices = batch.atom_index.detach().cpu()
        
        ba = accuracy_score(hard_labels, preds)
        bm = mcc(hard_labels, preds)
        bpr = average_precision_score(hard_labels, probs.detach().cpu().numpy()[:,1])

        test_batch_loss += bl
        test_batch_acc  += ba
        test_batch_mcc  += bm
        test_batch_pr_auc += bpr
        np.save(prob_path + assembly_name, probs.detach().cpu().numpy())
        np.save(label_path + assembly_name, labels.detach().cpu().numpy())
        np.save(surface_path + assembly_name, surf_masks.detach().cpu().numpy())
        np.save(index_path + assembly_name, atom_indices.detach().cpu().numpy())

        
        # writer.add_scalar('Batch_Loss/test', bl, test_batch_num)
        # writer.add_scalar('Batch_Acc/test',  ba,  test_batch_num)
        # writer.add_scalar('Batch_Acc/MCC',  bm,  test_batch_num)

    # test_epoch_loss.append(test_batch_loss/len(test_dataloader))
    # test_epoch_acc.append(test_batch_acc/len(test_dataloader))
    # test_epoch_mcc.append(test_batch_mcc/len(test_dataloader))
    test_epoch_loss.append(test_batch_loss/len(val_dataloader))
    test_epoch_acc.append(test_batch_acc/len(val_dataloader))
    test_epoch_mcc.append(test_batch_mcc/len(val_dataloader))
    test_epoch_pr_auc.append(test_batch_pr_auc/len(val_dataloader))
    print("Loss: {}".format(test_epoch_loss[-1]))
    print("Accu: {}".format(test_epoch_acc[-1]))
    print("MCC:  {}".format(test_epoch_mcc[-1]))
    print("PR AUC: {}".format(test_epoch_pr_auc[-1]))
    # writer.add_scalar('Epoch_Loss/test', test_epoch_loss[-1], test_epoch_num)
    # writer.add_scalar('Epoch_Acc/test',  test_epoch_acc[-1],  test_epoch_num)
    # writer.add_scalar('Epoch_Acc/MCC',  test_epoch_mcc[-1],  test_epoch_num)

# Code for calculating roc curves:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc%20curve

all_probs  =  all_probs.detach().cpu().numpy()
all_labels = all_labels.detach().cpu().numpy()

all_prob_path = prepend + metric_dir + '/all_probs/' + model_name + '/'
all_label_path = prepend + metric_dir + '/all_labels/' + model_name + '/'

if not os.path.isdir(all_prob_path):
    os.makedirs(all_prob_path)
if not os.path.isdir(all_label_path):
    os.makedirs(all_label_path)

np.savez(all_prob_path + 'all_probs', all_probs)
np.savez(all_label_path + 'all_labels', all_labels)

all_hard_labels = (all_labels >= .5).astype('float')

fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
n_classes = 2

for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(all_hard_labels[:, i],all_probs[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(all_hard_labels.ravel(), all_probs.ravel())
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

roc_path = prepend + metric_dir + '/roc_curves/' + model_name
if not os.path.isdir(roc_path):
    os.makedirs(roc_path)

np.savez(roc_path + "/roc_auc", roc_auc)
np.savez(roc_path + "/tpr", tpr)
np.savez(roc_path + "/fpr", fpr)
np.savez(roc_path + "/thresholds", thresholds)
