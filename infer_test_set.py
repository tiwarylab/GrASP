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

from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score
from sklearn.metrics import matthews_corrcoef as mcc
from torch.utils.tensorboard import SummaryWriter
import torch

from GASP_dataset import GASPData
from atom_wise_models import Hybrid_1g12_self_edges_transformer_GN

prepend = str(os.getcwd())

###################################################################################
''' Some bits that are surrounded in comments like this can be used to temporarily
    modify the code to use the training set.'''
###################################################################################


########################## Change Me To Change The Model ##########################
# model_name = "trained_model_1646775694.0918303/epoch_49" # Standard Model
# model_name = "trained_model_1647199519.6304853/epoch_49" # Noise Added to Node Features During Training Var = 0.2, Mean = 0, no second loss func
# model_name = "trained_model_1647218964.5406673/epoch_49" # Noisy Nodes With MSE loss
# model_name = "trained_model_1648747746.262174/epoch_26" # Mean Self Edges

# model_name = "trained_model_1g12_null_self_edges/epoch_49" # Null Self Edge. Use with Hybrid_1g12_self_edges
# model = Hybrid_1g12_self_edges(input_dim = 88)

# model_name = "trained_model_1g12_mean_self_edges/epoch_49" # Mean Self Edges. Use with Hybrid_1g12
# model = Hybrid_1g12(input_dim = 88)

# model_name = "trained_model_1g12_mean_self_edges_ligand_removed_SASA/epoch_49"
# model = Hybrid_1g12(input_dim = 88)

# model_name= "trained_model_1650260810.482072/epoch_46"   # After site relabeling and OB. Old model's hyperparams this is a Hybrid_1g12_self_edges() model
# model = Hybrid_1g12_self_edges(input_dim = 88)

# New model trained on freshly merged dataset, and fixed site labeling issues
# model_name = "trained_model_1652221046.78391/epoch_49"
# model = Hybrid_1g12_self_edges(input_dim = 88)

# Model trained exclusively on the holo4k split: "holo4k/trained_model_1656153741.4964042/epoch_49"
parser = argparse.ArgumentParser(description="Evaluate site prediction on test sets.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("test_set", choices=["coach420", "coach420_mlig", "holo4k", "holo4k_mlig"], help="Test set.")
parser.add_argument("model_path", help="Path to the model from ./trained_models/")
parser.add_argument("-sp", "--sigmoid_params", type=float, nargs=2, default=[6.5, 1], help="Parameters for sigmoid labels [label_midpoint, label_slope].")
args = parser.parse_args()
model_name = args.model_path
model = Hybrid_1g12_self_edges_transformer_GN(input_dim = 88)

model_path = prepend + "/trained_models/" + model_name
set_to_use = args.test_set

label_midpoint, label_slope = args.sigmoid_params

# model = Two_Track_GIN_GAT_Noisy_Nodes(input_dim=88, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add") 
# model = Two_Track_GIN_GAT_fixed_bn(input_dim=88, output_dim=2, drop_prob=0.1, GAT_aggr="mean", GIN_aggr="add") 

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

    data_set = GASPData(path_to_dataset, num_cpus, cutoff=5, label_midpoint=label_midpoint, label_slope=label_slope)
    train_mask, val_mask = k_fold(data_set, prepend, 0) 
    val_set     = data_set[val_mask]
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_cpus)
else:  
    if set_to_use == 'chen':
        print("Initializing Chen Set")    
        path_to_dataset = prepend + '/benchmark_data_dir/chen'
        metric_dir = '/test_metrics/chen'
    elif set_to_use ==  'coach420':
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
    elif set_to_use == 'sc6k':
        print("Initializing sc6k Set")    
        path_to_dataset = prepend + '/benchmark_data_dir/sc6k'
        metric_dir = '/test_metrics/sc6k'
    else:
        raise ValueError("Expected one of {'val','chen','coach420','holo4k','sc6k'} as set_to_use but got:", str(set_to_use))
    data_set = GASPData(path_to_dataset, num_cpus, cutoff=5, label_midpoint=label_midpoint, label_slope=label_slope)
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

if not os.path.isdir(prob_path):
    os.makedirs(prob_path)
if not os.path.isdir(label_path):
    os.makedirs(label_path)
if not os.path.isdir(surface_path):
    os.makedirs(surface_path)

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
        labels = batch.y.to(device)
        assembly_name = name[0][:-4]  

        out, _ = model.forward(batch.to(device))
        probs = F.softmax(out, dim=-1) 
        all_probs = torch.cat((all_probs, probs.detach().cpu()))
        all_labels = torch.cat((all_labels, labels.detach().cpu()))
        loss = F.nll_loss(torch.log(probs), labels,reduction='sum')         # Cross Entropy
        preds = np.argmax(out.detach().cpu().numpy(), axis=1) # [1 if x[1] > prediction_threshold else 0 for x in probs]
        bl = loss.detach().cpu().item()
        
        labels = batch.y.detach().cpu()
        hard_labels = (labels >= .5).astype('float')
        SASAs = batch.x[:,63].detach().cpu()
        
        ba = accuracy_score(hard_labels, preds)
        bm = mcc(hard_labels, preds)
        bpr = average_precision_score(hard_labels, probs[:,1])

        test_batch_loss += bl
        test_batch_acc  += ba
        test_batch_mcc  += bm
        test_batch_pr_auc += bpr
        np.save(prob_path + assembly_name, probs.detach().cpu().numpy())
        np.save(label_path + assembly_name, labels.detach().cpu().numpy())
        np.save(surface_path + assembly_name, SASAs.detach().cpu().numpy())
        
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
all_label_path = prepend + metric_dir + '/all_labels/'

if not os.path.isdir(all_prob_path):
    os.makedirs(all_prob_path)
if not os.path.isdir(all_label_path):
    os.makedirs(all_label_path)

np.savez(all_prob_path + 'all_probs', all_probs)
np.savez(all_label_path + 'all_labels', all_labels)

all_labels = (all_labels >= .5).astype('float')

fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
n_classes = 2

for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(all_labels[:, i],all_probs[:,i])
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

roc_path = prepend + metric_dir + '/roc_curves/' + model_name
if not os.path.isdir(roc_path):
    os.makedirs(roc_path)

np.savez(roc_path + "/roc_auc", roc_auc)
np.savez(roc_path + "/tpr", tpr)
np.savez(roc_path + "/fpr", fpr)
np.savez(roc_path + "/thresholds", thresholds)
