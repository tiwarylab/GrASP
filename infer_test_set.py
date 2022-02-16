import os
import numpy as np
from datetime import datetime
import time

# import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import roc_curve, auc 
from torch.utils.tensorboard import SummaryWriter
import torch

from KLIFS_dataset import KLIFSData
from atom_wise_models import GATModelv1, GATModelv2, Two_Track_GATModel

prepend = str(os.getcwd()) + "/trained_models/"

###################################################################################
''' Some bits that are surrounded in comments like this can be used to temporarily
    modify the code to use the training set.'''
###################################################################################


########################## Change Me To Change The Model ##########################
model_name = "trained_model_1644710425.1063097/epoch_25"
model_path = prepend + model_name
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


# Other Parameters
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

num_cpus = 4
print("The model will be using the following device:", device)
print("The model will be using {} cpus.".format(num_cpus))

# model = GATModelv2(input_dim=43, output_dim=2)
# model = Two_Track_GATModel(input_dim=43, output_dim=2, drop_prob=0.1, left_aggr="max", right_aggr="mean").to(device)
model = Two_Track_GATModel(input_dim=88, output_dim=2, drop_prob=0.1, left_aggr="max", right_aggr="mean").to(device) 

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
 
prepend = str(os.getcwd())
print("Initializing Test Set")
#########################
data_set = KLIFSData(prepend + '/data_dir', num_cpus, cutoff=4)
train_mask, val_mask = k_fold(data_set, prepend, 1)
val_set     = data_set[val_mask]
val_dataloader = DataLoader(val_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=num_cpus)

#########################
# data_set = KLIFSData(prepend + '/test_data_dir', num_cpus, cutoff=5, force_process=False)
# data_set.process()
# data_set = torch.utils.data.Subset(data_set, np.random.choice(len(data_set), size = 5000, replace=False))
######################### data_set = KLIFSData(prepend + '/test_data_dir', num_cpus)
# Sticking with batch size 1 because it makes it easier to track predictions
# test_dataloader =  DataLoader(data_set, batch_size = 1, shuffle=False, pin_memory=True, 
#                               num_workers=num_cpus, persistent_workers=True)

test_epoch_loss = []
test_epoch_acc = []
test_epoch_mcc = []

all_probs = torch.Tensor([])
all_labels = torch.Tensor([])

#########################
if not os.path.isdir(prepend + '/train_metrics/test_probs/' + model_name + '/'):
    os.makedirs(prepend + '/train_metrics/test_probs/' + model_name + '/')
if not os.path.isdir(prepend + '/train_metrics/test_labels/'):
    os.makedirs(prepend + '/train_metrics/test_labels/')
#########################
# if not os.path.isdir(prepend + '/test_metrics/test_probs/' + model_name + '/'):
#     os.makedirs(prepend + '/test_metrics/test_probs/' + model_name + '/')
# if not os.path.isdir(prepend + '/test_metrics/test_labels/'):
#     os.makedirs(prepend + '/test_metrics/test_labels/')

print("Begining Evaluation")
model.eval()
with torch.no_grad():
    test_batch_loss = 0.0
    test_batch_acc = 0.0
    test_batch_mcc = 0.0
    # for batch, name in test_dataloader:
    for batch, name in val_dataloader:
        
        labels = batch.y.to(device)
        assembly_name = name[0][:-4]
        # print(assembly_name)

        out = model.forward(batch.to(device))
        probs = F.softmax(out, dim=-1)
        all_probs = torch.cat((all_probs, probs.detach().cpu()))
        all_labels = torch.cat((all_labels, labels.detach().cpu()))
        loss = F.nll_loss(torch.log(probs), labels,reduction='sum')         # Cross Entropy
        preds = np.argmax(out.detach().cpu().numpy(), axis=1)
        bl = loss.detach().cpu().item()

        labels = batch.y.detach().cpu()
        
        ba = accuracy_score(labels, preds)
        bm = mcc(labels, preds)

        test_batch_loss += bl
        test_batch_acc  += ba
        test_batch_mcc  += bm
        # print("Test Batch Loss:", bl)
        # print("Test Batch Accu:", ba)
        # print("Test Batch MCC:", bm)
        #########################
        np.save(prepend + '/train_metrics/test_probs/' + model_name + '/' + assembly_name, probs.detach().cpu().numpy())
        np.save(prepend + '/train_metrics/test_labels/' + assembly_name, labels.detach().cpu().numpy())
        #########################
        # np.save(prepend + '/test_metrics/test_probs/' + model_name + '/' + assembly_name, probs.detach().cpu().numpy())
        # np.save(prepend + '/test_metrics/test_labels/' + assembly_name, labels.detach().cpu().numpy())
        
        # writer.add_scalar('Batch_Loss/test', bl, test_batch_num)
        # writer.add_scalar('Batch_Acc/test',  ba,  test_batch_num)
        # writer.add_scalar('Batch_Acc/MCC',  bm,  test_batch_num)


    # test_epoch_loss.append(test_batch_loss/len(test_dataloader))
    # test_epoch_acc.append(test_batch_acc/len(test_dataloader))
    # test_epoch_mcc.append(test_batch_mcc/len(test_dataloader))
    test_epoch_loss.append(test_batch_loss/len(val_dataloader))
    test_epoch_acc.append(test_batch_acc/len(val_dataloader))
    test_epoch_mcc.append(test_batch_mcc/len(val_dataloader))
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


#########################
# if not os.path.isdir(prepend + '/train_metrics/all_probs/'):
#     os.makedirs(prepend + '/train_metrics/all_probs/')
# if not os.path.isdir(prepend + '/train_metrics/all_labels/'):
#     os.makedirs(prepend + '/train_metrics/all_labels/')

# np.savez(prepend + "/train_metrics/all_probs/" + model_name, all_probs)
# np.savez(prepend + "/train_metrics/all_labels/" + model_name, all_labels)
#########################
if not os.path.isdir(prepend + '/test_metrics/all_probs/' + model_name):
    os.makedirs(prepend + '/test_metrics/all_probs/' + model_name)
if not os.path.isdir(prepend + '/test_metrics/all_labels/' + model_name):
    os.makedirs(prepend + '/test_metrics/all_labels/' + model_name) 

np.savez(prepend + "/test_metrics/all_probs/" + model_name, all_probs)
np.savez(prepend + "/test_metrics/all_labels/" + model_name, all_labels)

all_labels = np.array([[0,1] if x == 1 else [1,0] for x in all_labels])

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

#########################
# if not os.path.isdir(prepend + '/train_metrics/roc_curves/' + model_name):
#     os.makedirs(prepend + '/train_metrics/roc_curves/' + model_name)

# np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/roc_auc", roc_auc)
# np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/tpr", tpr)
# np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/fpr", fpr)
# np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/thresholds", thresholds)
#########################

if not os.path.isdir(prepend + '/test_metrics/roc_curves/' + model_name):
    os.makedirs(prepend + '/test_metrics/roc_curves/' + model_name)

np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/roc_auc", roc_auc)
np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/tpr", tpr)
np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/fpr", fpr)
np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/thresholds", thresholds)