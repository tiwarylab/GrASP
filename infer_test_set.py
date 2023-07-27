import os
import argparse
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GATv2Conv

from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score
from sklearn.metrics import matthews_corrcoef as mcc
import torch

from GASP_dataset import GASPData
from model import GAT_model
from utils import distance_sigmoid, initialize_model

def k_fold(dataset:GASPData, prepend:str, fold_number:int):
    """Returns a boolean mask over the dataset that seperates it into training and validation portions
     by UniProt ID. Cross-validation (CV) splits were precomputed in    
    Stepniewska-Dziubinska, M.M., Zielenkiewicz, P. & Siedlecki, P. Improving detection of 
    protein-ligand binding sites with 3D segmentation. Sci Rep 10, 5035 (2020). 
    https://doi.org/10.1038/s41598-020-61860-

    Parameters
    ----------
    dataset : GASPData
        GASPData object represented a dataset.
    prepend : str
        Prepend path where CV splits are stored.
    fold_number : int
        Which CV split to use

    Returns
    -------
    Tuple of (torch.Tensor, torch.Tensor)
        A tuple of (training_mask, validation_mask) which can be used as boolean masks over the dataset.
    """    
    val_names    = np.loadtxt(prepend + "/splits/test_ids_fold"  + str(fold_number), dtype='str')
    train_names   = np.loadtxt(prepend + "/splits/train_ids_fold" + str(fold_number), dtype='str')
    
    train_indices, val_indices = [], []
    
    # Iterate over the raw dataset file names and check if they're in the training or validation file.
    for idx, name in enumerate(dataset.raw_file_names):
        if name[:4] in val_names: 
            val_indices.append(idx)
        if name[:4] in train_names:
            train_indices.append(idx)

    train_mask = torch.ones(len(dataset), dtype=torch.bool)
    val_mask = torch.ones(len(dataset), dtype=torch.bool)
    
    # Remove all val_indices from the training set and vice-versa
    train_mask[val_indices] = 0
    val_mask[train_mask] = 0

    return train_mask, val_mask

def parse():
    """Parse command-line arguments for evaluating site prediction on test sets.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate site prediction on test sets.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Saved model path
    parser.add_argument("model_path", nargs="?", default="train_full/trained_model_s_train_full_ag_multi_1680643832.8660116/cv_0/epoch_49",
    help="Path to the model from ./trained_models/")

    # Dataset options
    dataset_choises = ["val", "coach420", "coach420_mlig", "coach420_intersect","holo4k", "holo4k_mlig", "holo4k_intersect", "holo4k_chains", "production"]
    parser.add_argument("-s", "--infer_set", default="production", choices=dataset_choises, help="Test or production set.")
    parser.add_argument("-f", "--fold", type=int, default=0, help="Cross-validation fold, only used for validation set.")
    
    # Model architecture
    model_choices=["transformer", "transformer_gn", "transformer_in", "transformer_in_stats","transformer_pn", "transformer_gns", "transformer_aon", "transformer_no_norm", "gat", "gatv2"]
    parser.add_argument("-m", "--model", default="gatv2", choices=model_choices, help="GNN architecture to test.")
    
    # Model hyperparameters
    parser.add_argument("-sp", "--sigmoid_params", type=float, nargs=2, default=[5, 3], help="Parameters for sigmoid labels [label_midpoint, label_slope].")
    parser.add_argument("-wg", "--weight_groups", type=int, default=1, help="Number of weight-sharing groups.")
    parser.add_argument("-gl", "--group_layers", type=int, default=12, help="Number of layers per weight-sharing group.")
    parser.add_argument("-ag", "--aggregator", default="multi", choices=["mean", "sum", "multi"], help="GNN message aggregation operator.")
    
    # Prediction parameters
    parser.add_argument("-ao", "--all_atom_prediction", action="store_true", help="Option to perform inference on all atoms as opposed to solvent exposed.")
    parser.add_argument("-kh", "--k_hops", type=int, default=1, help="Number of hops for constructing a surface graph.")
    parser.add_argument("-st", "--sasa_threshold", type=float, default=1e-4, help="SASA above which atoms are considered on the surface.")
    
    parser.add_argument("-n", "--n_tasks", type=int, default=4, help="Number of cpu workers.")

    args = parser.parse_args()
    return args


def infer_test(args):
    """Perform inference and evaluation on a given test or validation set using the specified GNN model.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed command-line arguments. It should include the following attributes:
        - args.infer_set
        - args.model_path
        - args.fold
        - args.model
        - args.sigmoid_params
        - args.all_atom_prediction
        - args.k_hops
        - args.sasa_threshold
        - args.n_tasks  
        
        For details on arguments, refer to argument help strings.
        

    Returns
    -------
    None
    """
    prepend = str(os.getcwd())

    # Dataset Parameters
    set_to_use = args.infer_set
    sasa_threshold = args.sasa_threshold
    k_hops = args.k_hops
    
    # Inference Parameters
    surface_only = not args.all_atom_prediction
    label_midpoint, label_slope = args.sigmoid_params

    # Device Parameters
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    num_cpus = args.n_tasks
    print("The model will be using the following device:", device)
    print("The model will be using {} cpus.".format(num_cpus))

    # Load model
    model_name = args.model_path
    model = initialize_model(args)
    model_path = prepend + "/trained_models/" + model_name
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Initialize dataset for inference
    if set_to_use == 'val':
        print("Initializing Validation Set")
        path_to_dataset = prepend + '/scPDB_data_dir'
        metric_dir = '/test_metrics/validation'
        fold = args.fold

        data_set = GASPData(path_to_dataset, 
                            num_cpus, 
                            cutoff=5, 
                            surface_subgraph_hops=k_hops, 
                            sasa_threshold=sasa_threshold)
        # Select only validation datapoints
        _, val_mask = k_fold(data_set, prepend, fold) 
        data_set     = data_set[val_mask]
        
    else:  
        print(f"Initializing {set_to_use} set")
        path_to_dataset = f'{prepend}/benchmark_data_dir/{set_to_use}'
        metric_dir = f'/test_metrics/{set_to_use}'

        data_set = GASPData(path_to_dataset, 
                            num_cpus, 
                            cutoff=5, 
                            surface_subgraph_hops=k_hops, 
                            sasa_threshold=sasa_threshold)
        data_set.process()  # Reprocess dataset for safety
        
    val_dataloader = DataLoader(data_set, 
                                batch_size=1, 
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=num_cpus)

    # Lists to store metrics over the entire dataset
    test_epoch_loss = []
    test_epoch_acc = []
    test_epoch_mcc = []
    test_epoch_pr_auc = []

    # Tensors that store probabilities and labels for every atom in the dataset
    all_probs = torch.Tensor([])
    all_labels = torch.Tensor([])

    # Create directories to save probabilities, labels, solvent accessible surface area, and atom indices
    prob_path = prepend + metric_dir + '/probs/' + model_name + '/'             # Per complex atomic probabilities
    label_path = prepend + metric_dir + '/labels/' + model_name + '/'           # Per complex atomic labels
    all_prob_path = prepend + metric_dir + '/all_probs/' + model_name + '/'     # Probabilities for entire dataset
    all_label_path = prepend + metric_dir + '/all_labels/' + model_name + '/'   # Labels for entire dataset
    surface_path = prepend + metric_dir + '/SASAs/'
    index_path = prepend + metric_dir + '/indices/' + model_name + '/'

    if not os.path.isdir(prob_path):
        os.makedirs(prob_path)
    if not os.path.isdir(label_path):
        os.makedirs(label_path)
    if not os.path.isdir(all_prob_path):
        os.makedirs(all_prob_path)
    if not os.path.isdir(all_label_path):
        os.makedirs(all_label_path)
    if not os.path.isdir(surface_path):
        os.makedirs(surface_path)
    if not os.path.isdir(index_path):
        os.makedirs(index_path)

    print("Begining Evaluation")
    model.eval()
    with torch.no_grad():
        
        
        datapoint_loss = 0.0
        datapoint_accuracy = 0.0
        datapoint_mcc = 0.0
        datapoint_pr_auc = 0.0

        # Iterate over datapoints (batch size = 1)
        for batch, name in tqdm(val_dataloader, position=0, leave=True):
            # Compute continuous labels based on sigmoid of distance
            batch.y = distance_sigmoid(batch.y, label_midpoint, label_slope)
            batch.y = torch.stack([1-batch.y, batch.y], dim=1)
            labels = batch.y.to(device)
            
            # Complex Name
            assembly_name = name[0][:-4]  

            out, _ = model.forward(batch.to(device))

            # Mask out all atoms not on surface if surface_only for loss
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
            
            # Mask out all atoms if not on surface and surface_only for metrics
            labels = batch.y.detach().cpu()
            if surface_only:
                labels = labels[surf_mask.detach().cpu()]

            hard_labels = np.argmax(labels, axis=1)
            surf_masks = batch.surf_mask.detach().cpu()
            atom_indices = batch.atom_index.detach().cpu()
            
            ba = accuracy_score(hard_labels, preds)
            bm = mcc(hard_labels, preds)
            bpr = average_precision_score(hard_labels, probs.detach().cpu().numpy()[:,1])

            datapoint_loss += bl
            datapoint_accuracy  += ba
            datapoint_mcc  += bm
            datapoint_pr_auc += bpr
            np.save(prob_path + assembly_name, probs.detach().cpu().numpy())
            np.save(label_path + assembly_name, labels.detach().cpu().numpy())
            np.save(surface_path + assembly_name, surf_masks.detach().cpu().numpy())
            np.save(index_path + assembly_name, atom_indices.detach().cpu().numpy())

        test_epoch_loss.append(datapoint_loss/len(val_dataloader))
        test_epoch_acc.append(datapoint_accuracy/len(val_dataloader))
        test_epoch_mcc.append(datapoint_mcc/len(val_dataloader))
        test_epoch_pr_auc.append(datapoint_pr_auc/len(val_dataloader))
        print("Loss: {}".format(test_epoch_loss[-1]))
        print("Accu: {}".format(test_epoch_acc[-1]))
        print("MCC:  {}".format(test_epoch_mcc[-1]))
        print("PR AUC: {}".format(test_epoch_pr_auc[-1]))

    all_probs  =  all_probs.detach().cpu().numpy()
    all_labels = all_labels.detach().cpu().numpy()

    np.savez(all_prob_path + 'all_probs', all_probs)
    np.savez(all_label_path + 'all_labels', all_labels)

    all_hard_labels = (all_labels >= .5).astype('float')

    # AUC Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc%20curve
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


def infer_production(args):
    """Perform inference and evaluation on a given structure or set of structures using a pretrained GrASP model.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed command-line arguments. It should include the following attributes:
        - args.model_path
        - args.all_atom_prediction
        - args.k_hops
        - args.sasa_threshold
        - args.n_tasks
        
        For details on arguments, refer to argument help strings.
    
    Returns
    -------
    None
    """
    prepend = str(os.getcwd())

    # Dataset Parameters
    k_hops = args.k_hops
    sasa_threshold = args.sasa_threshold

    # Device Parameters
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    num_cpus = args.n_tasks
    print("The model will be using the following device:", device)
    print("The model will be using {} cpus.".format(num_cpus))

    # Load model
    model_name = args.model_path
    model = initialize_model(args)
    model_path = prepend + "/trained_models/" + model_name
    surface_only = not args.all_atom_prediction
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Initialize Dataset
    print("Initializing production set")    
    path_to_dataset = prepend + '/benchmark_data_dir/production'
    data_set = GASPData(path_to_dataset, num_cpus, cutoff=5, surface_subgraph_hops=k_hops, sasa_threshold=sasa_threshold)
    data_set.process()
    val_dataloader = DataLoader(data_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_cpus)

    # Stores probabilities of each atom for the entire dataset
    all_probs = torch.Tensor([])

    # Create directories to save probabilities, labels, solvent accessible surface area, and atom indices
    metric_dir = '/test_metrics/production'
    prob_path = prepend + metric_dir + '/probs/' + model_name + '/'             # Per atom atom probabilities for each complex
    all_prob_path = prepend + metric_dir + '/all_probs/' + model_name + '/'     # Per atom probabilities for entire dataset
    surface_path = prepend + metric_dir + '/SASAs/'
    index_path = prepend + metric_dir + '/indices/' + model_name + '/'

    if not os.path.isdir(prob_path):
        os.makedirs(prob_path)
    if not os.path.isdir(all_prob_path):
        os.makedirs(all_prob_path)
    if not os.path.isdir(surface_path):
        os.makedirs(surface_path)
    if not os.path.isdir(index_path):
        os.makedirs(index_path)

    print("Begining Evaluation")
    model.eval()
    with torch.no_grad():
        # For each datapoint in the production dataset, compute the probabilities
        for batch, name in tqdm(val_dataloader, position=0, leave=True):
            assembly_name = name[0][:-4]  

            out, _ = model.forward(batch.to(device))

            if surface_only:
                surf_mask = batch.surf_mask.to(device)
                out = out[surf_mask]

            probs = F.softmax(out, dim=-1) 

            surf_masks = batch.surf_mask.detach().cpu()
            atom_indices = batch.atom_index.detach().cpu()

            # Save probabilities, surface mask, and indices for each complex
            np.save(prob_path + assembly_name, probs.detach().cpu().numpy())
            np.save(surface_path + assembly_name, surf_masks.detach().cpu().numpy())
            np.save(index_path + assembly_name, atom_indices.detach().cpu().numpy())

    # Save probabilities for all complexes
    all_probs = all_probs.detach().cpu().numpy()

    np.savez(all_prob_path + 'all_probs', all_probs)


if __name__ == "__main__":
    args = parse()
    if args.infer_set == "production":
        infer_production(args)
    else:
        infer_test(args)
