import os
import numpy as np
import sys
import argparse


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import matthews_corrcoef as mcc

from torch.utils.tensorboard import SummaryWriter
import time

from GASP_dataset import GASPData
from utils import distance_sigmoid, initialize_model

job_start_time = time.time()
prepend = str(os.getcwd())

def k_fold(dataset:GASPData,train_path:str, val_path, i):
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
    Tuple of (torch.Tensor, torch.Tensor, int)
        A tuple of (training_mask, validation_mask, fold_number) which can be used as boolean masks over the dataset.
    """    
    val_names    = np.loadtxt(val_path, dtype=str)
    train_names   = np.loadtxt(train_path, dtype=str)
    
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

    return (dataset[train_mask], dataset[val_mask], i)

def main(node_noise_std : float, training_split='cv'):
    """Train a GrASP model.

    Parameters
    ----------
    node_noise_std : float
        The standard distribution of the noise added to nodes for the Noisy Nodes protocol. 
        For more information see:
        Godwin, J.; Schaarschmidt, M.; Gaunt, A. L.; Sanchez-
        Gonzalez, A.; Rubanova, Y.; Veliˇckovi ́c, P.; Kirkpatrick, J.;
        Battaglia, P. Simple GNN Regularisation for 3D Molecular Prop-
        erty Prediction and Beyond. International Conference on Learn-
        ing Representations. 2022.
    training_split : str, optional
        Which dataset to use. If 'cv' use the cross validation splits, by default 'cv'
    """
    # Training Hyperparameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    class_loss_weight = args.class_loss_weight
    label_smoothing = args.label_smoothing
    head_loss_weight = args.head_loss_weight
    label_midpoint, label_slope = args.sigmoid_params
    
    # Dataset Parameters
    k_hops = args.k_hops
    sasa_threshold = args.sasa_threshold
    fold_number = args.fold
    
    # Device Parameters
    num_cpus = args.n_tasks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=torch.FloatTensor(class_loss_weight).to(device))
    
    head_loss_weight = torch.tensor(head_loss_weight).to(device)

    # Initialize Training Split for Each Dataset
    if training_split == 'chen':
        do_validation = True
        train_set = GASPData(f'{prepend}/benchmark_data_dir/chen11', num_cpus, cutoff=5, surface_subgraph_hops=k_hops, sasa_threshold=sasa_threshold)
        val_set = GASPData(f'{prepend}/benchmark_data_dir/joined', num_cpus, cutoff=5, surface_subgraph_hops=k_hops, sasa_threshold=sasa_threshold)

        gen = zip([train_set], [val_set], [0])
    
    else:
        data_set = GASPData(prepend + '/scPDB_data_dir', num_cpus, cutoff=5, surface_subgraph_hops=k_hops, sasa_threshold=sasa_threshold)
        
        do_validation = False
        if training_split == 'cv':
            do_validation = True
            val_paths = []
            train_paths = []
            
            val_paths.append(prepend + "/splits/test_ids_fold"  + str(fold_number))
            train_paths.append(prepend + "/splits/train_ids_fold" + str(fold_number))
            data_points = zip(train_paths,val_paths)

            gen = (k_fold(data_set, train_path, val_path, fold_number) for (train_path, val_path) in data_points)

        elif training_split == 'train_full':
            do_validation = False
            gen = zip([data_set], [0], [0])

        else:
            train_prefix = '/splits/train_ids_'
            # Training splits for different test sets
            if training_split == 'coach420':
                train_names = np.loadtxt(prepend + f'{train_prefix}coach420_uniprot', dtype=str)
            elif training_split == 'coach420_mlig':
                train_names = np.loadtxt(prepend + f'{train_prefix}coach420(mlig)_uniprot', dtype=str)
            elif training_split == 'holo4k':
                train_names = np.loadtxt(prepend + f'{train_prefix}holo4k_uniprot', dtype=str)
            elif training_split == 'holo4k_mlig':
                train_names = np.loadtxt(prepend + f'{train_prefix}holo4k(mlig)_uniprot', dtype=str)
            train_indices = []
            # Iterate over training names and add to the train set if included in the split
            for idx, name in enumerate(data_set.raw_file_names):
                if name.split('_')[0] in train_names:
                    train_indices.append(idx)
            train_mask = torch.zeros(len(data_set), dtype=torch.bool)
            train_mask[train_indices] = 1
            gen = zip([data_set[train_mask]],[data_set[torch.zeros(len(data_set),dtype=torch.bool)]],[0])

    for train_set, val_set, cv_iteration in gen:
        model = initialize_model(args)
        model =  DataParallel(model)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose=True)
        
        train_dataloader = DataListLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_cpus)
        if do_validation: val_dataloader = DataListLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_cpus)
        # Track Training Statistics
        training_epoch_loss = []
        training_epoch_acc = []
        training_epoch_mcc = []
        training_epoch_auc = []
        training_epoch_pr_auc = []

        val_epoch_loss = []
        val_epoch_acc = []
        val_epoch_mcc = []
        val_epoch_auc = []
        val_epoch_pr_auc = []

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
            training_batch_pr_auc = 0.0
            for batch in train_dataloader:
                # Strip batches of raw file names
                batch = list(map(lambda x: x[0].to(device), batch))
                
                unperturbed_x = torch.cat([data.x for data in batch]).clone().detach().to(device)
                for data in batch:
                    data.x += (data.x.std(dim=0) * node_noise_std) * torch.randn_like(data.x)   # Add noise for Noisy Nodes Regularization
                    data.y = distance_sigmoid(data.y, label_midpoint, label_slope)              # Compute soft labels with sigmoid of distnace
                    data.y = torch.stack([1-data.y, data.y], dim=1)
                    
                labels  = torch.cat([data.y for data in batch]).clone().detach().cpu().numpy()  # Labels for metrics
                y       = torch.cat([data.y for data in batch]).to(device)                      # Labels for training
                surf_mask = torch.cat([data.surf_mask for data in batch]).to(device)
            
                optimizer.zero_grad(set_to_none=True)
                out, out_recon = model.forward(batch)

                if surface_only:
                    labels = labels[surf_mask.detach().cpu().numpy()]
                    y = y[surf_mask]
                    unperturbed_x = unperturbed_x[surf_mask]

                    out = out[surf_mask]
                    out_recon = out_recon[surf_mask]

                weighted_xent_l = head_loss_weight[0] * loss_fn(out,y),  
                mse_l           = head_loss_weight[1] * F.mse_loss(out_recon, unperturbed_x)          
                
                loss = weighted_xent_l + mse_l
                loss.backward() 
                optimizer.step()

                probs = out.softmax(dim=-1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)
                hard_labels = np.argmax(labels, axis=1)

                l = loss.detach().cpu().item()
                
                # Compute and save batch training statistics
                bl = l 
                ba = accuracy_score(hard_labels, preds)
                bm = mcc(hard_labels, preds)
                bc = roc_auc_score(hard_labels, probs[:,1])
                bpr = average_precision_score(hard_labels, probs[:,1])
                training_batch_loss += bl
                training_batch_acc  += ba
                training_batch_mcc  += bm
                training_batch_auc  += bc
                training_batch_pr_auc += bpr

                writer.add_scalar('Batch_Loss/Train', bl, train_batch_num)
                writer.add_scalar('Batch_ACC/Train',  ba,  train_batch_num)
                writer.add_scalar('Batch_MCC/Train',  bm,  train_batch_num)
                writer.add_scalar('Batch_AUC/Train',  bc,  train_batch_num)
                writer.add_scalar('Batch_PR_AUC/Train', bpr, train_batch_num)
                train_batch_num += 1
                
            scheduler.step()
            print("******* EPOCH END, EPOCH TIME: {}".format(time.time() - epoch_start))
            
            # Compute and save epoch training statistics
            training_epoch_loss.append(training_batch_loss/len(train_dataloader))
            training_epoch_acc.append(training_batch_acc/len(train_dataloader))
            training_epoch_mcc.append(training_batch_mcc/len(train_dataloader))
            training_epoch_auc.append(training_batch_auc/len(train_dataloader))
            training_epoch_pr_auc.append(training_batch_pr_auc/len(train_dataloader))
            print("Training Epoch {} Loss: {}".format(epoch, training_epoch_loss[-1]))
            print("Training Epoch {} Accu: {}".format(epoch, training_epoch_acc[-1]))
            print("Training Epoch {} MCC: {}".format(epoch, training_epoch_mcc[-1]))
            print("Training Epoch {} AUC: {}".format(epoch, training_epoch_auc[-1]), flush=True)
            print("Training Epoch {} PR AUC: {}".format(epoch, training_epoch_pr_auc[-1]), flush=True)
            writer.add_scalar('Epoch_Loss/Train', training_epoch_loss[-1], train_epoch_num)
            writer.add_scalar('Epoch_ACC/Train',  training_epoch_acc[-1],  train_epoch_num)
            writer.add_scalar('Epoch_MCC/Train',  training_epoch_mcc[-1],  train_epoch_num)
            writer.add_scalar('Epoch_AUC/Train',  training_epoch_auc[-1],  train_epoch_num)
            writer.add_scalar('Epoch_PR_AUC/Train', training_epoch_pr_auc[-1], train_epoch_num)

            # Save model checkpoint
            if not os.path.isdir("./trained_models/{}/trained_model_{}/cv_{}/".format(training_split, model_id, cv_iteration)):
                os.makedirs("./trained_models/{}/trained_model_{}/cv_{}/".format(training_split, model_id, cv_iteration))
            torch.save(model.module.state_dict(), "./trained_models/{}/trained_model_{}/cv_{}/epoch_{}".format(training_split, model_id, cv_iteration, train_epoch_num))
            
            train_epoch_num += 1

            if do_validation:
                model.eval()
                with torch.no_grad():
                    val_batch_loss = 0.0
                    val_batch_acc = 0.0
                    val_batch_mcc = 0.0
                    val_batch_auc = 0.0
                    val_batch_pr_auc = 0.0

                    for batch in val_dataloader:
                        # Strip batches of raw file names
                        batch = list(map(lambda x: x[0].to(device), batch))
                
                        # Note: we do not apply the Noisy Nodes protocol to validation samples
                        
                        for data in batch:
                            data.y = distance_sigmoid(data.y, label_midpoint, label_slope)              # Compute soft labels with sigmoid of distnace
                            data.y = torch.stack([1-data.y, data.y], dim=1)
                        labels  = torch.cat([data.y for data in batch]).clone().detach().cpu().numpy()  # Labels for metrics
                        y       = torch.cat([data.y for data in batch]).to(device)                      # Labels for loss computation
                        surf_mask = torch.cat([data.surf_mask for data in batch]).to(device)            
                    
                        optimizer.zero_grad(set_to_none=True)

                        out, _ = model.forward(batch)

                        if surface_only:
                            labels = labels[surf_mask.detach().cpu().numpy()]
                            y = y[surf_mask]
                            out = out[surf_mask]

                        loss = loss_fn(out,y)
                        probs = out.softmax(dim=-1).detach().cpu().numpy()
                        preds = np.argmax(probs, axis=1)
                        hard_labels = np.argmax(labels, axis=1) # converting to binary labels for metrics
                        
                        # Compute and save batch training statistics
                        bl = loss.detach().cpu().item()
                        ba = accuracy_score(hard_labels, preds)
                        bm = mcc(hard_labels, preds)
                        bc = roc_auc_score(hard_labels, probs[:,1])
                        bpr = average_precision_score(hard_labels, probs[:,1])
                        val_batch_loss += bl
                        val_batch_acc  += ba
                        val_batch_mcc  += bm
                        val_batch_auc  += bc
                        val_batch_pr_auc += bpr

                        writer.add_scalar('Batch_Loss/Val', bl, val_batch_num)
                        writer.add_scalar('Batch_ACC/Val',  ba,  val_batch_num)
                        writer.add_scalar('Batch_MCC/Val',  bm,  val_batch_num)
                        writer.add_scalar('Batch_AUC/Val',  bc,  val_batch_num)
                        writer.add_scalar('Batch_PR_AUC/Val', bpr, val_batch_num)
                        val_batch_num += 1

                    # Compute and save epoch training statistics
                    val_epoch_loss.append(val_batch_loss/len(val_dataloader))
                    val_epoch_acc.append(val_batch_acc/len(val_dataloader))
                    val_epoch_mcc.append(val_batch_mcc/len(val_dataloader))
                    val_epoch_auc.append(val_batch_auc/len(val_dataloader))
                    val_epoch_pr_auc.append(val_batch_pr_auc/len(val_dataloader))
                    print("Validation Epoch {} Loss: {}".format(epoch, val_epoch_loss[-1]))
                    print("Validation Epoch {} Accu: {}".format(epoch, val_epoch_acc[-1]))
                    print("Validation Epoch {} MCC: {}".format(epoch, val_epoch_mcc[-1]))
                    print("Validation Epoch {} AUC: {}".format(epoch, val_epoch_auc[-1]))
                    print("Validation Epoch {} PR AUC: {}".format(epoch, val_epoch_pr_auc[-1]))
                    writer.add_scalar('Epoch_Loss/Val', val_epoch_loss[-1], val_epoch_num)
                    writer.add_scalar('Epoch_ACC/Val',  val_epoch_acc[-1],  val_epoch_num)
                    writer.add_scalar('Epoch_MCC/Val',  val_epoch_mcc[-1],  val_epoch_num)
                    writer.add_scalar('Epoch_AUC/Val',  val_epoch_auc[-1],  val_epoch_num)
                    writer.add_scalar('Epoch_PR_AUC/Val',  val_epoch_pr_auc[-1],  val_epoch_num)

                    val_epoch_num += 1

        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN for binding site prediction.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--training_split", default="cv", choices=["cv", "train_full", "coach420", "coach420_mlig", "holo4k", "holo4k_mlig", "chen"], help="Training set.")
    parser.add_argument("-f", "--fold", type=int, default=0, help="Cross-validation fold, only used for -s cv.")
    parser.add_argument("-nn", "--node_noise_std", type=float, default=0.02, help="NoisyNodes standard deviation.")
    parser.add_argument("-m", "--model", default="gatv2", choices=["gat", "gatv2"], help="GNN architecture to train.")
    parser.add_argument("-e", "--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.005, help="Adam learning rate.")
    parser.add_argument("-cw", "--class_loss_weight", type=float, nargs=2, default=[1.0, 1.0], help="Loss weight for [negative, positive] classes.")
    parser.add_argument("-ls", "--label_smoothing", type=float, default=0, help="Level of label smoothing.")
    parser.add_argument("-hw", "--head_loss_weight", type=float, nargs=2, default=[.9,.1], help="Weight of the loss functions for the [inference, reconstruction] heads.")
    parser.add_argument("-sp", "--sigmoid_params", type=float, nargs=2, default=[5, 3], help="Parameters for sigmoid labels [label_midpoint, label_slope].")
    parser.add_argument("-wg", "--weight_groups", type=int, default=1, help="Number of weight-sharing groups.")
    parser.add_argument("-gl", "--group_layers", type=int, default=12, help="Number of layers per weight-sharing group.")
    parser.add_argument("-ag", "--aggregator", default="multi", choices=["mean", "sum", "multi"], help="GNN message aggregation operator.")
    parser.add_argument("-ao", "--all_atom_prediction", action="store_true", help="Option to perform inference on all atoms as opposed to solvent exposed.")
    parser.add_argument("-kh", "--k_hops", type=int, default=1, help="Number of hops for constructing a surface graph.")
    parser.add_argument("-st", "--sasa_threshold", type=float, default=1e-4, help="SASA above which atoms are considered on the surface.")
    parser.add_argument("-n", "--n_tasks", type=int, default=8, help="Number of cpu workers.")
    args = parser.parse_args()
    argstring='_'.join(sys.argv[1:]).replace('-','')
    model_id = f'{argstring}_{str(job_start_time)}'

    node_noise_std = args.node_noise_std
    training_split = args.training_split
    surface_only = not args.all_atom_prediction

    print("Training with noise with std", node_noise_std, "and mean 0 added to nodes.")
    
    main(node_noise_std, training_split)
