import numpy as np
import MDAnalysis as mda
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from scipy.spatial import ConvexHull, HalfspaceIntersection, Delaunay
from scipy.optimize import linprog
import os
from tqdm import tqdm
from glob import glob

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import roc_curve, auc 
import time

'''
JOBS:
54794038: epoch 46 validation
killed:54794030 : epoch 46 chen
54793794: epoch 23 val
killed:54793936: epoch 23 chen

54795966: epoch 46 chen
'''





def center_of_mass(coords, masses):
    return np.sum(coords*np.tile(masses, (3,1)).T, axis=0)/np.sum(masses)

def DCA_dist(center, lig_coords):
    distances = np.sqrt(np.sum((center - lig_coords)**2, axis=1))
    shortest = np.min(distances)
    
    return shortest

def sort_clusters(cluster_ids, probs, labels):
    c_probs = []
    unique_ids = np.unique(cluster_ids)

    for c_id in unique_ids[unique_ids >= 0]:
        c_prob = np.mean(probs[:,1][labels][cluster_ids==c_id])
        c_probs.append(c_prob)

    c_order = np.argsort(c_probs)

    sorted_ids = -1*np.ones(cluster_ids.shape)
    for new_c in range(len(c_order)):
        old_c = c_order[new_c]
        sorted_ids[cluster_ids == old_c] = new_c
        
    return sorted_ids

def cluster_atoms(all_coords, predicted_probs, threshold=.5, quantile=.3, **kwargs):
    predicted_labels = predicted_probs[:,1] > threshold
    if np.sum(predicted_labels) == 0:
        # No positive predictions were made with specified cutoff
        return None, None, None
    bind_coords = all_coords[predicted_labels]
    if bind_coords.shape[0] != 1:
        bw = estimate_bandwidth(bind_coords, quantile=quantile)
        if bw == 0:
            bw = 1e-17
        try:
            ms_clustering = MeanShift(bandwidth=bw, **kwargs).fit(bind_coords)
        except Exception as e:
            print(bind_coords, flush=True)
            raise e
        cluster_ids = ms_clustering.labels_
        
        sorted_ids = sort_clusters(cluster_ids, predicted_probs, predicted_labels)
    else:
        # Under rare circumstances only one atom may be predicted as the binding pocket. In this case
        # the clustering fails so we'll just call this one atom our best 'cluster'.
        sorted_ids = [0]

    all_ids = -1*np.ones(predicted_labels.shape)
    all_ids[predicted_labels] = sorted_ids

    # print(sorted_ids)
    # print(all_ids)
    return bind_coords, sorted_ids, all_ids

def hull_center(hull):
    hull_com = np.zeros(3)
    tetras = Delaunay(hull.points[hull.vertices])

    for i in range(len(tetras.simplices)):
        tetra_verts = tetras.points[tetras.simplices][i]

        a, b, c, d = tetra_verts
        a, b, c = a - d, b - d, c - d
        tetra_vol = np.abs(np.linalg.det([a, b, c])) / 6

        tetra_com = np.mean(tetra_verts, axis=0)

        hull_com += tetra_com * tetra_vol

    hull_com = hull_com / hull.volume
    return hull_com

def get_centroid(coords):
    return np.mean(coords, axis=0)

def DCA_dist(center, lig_coords):
    distances = np.sqrt(np.sum((center - lig_coords)**2, axis=1))
    shortest = np.min(distances)
    
    return shortest

def cheb_center(halfspaces):
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
    (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    center = res.x[:-1]
    radius = res.x[-1]
    
    return center, radius

def hull_jaccard(hull1, hull2, intersect_hull):
    intersect_vol = intersect_hull.volume
    union_vol = hull1.volume + hull2.volume - intersect_hull.volume
    return intersect_vol/union_vol

def volumetric_overlap(hull1, hull2):
    halfspaces = np.append(hull1.equations, hull2.equations, axis=0)
    center, radius = cheb_center(halfspaces)
    
    if radius <= 1e-5: # We use an epsilon here because cheb_center will return a value requiring more accruacy than we're using
        return 0

    half_inter = HalfspaceIntersection(halfspaces, center)
    intersect_hull = ConvexHull(half_inter.intersections)
    
    jaccard = hull_jaccard(hull1, hull2, intersect_hull)
    
    return jaccard


def multi_site_metrics(prot_coords, lig_coord_list, ligand_mass_list, predicted_probs, site_coords_list, top_n_plus=0, threshold=.5, quantile=.3, cluster_all=False):
    """Cluster multiple binding sites and calculate distance from true site center, distance from ligand and volumetric overlap with true site 

    Parameters
    ----------
    prot_coords : numpy array
        Protein atomic coordinates.
        
    lig_coord_list : list of numpy arrays
        Ligand atomic coordinates for each ligand.
        
    ligand_mass_list: list of numpy arrays
        Ligand atomic masses for each ligand.

    predicted_probs : list of numpy arrays
        Class probabilities for not site in column 0 and site in column 1.

    site_coords_list : list of numpy arrays
        Protein atomic coordinates for each site, listed in the same order as the ligands.
        
    top_n_plus : int
        Number of predicted sites to include compared to number of true sites (eg. 1 means 1 more predicted).

    threshold : float
        Probability threshold to predict binding site atoms.

    quantile : float
        Quantile used in bandwidth selection for mean shift clustering.

    cluster_all : bool
        Whether to assign points outside kernels to the nearest cluster or leave them unlabeled.

    Returns
    -------
    DCC_lig: list
        List of distances from predicted site center to ligand center of mass. 
    
    DCC_site: list
        List of distances from predicted site center to true center. 
        
    DCA: list
        List of closest distances from predicted site center to any ligand heavy atom. 

    volumentric_overlaps: list
        Jaccard similarity between predicted site convex hull and true site convex hull. 

    """
    bind_coords, sorted_ids, _ = cluster_atoms(prot_coords, predicted_probs, threshold=threshold, quantile=quantile, cluster_all=cluster_all)


    true_hull_list = [ConvexHull(true_points) for true_points in site_coords_list]
    true_center_list = [hull_center(true_hull) for true_hull in true_hull_list]
    
    ligand_center_list = [center_of_mass(lig_coord_list[i], ligand_mass_list[i]) for i in range(len(lig_coord_list))]

    DCC_lig = []
    DCC_site = []
    DCA = []
    volumetric_overlaps = []
    
    
    top_ids = np.unique(sorted_ids)[::-1][:len(site_coords_list)+top_n_plus]
    predicted_points_list = []
    predicted_hull_list = []
    predicted_center_list = []
    
    for c_id in top_ids:
        if c_id != None:
            if c_id >= 0:
                predicted_points = bind_coords[sorted_ids == c_id]
                predicted_points_list.append(predicted_points)
                if len(predicted_points) < 4:  # You need four points to define a convex hull so we'll say the overlap is 0
                    predicted_center_list.append(get_centroid(predicted_points))
                    predicted_hull_list.append(None)
                else:
                    predicted_hull = ConvexHull(predicted_points)
                    predicted_hull_list.append(ConvexHull(predicted_points))
                    predicted_center_list.append(hull_center(predicted_hull))
    if len(predicted_center_list) > 0:            
        center_dist_matrix = np.zeros([len(true_center_list), len(predicted_center_list)]) # changed from [_, len(ligand_center_list)]
        for index, x in np.ndenumerate(center_dist_matrix):
            true_ind, pred_ind = index
            ligand_center = ligand_center_list[true_ind]
            predicted_center = predicted_center_list[pred_ind]
            center_dist_matrix[index] = np.sqrt(np.sum((predicted_center - ligand_center)**2))
                  
        print(center_dist_matrix)
        closest_predictions = np.argmin(center_dist_matrix, axis=1)
        site_pairs = np.column_stack([np.arange(len(closest_predictions)), closest_predictions])
                                    
        for pair in site_pairs:
            true_ind, pred_ind = pair
            # DCC_lig.append(center_dist_matrix[pair]) change to:
            DCC_lig.append(center_dist_matrix[true_ind][pred_ind])
            
            true_center = true_center_list[true_ind]
            predicted_center = predicted_center_list[pred_ind]
            DCC_site.append(np.sqrt(np.sum((predicted_center - true_center)**2)))
                                    
            lig_coords = lig_coord_list[true_ind]
            DCA.append(DCA_dist(predicted_center, lig_coords))

            true_hull = true_hull_list[true_ind]
            predicted_hull = predicted_hull_list[pred_ind]                    
            if predicted_hull != None: 
                volumetric_overlaps.append(volumetric_overlap(predicted_hull, true_hull))
            elif true_hull == None:
                raise ValueError ("There were < 3 atoms in your true site label. The indicates that the associated ligand is not burried.")
            else:
                volumetric_overlaps.append([])   
    return DCC_lig, DCC_site, DCA, volumetric_overlaps

def compute_metrics_for_all(top_n_plus=0, threshold = 0.5, path_to_mol2='/test_data_dir/mol2/', path_to_labels = '/test_metrics/'):
    DCC_lig_list = []
    DCC_site_list = []
    DCA_list = []
    volumentric_overlaps_list = []
    
    no_prediction_count = 0

    for file in tqdm(os.listdir(prepend + path_to_labels + 'test_probs/' + model_name + '/'),  position=0, leave=True): 
        # print(file)
        # print(type(file))
        assembly_name = file.split('.')[-2]
        try:
            trimmed_protein = mda.Universe(prepend + path_to_mol2 + assembly_name + '.mol2')
            labels = np.load(prepend + path_to_labels + 'test_labels/' + assembly_name + '.npy')
            probs = np.load(prepend + path_to_labels + 'test_probs/' + model_name + '/' + assembly_name + '.npy')

            lig_coord_list = []
            ligand_mass_list = []
            
            site_coords_list = []
            
            # for file_path in sorted(glob(prepend + "/scPDB_data_dir/ready_to_parse_mol2/" + assembly_name + '/*')):
            for file_path in sorted(glob(prepend + "/benchmark_data_dir/unprocessed_mol2/" + assembly_name + '/*')):
                if 'ligand' in file_path.split('/')[-1] and not 'site' in file_path.split('/')[-1]:
                    ligand = mda.Universe(file_path).select_atoms("not type H")
                    lig_coord_list.append(list(ligand.atoms.positions))
                    ligand_mass_list.append(list(ligand.atoms.masses))
                elif 'site_for_ligand' in file_path.split('/')[-1]:
                    site = mda.Universe(file_path).select_atoms("not type H")
                    site_coords_list.append(site.atoms.positions)
            DCC_lig, DCC_site, DCA, volumentric_overlaps = multi_site_metrics(trimmed_protein.atoms.positions, lig_coord_list, ligand_mass_list, probs, site_coords_list, top_n_plus=top_n_plus, threshold=threshold, quantile=.3, cluster_all=False)
            if DCC_lig == [] or DCC_site == [] or DCA == [] or volumentric_overlaps == []: 
                no_prediction_count += 1
            DCC_lig_list.append(DCC_lig)
            DCC_site_list.append(DCC_site)
            DCA_list.append(DCA)
            volumentric_overlaps_list.append(volumentric_overlaps)
            

        except Exception as e:
            print(assembly_name, flush=True)
            raise e
        
    return DCC_lig_list, DCC_site_list, DCA_list, volumentric_overlaps_list, no_prediction_count

#######################################################################################
# model_name = "trained_model_1640072931.267488_epoch_49"
# model_name = "trained_model_1640067496.5729342_epoch_30"
# model_name = "trained_model_1642111399.8650987/epoch_33"
# model_name = "trained_model_1644710425.1063097/epoch_25"
# model_name = "trained_model_1645166373.4874966/epoch_17"      # 5 Angs JK
# model_name = "trained_model_1645166379.4346104/epoch_18"      # 5 Angs no JK
# model_name = "trained_model_1645478750.6828046/epoch_28"      # 5 Angs JK, Gat GIN Hybrid
# model_name = "trained_model_1646263201.0032232/epoch_28"        # Added skip con from preprocessing to postprocessing. Added BN before postproccessing
# model_name = "trained_model_1646775694.0918303/epoch_49"
# model_name = "trained_model_1647199519.6304853/epoch_49" # Noise Added to Node Features During Training Var = 0.2, Mean = 0, no second loss func
# model_name = "trained_model_1647218964.5406673/epoch_49" # Noisy Nodes With MSE loss
# model_name = "/trained_model_hybrid_1g8/epoch_18"
# model_name = "trained_model_1648747746.262174/epoch_26" # 1g12 Mean Self Edges Epoch 26, scPDB Dataset
# model_name = "trained_model_1g12_null_self_edges/epoch_49"
# model_name = "trained_model_1g12_mean_self_edges/epoch_49"
model_name = "trained_model_1650260810.482072/epoch_46" # Old params, new labeling, ob

data_dir = 'benchmark_data_dir'

prepend = str(os.getcwd())
threshold_lst = [0.4, 0.45, 0.5]
compute_optimal = True
top_n_plus=2

set_to_use = "chen" #"chen"|"val"

#######################################################################################
if compute_optimal:
    if set_to_use == "chen":
        all_probs  = np.load(prepend + "/test_metrics/all_probs/" + model_name + ".npz")['arr_0']
        all_labels = np.load(prepend + "/test_metrics/all_labels/" + model_name + ".npz")['arr_0']
    elif set_to_use == "val":
        all_probs  = np.load(prepend + "/train_metrics/all_probs/" + model_name + ".npz")['arr_0']
        all_labels = np.load(prepend + "/train_metrics/all_labels/" + model_name + ".npz")['arr_0']
    start = time.time()

    binarized_labels = np.array([[0,1] if x == 1 else [1,0] for x in all_labels])
    # Compute roc, auc and optimal threshold
    all_probs = np.array(all_probs, dtype=object)

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    n_classes = 2

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(binarized_labels[:, i],all_probs[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binarized_labels.ravel(), all_probs.ravel())
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

    if set_to_use =="chen":
        if not os.path.isdir(prepend + '/test_metrics/roc_curves/' + model_name):
            os.makedirs(prepend + '/test_metrics/roc_curves/' + model_name)
    elif set_to_use == "val":
        if not os.path.isdir(prepend + '/train_metrics/roc_curves/' + model_name):
            os.makedirs(prepend + '/train_metrics/roc_curves/' + model_name)

    # Find optimal threshold
    gmeans = np.sqrt(tpr[1] * (1-fpr[1]))
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[1][ix]

    print('Best Threshold=%f, G-Mean=%.3f' % (optimal_threshold, gmeans[ix]))
    print("Micro Averaged AUC:", roc_auc["micro"])
    print("Macro Averaged AUC:", roc_auc["macro"])
    print("Negative Class AUC:", roc_auc[0])
    print("Positive Class AUC:", roc_auc[1])

if set_to_use == "chen":
    np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/roc_auc", roc_auc)
    np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/tpr", tpr)
    np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/fpr", fpr)
    np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/thresholds", thresholds)
elif set_to_use == "val":  
    np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/roc_auc", roc_auc)
    np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/tpr", tpr)
    np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/fpr", fpr)
    np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/thresholds", thresholds)
    print("Done. {}".format(time.time()- start))
    threshold_lst.insert(0, optimal_threshold)
    
#######################################################################################

#######################################################################################
for threshold in threshold_lst:
    print("Calculating overlap and center distance metrics for "+str(threshold)+" threshold.", flush=True)
    start = time.time()
    # DCC_lig, DCC_site, DCA, volumentric_overlaps, no_prediction_count = compute_metrics_for_all(top_n_plus=top_n_plus, threshold=threshold,path_to_mol2='/' + data_dir + '/mol2/',path_to_labels='/train_metrics/')
    DCC_lig, DCC_site, DCA, volumentric_overlaps, no_prediction_count = compute_metrics_for_all(top_n_plus=top_n_plus, threshold=threshold,path_to_mol2='/' + data_dir + '/mol2/',path_to_labels='/test_metrics/')
    for x in [DCC_lig, DCC_site, DCA, volumentric_overlaps, no_prediction_count]:
        print(x)

    cleaned_DCC_lig =  [entry[0] if len(entry) > 0 else np.nan for entry in DCC_lig]
    cleaned_DCC_site =  [entry[0] if len(entry) > 0 else np.nan for entry in DCC_site]
    cleaned_DCA =  [entry[0] if len(entry) > 0 else np.nan for entry in DCA]
    cleaned_volumentric_overlaps =  [entry[0] if len(entry) > 0 else np.nan for entry in volumentric_overlaps]
    print("Done. {}".format(time.time()- start))
if set_to_use == "val":
    np.savez(prepend + '/vol_overlap_cent_dist_val_set_threshold_{}_{}.npz'.format(model_name.replace("/", "_"), threshold), cleaned_DCC_lig = cleaned_DCC_lig, cleaned_DCC_site = cleaned_DCC_site, cleaned_DCA = cleaned_DCA, cleaned_volumentric_overlaps = cleaned_volumentric_overlaps)
elif set_to_use == "chen":
    np.savez(prepend + '/chen_vol_overlap_cent_dist_val_set_threshold_{}_{}.npz'.format(model_name.replace("/", "_"), threshold), cleaned_DCC_lig = cleaned_DCC_lig, cleaned_DCC_site = cleaned_DCC_site, cleaned_DCA = cleaned_DCA, cleaned_volumentric_overlaps = cleaned_volumentric_overlaps)

    print("-----------------------------------------------------------------------------------", flush=True)
    print("Cutoff (Prediction Threshold):", threshold)
    print("n (for top n):", top_n_plus)
    print("-----------------------------------------------------------------------------------", flush=True)
    print("Number of systems with no predictions:", no_prediction_count, flush=True)
    print("Average Distance From Center:", np.nanmean(cleaned_DCC_lig), flush=True)
    print("Average Distance From Ligand:", np.nanmean(cleaned_DCC_site), flush=True)
    print("Average Discretized Volume Overlap:", np.nanmean(cleaned_DCA), flush=True)
#######################################################################################