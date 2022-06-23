import numpy as np
import MDAnalysis as mda
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from scipy.spatial import ConvexHull, HalfspaceIntersection, Delaunay
from scipy.optimize import linprog
import os
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import roc_curve, auc 
import time

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

def center_to_ligand_dist(center, lig_coords):
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

def site_metrics(prot_coords, lig_coords, predicted_probs, true_labels, threshold=.5, quantile=.3, cluster_all=False):
    """Cluster binding sites and calculate distance from true site center and volumetric overlap with true site 

    Parameters
    ----------
    prot_coords : numpy array
        Protein atomic coordinates.
        
    lig_coords : numpy array
        Ligand atomic coordinates.

    predicted_probs : numpy_array
        Class probabilities for not site in column 0 and site in column 1.

    true_labels : numpy_array
        Class membership, 0 for not site, 1 for site.

    threshold : float
        Probability threshold to predict binding site atoms.

    quantile : float
        Quantile used in bandwidth selection for mean shift clustering.

    cluster_all : bool
        Whether to assign points outside kernels to the nearest cluster or leave them unlabeled.

    Returns
    -------
    center_distances: list
        List of distances from predicted site center to true center. 
        Listed in descending order by predicted site probability.
        
    ligand_distances: list
        List of closest distances from predicted site center to any ligand heavy atom. 
        Listed in descending order by predicted site probability.

    volumentric_overlaps: list
        Jaccard similarity between predicted site convex hull and true site convex hull. 
        Listed in descending order by predicted site probability.

    """
    bind_coords, sorted_ids, _ = cluster_atoms(prot_coords, predicted_probs, threshold=threshold, quantile=quantile, cluster_all=cluster_all)


    true_points = prot_coords[true_labels==1]
    true_hull = ConvexHull(true_points)
    true_center = hull_center(true_hull)

    center_distances = []
    ligand_distances = []
    volumetric_overlaps = []
    
    for c_id in np.unique(sorted_ids)[::-1]:
        if c_id != None:
            if c_id >= 0:
                predicted_points = bind_coords[sorted_ids == c_id]
                
                if len(predicted_points) < 4:  # You need four points to define a convex hull so we'll say the overlap is 0
                    predicted_center = get_centroid(predicted_points)
                    volumetric_overlaps.append(0) 
                else:
                    predicted_hull = ConvexHull(predicted_points)
                    predicted_center = hull_center(predicted_hull)
                    volumetric_overlaps.append(volumetric_overlap(predicted_hull, true_hull))
                 
                center_dist = np.sqrt(np.sum((predicted_center - true_center)**2))
                center_distances.append(center_dist)
                
                ligand_distances.append(center_to_ligand_dist(predicted_center, lig_coords))

    return center_distances, ligand_distances, volumetric_overlaps

def compute_metrics_for_all(threshold = 0.5, path_to_mol2='/test_data_dir/mol2/', path_to_labels = '/test_metrics/'):
    cent_dist_list = []
    lig_dist_list = []
    vol_overlap_list = []
    no_prediction_count = 0

    for file in os.listdir(prepend + path_to_labels + 'test_probs/' + model_name + '/'): 
        # print(file)
        # print(type(file))
        assembly_name = file.split('.')[-2]
        try:
            trimmed_protein = mda.Universe(prepend + path_to_mol2 + assembly_name + '.mol2')
            labels = np.load(prepend + path_to_labels + 'test_labels/' + assembly_name + '.npy')
            probs = np.load(prepend + path_to_labels + 'test_probs/' + model_name + '/' + assembly_name + '.npy')
            # probs = np.load(prepend + '/test_metrics/test_probs/' + model_name + '_' + assembly_name + '.npy')
            ligand = mda.Universe(prepend + "/data_dir/unprocessed_scPDB_mol2/" + assembly_name + '/ligand.mol2')
            ligand = ligand.select_atoms("not type H")
            


            cent_dist, lig_dist, vol_overlap = site_metrics(trimmed_protein.atoms.positions, ligand.atoms.positions, probs, labels, threshold=threshold)
            if cent_dist == [] or lig_dist == [] or vol_overlap_list == []: 
                no_prediction_count += 1
            cent_dist_list.append(cent_dist)
            lig_dist_list.append(lig_dist)
            vol_overlap_list.append(vol_overlap)


        except Exception as e:
            print(assembly_name, flush=True)
            raise e
    return cent_dist_list, lig_dist_list, vol_overlap_list, no_prediction_count

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
model_name = "trained_model_1650260810.482072/epoch_23" # Old params, new labeling, ob

data_dir = 'scPDB_data_dir'

prepend = str(os.getcwd())
threshold_lst = [0.4, 0.45, 0.5]
compute_optimal = True

#######################################################################################
if compute_optimal:
    all_probs  = np.load(prepend + "/test_metrics/all_probs/" + model_name + ".npz")['arr_0']
    all_labels = np.load(prepend + "/test_metrics/all_labels/" + model_name + ".npz")['arr_0']
    # all_probs  = np.load(prepend + "/train_metrics/all_probs/" + model_name + ".npz")['arr_0']
    # all_labels = np.load(prepend + "/train_metrics/all_labels/" + model_name + ".npz")['arr_0']
    print("Calculating optimal cutoffs.")
    start = time.time()
    # all_probs  =  all_probs.numpy()
    # all_labels = all_labels.numpy()
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

    if not os.path.isdir(prepend + '/test_metrics/roc_curves/' + model_name):
        os.makedirs(prepend + '/test_metrics/roc_curves/' + model_name)
    # if not os.path.isdir(prepend + '/train_metrics/roc_curves/' + model_name):
    #      os.makedirs(prepend + '/train_metrics/roc_curves/' + model_name)

    # Find optimal threshold
    gmeans = np.sqrt(tpr[1] * (1-fpr[1]))
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[1][ix]

    print('Best Threshold=%f, G-Mean=%.3f' % (optimal_threshold, gmeans[ix]))
    print("Micro Averaged AUC:", roc_auc["micro"])
    print("Macro Averaged AUC:", roc_auc["macro"])
    print("Negative Class AUC:", roc_auc[0])
    print("Positive Class AUC:", roc_auc[1])

    np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/roc_auc", roc_auc)
    np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/tpr", tpr)
    np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/fpr", fpr)
    np.savez(prepend + "/test_metrics/roc_curves/" + model_name + "/thresholds", thresholds)
    # np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/roc_auc", roc_auc)
    # np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/tpr", tpr)
    # np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/fpr", fpr)
    # np.savez(prepend + "/train_metrics/roc_curves/" + model_name + "/thresholds", thresholds)
    print("Done. {}".format(time.time()- start))
    threshold_lst.insert(0, optimal_threshold)
    
#######################################################################################

#######################################################################################
for threshold in threshold_lst:
    print("Calculating overlap and center distance metrics for "+str(threshold)+" threshold.", flush=True)
    start = time.time()
    cent_dist_list, lig_dist_list, vol_overlap_list, no_prediction_count = compute_metrics_for_all(threshold=threshold,path_to_mol2='/' + data_dir + '/mol2/',path_to_labels='/train_metrics/')

    cleaned_vol_overlap_list =  [entry[0] if len(entry) > 0 else np.nan for entry in vol_overlap_list]
    cleaned_cent_dist_list =  [entry[0] if len(entry) > 0 else np.nan for entry in cent_dist_list]
    cleaned_lig_dist_list =  [entry[0] if len(entry) > 0 else np.nan for entry in lig_dist_list]
    print("Done. {}".format(time.time()- start))

    np.savez(prepend + '/vol_overlap_cent_dist_val_set_threshold_{}_{}.npz'.format(model_name.replace("/", "_"), threshold), overlaps=vol_overlap_list, dist_lst=cent_dist_list, lig_list=lig_dist_list)

    print("-----------------------------------------------------------------------------------", flush=True)
    print("Cutoff (Prediction Threshold):", threshold)
    print("-----------------------------------------------------------------------------------", flush=True)
    print("Number of systems with no predictions:", no_prediction_count, flush=True)
    print("Average Distance From Center (Top 1):", np.nanmean(cleaned_cent_dist_list), flush=True)
    print("Average Distance From Ligand (Top 1):", np.nanmean(cleaned_lig_dist_list), flush=True)
    print("Average Discretized Volume Overlap (Top 1):", np.nanmean(cleaned_vol_overlap_list), flush=True)
#######################################################################################
