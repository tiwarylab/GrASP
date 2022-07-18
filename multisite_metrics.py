import numpy as np
import MDAnalysis as mda
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
from rdkit import Chem
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

from joblib import Parallel, delayed

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
            # print(bind_coords, flush=True)
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
    DCC_lig: numpy array
        List of distances from predicted site center to ligand center of mass. 
    
    DCC_site: numpy array
        List of distances from predicted site center to true center. 
        
    DCA: numpy array
        List of closest distances from predicted site center to any ligand heavy atom. 

    volumetric_overlaps: numpy array
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
        DCC_lig_matrix = np.zeros([len(true_center_list), len(predicted_center_list)])
        DCC_site_matrix = np.zeros([len(true_center_list), len(predicted_center_list)])
        DCA_matrix = np.zeros([len(true_center_list), len(predicted_center_list)])
        for index, x in np.ndenumerate(DCC_lig_matrix):
            true_ind, pred_ind = index
            ligand_center = ligand_center_list[true_ind]
            lig_coords = lig_coord_list[true_ind]
            site_center = true_center_list[true_ind]
            predicted_center = predicted_center_list[pred_ind]

            DCC_lig_matrix[index] = np.sqrt(np.sum((predicted_center - ligand_center)**2))
            DCC_site_matrix[index] = np.sqrt(np.sum((predicted_center - site_center)**2))
            DCA_matrix[index] = DCA_dist(predicted_center, lig_coords)
                  
        # print(DCC_lig_matrix)
        closest_predictions = np.argmin(DCC_site_matrix, axis=1)
        site_pairs = np.column_stack([np.arange(len(closest_predictions)), closest_predictions])

        DCC_lig = np.min(DCC_lig_matrix, axis=1)
        DCC_site = np.min(DCC_site_matrix, axis=1)
        DCA = np.min(DCA_matrix, axis=1)
                                    
        for pair in site_pairs:
            true_ind, pred_ind = pair
            true_hull = true_hull_list[true_ind]
            predicted_hull = predicted_hull_list[pred_ind]                    
            if predicted_hull != None: 
                volumetric_overlaps.append(volumetric_overlap(predicted_hull, true_hull))
            elif true_hull == None:
                raise ValueError ("There were < 3 atoms in your true site label. The indicates that the associated ligand is not burried.")
            else:
                volumetric_overlaps.append(np.nan)   
        volumetric_overlaps = np.array(volumetric_overlaps)

        return DCC_lig, DCC_site, DCA, volumetric_overlaps

    else:
        nan_arr =  np.empty(len(true_center_list))
        nan_arr[:] = np.nan

        return nan_arr, nan_arr, nan_arr, nan_arr


def compute_metrics_for_all(path_to_mol2, path_to_labels, top_n_plus=0, threshold = 0.5, ):
    DCC_lig_list = []
    DCC_site_list = []
    DCA_list = []
    volumetric_overlaps_list = []

    def helper(file):
        no_prediction_count = 0
        assembly_name = file.split('.')[-2]
        try:
            trimmed_protein = mda.Universe(path_to_mol2 + assembly_name + '.mol2')
            labels = np.load(prepend + metric_dir + '/labels/' + assembly_name + '.npy')
            probs = np.load(prepend + metric_dir + '/probs/' + model_name + '/' + assembly_name + '.npy')
            # print(probs.shape)
            ############### THIS IS TEMPORARY AF REMOVE BEFORE PUBLICAITON ##############
            probs = np.array([[1,0] if x ==0 else [0,1] for x in labels])
            # print(probs.shape)

            lig_coord_list = []
            ligand_mass_list = []
            
            site_coords_list = []
            print(data_dir + '/ready_to_parse_mol2/' + assembly_name + '/*')
            for file_path in sorted(glob(data_dir + '/ready_to_parse_mol2/' + assembly_name + '/*')):
                print(file_path)
                if 'ligand' in file_path.split('/')[-1] and not 'site' in file_path.split('/')[-1]:
                    ligand = mda.Universe(file_path).select_atoms("not type H")
                    rdk_ligand = Chem.MolFromMol2File(file_path, removeHs = False, sanitize=False, cleanupSubstructures=False)
                    lig_coord_list.append(list(ligand.atoms.positions))
                    ligand_mass_list.append([rdk_ligand.GetAtomWithIdx(int(i)).GetMass() for i in ligand.atoms.indices])
                elif 'site_for_ligand' in file_path.split('/')[-1]:
                    site = mda.Universe(file_path).select_atoms("not type H")
                    site_coords_list.append(site.atoms.positions)
            DCC_lig, DCC_site, DCA, volumetric_overlaps = multi_site_metrics(trimmed_protein.atoms.positions, lig_coord_list, ligand_mass_list, probs, site_coords_list, top_n_plus=top_n_plus, threshold=threshold, quantile=.3, cluster_all=False)
            if np.all(np.isnan(DCC_lig)) or np.all(np.isnan(DCC_site)) or np.all(np.isnan(DCA)) or np.all(np.isnan(volumetric_overlaps)): 
                no_prediction_count += 1
            return DCC_lig, DCC_site, DCA, volumetric_overlaps, no_prediction_count

        except Exception as e:
            print(assembly_name, flush=True)
            raise e

    # for file in tqdm(os.listdir(prepend + path_to_labels + 'test_probs/' + model_name + '/'),  position=0, leave=True): 
    #     # print(file)
    #     # print(type(file))
    #     assembly_name = file.split('.')[-2]
    #     try:
    #         trimmed_protein = mda.Universe(prepend + path_to_mol2 + assembly_name + '.mol2')
    #         labels = np.load(prepend + path_to_labels + 'test_labels/' + assembly_name + '.npy')
    #         probs = np.load(prepend + path_to_labels + 'test_probs/' + model_name + '/' + assembly_name + '.npy')

    #         lig_coord_list = []
    #         ligand_mass_list = []
            
    #         site_coords_list = []
            
    #         for file_path in sorted(glob(prepend + "/scPDB_data_dir/ready_to_parse_mol2/" + assembly_name + '/*')):
    #         # for file_path in sorted(glob(prepend + "/benchmark_data_dir/unprocessed_mol2/" + assembly_name + '/*')):
    #             if 'ligand' in file_path.split('/')[-1] and not 'site' in file_path.split('/')[-1]:
    #                 ligand = mda.Universe(file_path).select_atoms("not type H")
    #                 lig_coord_list.append(list(ligand.atoms.positions))
    #                 ligand_mass_list.append(list(ligand.atoms.masses))
    #             elif 'site_for_ligand' in file_path.split('/')[-1]:
    #                 site = mda.Universe(file_path).select_atoms("not type H")
    #                 site_coords_list.append(site.atoms.positions)
    #         DCC_lig, DCC_site, DCA, volumetric_overlaps = multi_site_metrics(trimmed_protein.atoms.positions, lig_coord_list, ligand_mass_list, probs, site_coords_list, top_n_plus=top_n_plus, threshold=threshold, quantile=.3, cluster_all=False)
    #         if np.all(np.isnan(DCC_lig)) or np.all(np.isnan(DCC_site)) or np.all(np.isnan(DCA)) or np.all(np.isnan(volumetric_overlaps)): 
    #             no_prediction_count += 1
    #         DCC_lig_list.append(DCC_lig)
    #         DCC_site_list.append(DCC_site)
    #         DCA_list.append(DCA)
    #         volumetric_overlaps_list.append(volumetric_overlaps)
            

    #     except Exception as e:
    #         print(assembly_name, flush=True)
    #         raise e
    
    # Normally 15 jobs
    r = Parallel(n_jobs=1)(delayed(helper)(file) for file in tqdm(os.listdir(path_to_labels)[:],  position=0, leave=True))
    DCC_lig_list, DCC_site_list, DCA_list, volumetric_overlaps_list, no_prediction_count = zip(*r)
        
    return DCC_lig_list, DCC_site_list, DCA_list, volumetric_overlaps_list, no_prediction_count

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
# model_name = "trained_model_1650260810.482072/epoch_46" # Old params, new labeling, ob
model_name = "trained_model_1652221046.78391/epoch_49"


prepend = str(os.getcwd()) #+ "/chen_benchmark_site_metrics/"
threshold_lst = [0.4, 0.45, 0.5]
compute_optimal = True
top_n_plus=2

set_to_use = "chen" #"chen"|"val"

if set_to_use == 'val':
    print("Performing Metrics on the Validation Set")
    data_dir = prepend + '/scPDB_data_dir'
    metric_dir = '/test_metrics/validation'
elif set_to_use == 'chen':
    print("Performing Metrics on the Chen Set")    
    data_dir = prepend + '/benchmark_data_dir/chen'
    metric_dir = '/test_metrics/chen'
elif set_to_use ==  'coach420':
    print("Performing Metrics on the coach420 Set")    
    data_dir = prepend + '/benchmark_data_dir/coach420'
    metric_dir = '/test_metrics/coach420'
elif set_to_use == 'holo4k':
    print("Performing Metrics on the holo4k Set")    
    data_dir = prepend + '/benchmark_data_dir/holo4k'
    metric_dir = '/test_metrics/holo4k'
elif set_to_use == 'sc6k':
    print("Performing Metrics on the sc6k Set")    
    data_dir = prepend + '/benchmark_data_dir/sc6k'
    metric_dir = '/test_metrics/sc6k'
else:
    raise ValueError("Expected one of {'val','chen','coach420','holo4k','sc6k'} as set_to_use but got:", str(set_to_use))

#######################################################################################
if compute_optimal:
    all_prob_path = prepend + metric_dir + '/all_probs/' + model_name + '/'
    all_label_path = prepend + metric_dir + '/all_labels/'
    all_probs  = np.load(all_prob_path + "all_probs.npz")['arr_0']
    all_labels = np.load(all_label_path + "all_labels.npz")['arr_0']
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

    roc_path = prepend + metric_dir + '/roc_curves/' + model_name

    if not os.path.isdir(roc_path):
        os.makedirs(roc_path)

    # Find optimal threshold
    gmeans = np.sqrt(tpr[1] * (1-fpr[1]))
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[1][ix]
    threshold_lst.insert(0, optimal_threshold)

    print('Best Threshold=%f, G-Mean=%.3f' % (optimal_threshold, gmeans[ix]))
    print("Micro Averaged AUC:", roc_auc["micro"])
    print("Macro Averaged AUC:", roc_auc["macro"])
    print("Negative Class AUC:", roc_auc[0])
    print("Positive Class AUC:", roc_auc[1])

    np.savez(roc_path + "/roc_auc", roc_auc)
    np.savez(roc_path + "/tpr", tpr)
    np.savez(roc_path + "/fpr", fpr)
    np.savez(roc_path + "/thresholds", thresholds)
    print("Done. {}".format(time.time()- start))
    
#######################################################################################

def extract_multi(metric_array):
    success_rate = np.mean(np.concatenate(metric_array) < 4)
    mean = np.nanmean(np.concatenate(metric_array))
        
    return success_rate, mean

#######################################################################################
for threshold in threshold_lst:
    print("Calculating overlap and center distance metrics for "+str(threshold)+" threshold.", flush=True)
    start = time.time()
    path_to_mol2= data_dir + '/mol2/'
    path_to_labels=prepend + metric_dir + '/labels/'
    DCC_lig, DCC_site, DCA, volumetric_overlaps, no_prediction_count = compute_metrics_for_all(path_to_mol2,path_to_labels,top_n_plus=top_n_plus, threshold=threshold,)
    # for x in [DCC_lig, DCC_site, DCA, volumetric_overlaps, no_prediction_count]:
    #     print(x)

    print("Done. {}".format(time.time()- start))
    
    overlap_path = prepend + metric_dir + '/overlaps/' + model_name
    if not os.path.isdir(overlap_path):
        os.makedirs(overlap_path)
    
    np.savez(overlap_path + 'overlaps_for_threshold_{}.npz'.format(threshold), DCC_lig = DCC_lig, DCC_site = DCC_site, DCA = DCA, volumetric_overlaps = volumetric_overlaps)
    VO = volumetric_overlaps

    print("-----------------------------------------------------------------------------------", flush=True)
    print("Cutoff (Prediction Threshold):", threshold)
    print("n (for top n):", top_n_plus)
    print("-----------------------------------------------------------------------------------", flush=True)
    print("Number of systems with no predictions:", np.sum(no_prediction_count), flush=True)
    # print("Average DCC_lig:", np.nanmean(DCC_lig), flush=True)
    # print("Average DCC_site:", np.nanmean(DCC_site), flush=True)
    # print("Average DCA:", np.nanmean(DCA), flush=True)
    # print("Average VO:", np.nanmean(volumetric_overlaps), flush=True)
    DCC_lig_succ, DCC_lig_mean = extract_multi(DCC_lig)
    DCC_site_succ, DCC_site_mean = extract_multi(DCC_site)
    DCA_succ, DCA_mean = extract_multi(DCA)
    print(f"Number of systems with no predictions: {np.sum([np.all(np.isnan(i)) for i in VO])}", flush=True)

    print(f"Average DCC_lig: {DCC_lig_mean}", flush=True)
    print(f"DCC_lig Success: {DCC_lig_succ}", flush=True)

    print(f"Average DCC_site: {DCC_site_mean}", flush=True)
    print(f"DCC_site Success: {DCC_site_succ}", flush=True)

    print(f"Average DCA: {DCA_mean}", flush=True)
    print(f"DCA Success: {DCA_succ}", flush=True)

    print(f"Average VO: {np.nanmean(np.concatenate(VO))}", flush=True)
    print(f"Average VO (DCC_site Success): {np.nanmean(np.concatenate(VO)[np.concatenate(DCC_site) < 4])}", flush=True)
    #######################################################################################
