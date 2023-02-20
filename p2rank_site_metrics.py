import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
from rdkit import Chem
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, AgglomerativeClustering
import networkx as nx 
from networkx.algorithms.community import louvain_communities
from scipy.spatial import ConvexHull, HalfspaceIntersection, Delaunay
from scipy.optimize import linprog
import os
from tqdm import tqdm
from glob import glob

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import roc_curve, auc 
import time

import sys
import argparse
from joblib import Parallel, delayed


def DCA_dist(center, lig_coords):
    distances = np.sqrt(np.sum((center - lig_coords)**2, axis=1))
    shortest = np.min(distances)
    
    return shortest


def get_p2rank_centers(prediction_dir, system):
    pred_df = pd.read_csv(f'{prediction_dir}/{system}.pdb_predictions.csv', delimiter='\s*,\s*', engine='python')
    predicted_centers = pred_df[['center_x', 'center_y', 'center_z']].to_numpy()
    
    return predicted_centers


def multisite_metrics(lig_coord_list, predicted_center_list, top_n_plus=0):
    n_sites = len(lig_coord_list)
    n_predicted = len(predicted_center_list)
    predicted_center_list = predicted_center_list[:n_sites+top_n_plus]
    

    if len(predicted_center_list) > 0:          
        DCA_matrix = np.zeros([len(lig_coord_list), len(predicted_center_list)])

        for index, x in np.ndenumerate(DCA_matrix):
            true_ind, pred_ind = index

            predicted_center = predicted_center_list[pred_ind]
            lig_coords = lig_coord_list[true_ind]

            DCA_matrix[index] = DCA_dist(predicted_center, lig_coords)

        DCA = np.min(DCA_matrix, axis=1)

        return  DCA, n_predicted

    else:
        nan_arr =  np.empty(len(lig_coord_list))
        nan_arr[:] = np.nan

        return  nan_arr, n_predicted


def compute_metrics_for_all(path_to_mol2, path_to_predictions, top_n_plus=0):
    DCA_list = []

    def helper(file):
        no_prediction_count = 0
        assembly_name = file.split('.')[-2]
        predicted_center_list = get_p2rank_centers(path_to_predictions, assembly_name)

        try:
            lig_coord_list = []
            
            for file_path in sorted(glob(data_dir + '/ready_to_parse_mol2/' + assembly_name + '/*')):
                # print(file_path)
                if 'ligand' in file_path.split('/')[-1] and not 'site' in file_path.split('/')[-1]:
                    ligand = mda.Universe(file_path)
                    lig_coord_list.append(list(ligand.atoms.positions))
            DCA, n_predicted = multisite_metrics(lig_coord_list, predicted_center_list, top_n_plus=top_n_plus)

            if np.all(np.isnan(DCA)): 
                no_prediction_count += 1
            return DCA, n_predicted, no_prediction_count

        except Exception as e:
            print("ERROR")
            print(assembly_name, flush=True)
            raise e

    r = Parallel(n_jobs=n_jobs)(delayed(helper)(file) for file in tqdm(os.listdir(path_to_mol2)[:],  position=0, leave=True))
    DCA_list, n_predicted, no_prediction_count = zip(*r)
    names = [file for file in os.listdir(path_to_mol2)]
    return DCA_list, n_predicted, no_prediction_count, names

def extract_multi(metric_array):
    success_rate = np.mean(np.concatenate(metric_array) < 4)
    mean = np.nanmean(np.concatenate(metric_array))
        
    return success_rate, mean
#######################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster GNN predictions into binding sites.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("test_set", choices=["val", "coach420", "coach420_mlig", "coach420_intersect",
     "holo4k", "holo4k_mlig", "holo4k_intersect"], help="Test set.")
    parser.add_argument("-tn", "--top_n_plus", type=int, nargs="+", default=[0,2,100], help="Number of additional sites to consider.")
    parser.add_argument("-n", "--n_tasks", type=int, default=15, help="Number of cpu workers.")

    args = parser.parse_args()
    non_path_args = [sys.argv[1]]
    argstring='_'.join(non_path_args).replace('-','')


    prepend = str(os.getcwd()) #+ "/chen_benchmark_site_metrics/"
    top_n_list=args.top_n_plus
    n_jobs = args.n_tasks

    set_to_use = args.test_set
    if set_to_use == 'val':
        print("Calculating metrics on the validation set")
        data_dir = prepend + '/scPDB_data_dir'
        metric_dir = '/test_metrics/validation'
    else:
        print(f"Calculating metrics on the {set_to_use} set")    
        data_dir = f'{prepend}/benchmark_data_dir/{set_to_use}'
        metric_dir = f'/test_metrics/{set_to_use}'


    #######################################################################################
    outdir = f"{prepend}{metric_dir}/clustering/p2rank"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = f"{outdir}/p2rank_{argstring}.dat"
    if os.path.exists(outfile):
        os.remove(outfile)
    out = open(outfile, 'a')

    path_to_predictions = f"{prepend}{metric_dir}/p2rank/predictions"
    if "intersect" in set_to_use:
        path_to_predictions = path_to_predictions.replace("intersect", "mlig")

    for top_n_plus in top_n_list:
        print(f"Calculating n+{top_n_plus} metrics for p2rank.", flush=True)
        out.write(f"Calculating n+{top_n_plus} metrics for p2rank.\n")
        start = time.time()
        path_to_mol2= data_dir + '/mol2/'
        DCA, n_predicted, no_prediction_count, names = compute_metrics_for_all(path_to_mol2, path_to_predictions, top_n_plus=top_n_plus)

        print("Done. {}".format(time.time()- start))
        out.write("Done. {}\n".format(time.time()- start))
        
        overlap_path = f"{prepend}{metric_dir}/overlaps/p2rank"
        if not os.path.isdir(overlap_path):
            os.makedirs(overlap_path)
        

        np.savez(f"{overlap_path}/p2rank_{argstring}_n+{top_n_plus}.npz", DCA=np.array(DCA, dtype=object),
            n_predicted=n_predicted, names=names)

        n_predicted = np.array(n_predicted)

        print("-----------------------------------------------------------------------------------", flush=True)
        print(f"top n + {top_n_plus} prediction")
        print("-----------------------------------------------------------------------------------", flush=True)
        print(f"Number of systems with no predictions: {np.sum(no_prediction_count)}", flush=True)

        out.write("-----------------------------------------------------------------------------------\n")
        out.write(f"top n + {top_n_plus} prediction\n")
        out.write("-----------------------------------------------------------------------------------\n")
        out.write(f"Number of systems with no predictions: {np.sum(no_prediction_count)}\n")

        DCA_succ, DCA_mean = extract_multi(DCA)

        print(f"Average DCA: {DCA_mean}", flush=True)
        print(f"DCA Success: {DCA_succ}", flush=True)

        print(f"Average n_predicted: {np.nanmean(n_predicted)}", flush=True)

        out.write(f"Average DCA: {DCA_mean}\n")
        out.write(f"DCA Success: {DCA_succ}\n")

        out.write(f"Average n_predicted: {np.nanmean(n_predicted)}\n")
        #######################################################################################
    out.close()
