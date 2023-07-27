import MDAnalysis as mda
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
import numpy as np
import argparse
import os
from glob import glob
from tqdm import tqdm

def color_probs(protein, probs, out_file):
    protein.add_TopologyAttr('tempfactors')
    protein.atoms.tempfactors = probs
    protein.atoms.write(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color PDB by ligandability scores.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_name", nargs="?", default="train_full/trained_model_s_train_full_ag_multi_1680643832.8660116/cv_0/epoch_49", help="Model file path.")
    args = parser.parse_args()

    prepend = str(os.getcwd())
    model_name = args.model_name
    data_dir = f'{prepend}/benchmark_data_dir/production'
    path_to_mol2= f'{data_dir}/mol2'
    metric_dir = f'{prepend}/test_metrics/production'
    color_dir = f'{metric_dir}/colors/{model_name}'

    if not os.path.isdir(color_dir):
        os.makedirs(color_dir)
    

    paths = glob(f'{metric_dir}/probs/{model_name}/*.npy')
    for file in tqdm(paths):
        print(file)
        assembly_name = file.split('/')[-1].split('.')[-2]
        protein = mda.Universe(f'{path_to_mol2}/{assembly_name}.mol2')
        probs = np.load(f'{metric_dir}/probs/{model_name}/{assembly_name}.npy')
        atom_indices = np.load(f'{metric_dir}/indices/{model_name}/{assembly_name}.npy')
        surf_mask = np.load(f'{metric_dir}/SASAs/{assembly_name}.npy')

        all_probs = np.zeros(len(protein.atoms))
        all_probs[atom_indices[surf_mask]] = probs[:,1]
        color_probs(protein, all_probs, f'{color_dir}/{assembly_name}_probs.pdb')
