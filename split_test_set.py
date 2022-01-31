import numpy as np
import shutil
import glob
from tqdm import tqdm

dict = np.load("./shaper_similarity_dict.npy", allow_pickle=True).item()

to_move = dict['1iep_1'] + dict['3acj_1']

verbose = False

for name, similarity_val in tqdm(to_move):
    for file in glob.glob(r'./data_atoms_w_atom_feats/{}*'.format(name)):
        if verbose: print(file)
        shutil.move(file, './test_data_dir/raw/')

    for file in glob.glob(r'./mol2_atoms_w_atom_feats/{}*'.format(name)):
        if verbose: print(file)
        shutil.move(file, './test_data_dir/mol2/')
