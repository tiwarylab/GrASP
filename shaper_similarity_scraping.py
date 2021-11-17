import os
import pandas as pd
from collections import defaultdict
from time import sleep
from tqdm import tqdm
import numpy as np

processed_dir = "data_atoms_w_atom_feats"
shaper_similarity_dict = defaultdict(list)
complex_list = sorted(os.listdir(processed_dir))

for complex in tqdm(complex_list):
    complex_name = complex.split('.')[0]
    url = "http://bioinfo-pharma.u-strasbg.fr/scPDB/EXPORTSHAP=" + complex_name
    # sleep(0.1)
    df = pd.read_table(url)
    for idx, row in df.iterrows():
        PDB_ID = row['PDB_ID']
        similarity = row['Binding Site Similarity']
        shaper_similarity_dict[complex_name].append((PDB_ID, similarity))


# shaper_similarity_dict
np.save('shaper_similarity_dict.npy', shaper_similarity_dict)