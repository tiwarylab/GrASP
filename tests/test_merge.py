import os
import sys
from glob import glob
import shutil
import tempfile
import pandas as pd
import numpy as np
from test_utils import mol_equality_test
sys.path.append("..")
from merge import generate_structures

def test_generate_structures_regression():
    test_files = './test_data/merge/input'
    standard_dir = './test_data/merge/standards'
    
    with tempfile.TemporaryDirectory(dir='./') as test_dir:
        print(test_dir, flush=True)
        input_directory = os.path.join(test_dir, 'input')
        save_directory = os.path.join(test_dir, 'output')
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        # Copy files from the test_files to the temporary directory
        shutil.copytree(test_files, input_directory)
        print(os.listdir(input_directory), flush=True)
        print(save_directory, flush=True)
        pdbID_i_list = os.listdir(input_directory)
        pdbID_list = np.unique(sorted([x[:4] for x in pdbID_i_list]))  
        for i, id, in enumerate(pdbID_list):
            print(i, id, str(input_directory), str(save_directory), flush=True)
            generate_structures(i, id, str(input_directory), str(save_directory))
            
        # Compare to the standard files
        output_csvs = sorted(glob(save_directory + '/*/about.csv'), key=lambda x: x.split('/')[-2])
        output_mol2s = sorted(glob(save_directory + '/*/*.mol2'), key=lambda x: x.split('/')[-2])
        
        expected_csvs = sorted(glob(standard_dir + '/*/about.csv'), key=lambda x: x.split('/')[-2])
        expected_mol2s = sorted(glob(standard_dir + '/*/*.mol2'), key=lambda x: x.split('/')[-2])
         
        assert len(output_csvs) == len(expected_csvs), "Incorrect number of csv files"
        assert len(output_mol2s) == len(expected_mol2s), "Incorrect number of mol2 files"
        
        for output, expected in zip(output_csvs, expected_csvs):
            assert pd.DataFrame.equals(pd.read_csv(output), pd.read_csv(expected)), f"CSV files do not match {output} {expected}"
            
        for output, expected in zip(output_mol2s, expected_mol2s):
            mol_equality_test(output, expected), f"Mol2 files do not match {output} {expected}"