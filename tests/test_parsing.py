import os
import shutil
import sys
import numpy as np
from glob import glob

from test_utils import mol_equality_test, preprocessed_npz_equality_test

sys.path.append("..")
from parse_files import process_production_set, process_train_p2rank_style, clean_alternate_positions, load_p2rank_test_ligands, p2rank_df_intersect, process_p2rank_chains

def test_data_processing_production_regression():
    """A regression test to ensure that data processing has not changed between updates. 
    This test targets the "production path".
    """
    
    # It would be better to use tempdir, but there seems to be some path issues associated with that
    test_dir = "./testing_temp_dir" 
    source_data_dir = "./test_data/parsing/production/"
    
    if os.path.exists(test_dir):
        raise ValueError("Testing Directory Already Exists")
    
    try:
        shutil.copytree(f'{source_data_dir}/inputs', f'{test_dir}/unprocessed_inputs')
       
        paths = sorted(glob(f'{test_dir}/unprocessed_inputs/*'))
        assert len(paths) == len(os.listdir(f'{source_data_dir}/inputs'))
        for path in paths:
            process_production_set(path, test_dir)
        
        # Compare the outputs of the test_data to the expected outputs
        expected_mol2                   = sorted(glob(f'{source_data_dir}/standards/mol2/*'))
        expected_raw                    = sorted(glob(f'{source_data_dir}/standards/raw/*'))
        expected_ready_to_parse_mol2    = sorted(glob(f'{source_data_dir}/standards/ready_to_parse_mol2/**/*'))
        
        output_mol2                     = sorted(glob(f'{test_dir}/mol2/*'))
        output_raw                      = sorted(glob(f'{test_dir}/raw/*'))
        output_ready_to_parse_mol2      = sorted(glob(f'{test_dir}/ready_to_parse_mol2/**/*'))
        
        # Check number of files is correct
        assert len(expected_mol2) == len(output_mol2), "Number of mol2 files is incorrect"
        assert len(expected_raw) == len(output_raw), "Number of raw files is incorrect"
        assert len(expected_ready_to_parse_mol2) == len(output_ready_to_parse_mol2), "Number of ready_to_parse_mol2 files is incorrect"
        
        # Check file name equality
        for i in range(len(expected_mol2)):
            assert expected_mol2[i].split('/')[-1] == output_mol2[i].split('/')[-1], \
                f"mol2 file name mismatch: {expected_mol2[i].split('/')[-1]} != {output_mol2[i].split('/')[-1]}"
        for i in range(len(expected_raw)):
            assert expected_raw[i].split('/')[-1] == output_raw[i].split('/')[-1], \
                f"raw file name mismatch: {expected_raw[i].split('/')[-1]} != {output_raw[i].split('/')[-1]}"
        for i in range(len(expected_ready_to_parse_mol2)):
            assert expected_ready_to_parse_mol2[i].split('/')[-1] == output_ready_to_parse_mol2[i].split('/')[-1], \
                f"ready_to_parse_mol2 file name mismatch: {expected_ready_to_parse_mol2[i].split('/')[-1]} != {output_ready_to_parse_mol2[i].split('/')[-1]}"

        # Iterate over the files to ensure they are the same
        for expected, output in zip(expected_mol2, output_mol2):
            mol_equality_test(expected, output)
        for expected, output in zip(expected_ready_to_parse_mol2, output_ready_to_parse_mol2):
            mol_equality_test(expected, output)
        for expected, output in zip(expected_raw, output_raw):
            preprocessed_npz_equality_test(expected, output)

    finally:
        print("test_dir", test_dir)
        if os.path.isdir(test_dir):
            print("removing")
            shutil.rmtree(test_dir)
   
def test_data_processing_scpdb_regression():
    test_dir = "./testing_temp_dir"
    source_data_dir = "./test_data/parsing/scpdb/"

    if os.path.exists(test_dir):
        raise ValueError("Testing Directory Already Exists")
    
    cwd = os.getcwd()

    try:
        shutil.copytree(f'{source_data_dir}/inputs', f'{test_dir}/scPDB_data_dir/unprocessed_mol2')
        # test_dir = os.path.abspath(test_dir)
        source_data_dir = os.path.abspath(source_data_dir)
        
        os.chdir(test_dir)
       
        paths = sorted(os.listdir('./scPDB_data_dir/unprocessed_mol2/'))
        assert len(paths) == len(os.listdir(f'{source_data_dir}/inputs'))
        for path in paths:
            process_train_p2rank_style(path, './')
        
        # Compare the outputs of the test_data to the expected outputs
        expected_mol2                   = sorted(glob(f'{source_data_dir}/standards/mol2/*'))
        expected_raw                    = sorted(glob(f'{source_data_dir}/standards/raw/*'))
        expected_ready_to_parse_mol2    = sorted(glob(f'{source_data_dir}/standards/ready_to_parse_mol2/**/*'))
        
        output_mol2                     = sorted(glob('./mol2/*'))
        output_raw                      = sorted(glob('./raw/*'))
        output_ready_to_parse_mol2      = sorted(glob('./ready_to_parse_mol2/**/*'))
        
        # Check number of files is correct
        assert len(expected_mol2) == len(output_mol2), "Number of mol2 files is incorrect"
        assert len(expected_raw) == len(output_raw), "Number of raw files is incorrect"
        assert len(expected_ready_to_parse_mol2) == len(output_ready_to_parse_mol2), "Number of ready_to_parse_mol2 files is incorrect"
        
        # Check file name equality
        for i in range(len(expected_mol2)):
            assert expected_mol2[i].split('/')[-1] == output_mol2[i].split('/')[-1], \
                f"mol2 file name mismatch: {expected_mol2[i].split('/')[-1]} != {output_mol2[i].split('/')[-1]}"
        for i in range(len(expected_raw)):
            assert expected_raw[i].split('/')[-1] == output_raw[i].split('/')[-1], \
                f"raw file name mismatch: {expected_raw[i].split('/')[-1]} != {output_raw[i].split('/')[-1]}"
        for i in range(len(expected_ready_to_parse_mol2)):
            assert expected_ready_to_parse_mol2[i].split('/')[-1] == output_ready_to_parse_mol2[i].split('/')[-1], \
                f"ready_to_parse_mol2 file name mismatch: {expected_ready_to_parse_mol2[i].split('/')[-1]} != {output_ready_to_parse_mol2[i].split('/')[-1]}"

        # Iterate over the files to ensure they are the same
        for expected, output in zip(expected_mol2, output_mol2):
            mol_equality_test(expected, output)
        for expected, output in zip(expected_ready_to_parse_mol2, output_ready_to_parse_mol2):
            mol_equality_test(expected, output)
        for expected, output in zip(expected_raw, output_raw):
            preprocessed_npz_equality_test(expected, output)

    finally:
        os.chdir(cwd)
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)

def pytest_benchmark_ds_regression(dataset_name):
    
        test_dir = "./testing_temp_dir" 
        source_data_dir = f"./test_data/parsing/{dataset_name}/"
        
        if os.path.exists(test_dir):
            raise ValueError("Testing Directory Already Exists")
        
        try:
            # Copy inputs to unprocessed_pdb
            shutil.copytree(f'{source_data_dir}/inputs/unprocessed_pdb', f'{test_dir}/benchmark_data_dir/{dataset_name}/unprocessed_pdb')
            # shutil.copytree(f'{source_data_dir}/inputs/ligands', f'{test_dir}/ligands')
                    
            print('Cleaning alternate positions...')
            clean_alternate_positions(f'{test_dir}/benchmark_data_dir/{dataset_name}/unprocessed_pdb/', f'{test_dir}/benchmark_data_dir/{dataset_name}/cleaned_pdb/')        

            if '_intersect' in dataset_name or '_chains' in dataset_name:
                path_to_p2rank_ligands = dataset_name.replace('_intersect', "").replace("_chains", "")
            else:
                path_to_p2rank_ligands = dataset_name
            p2rank_df = load_p2rank_test_ligands(f'./test_data/parsing/{path_to_p2rank_ligands}/inputs/ligands.csv')
            if 'intersect' in dataset_name or 'chains' in dataset_name:
                path_to_mlig_ligands = dataset_name.replace('intersect', 'mlig').replace('chains', 'mlig')
                mlig_df =  load_p2rank_test_ligands(f'./test_data/parsing/{path_to_mlig_ligands}/inputs/ligands.csv')
                ligand_df = p2rank_df_intersect(mlig_df, p2rank_df)
            else:
                ligand_df = p2rank_df

            data_dir = f'{test_dir}/benchmark_data_dir/{dataset_name}'  # Output Dir
            nolig_file = f'{data_dir}/no_ligands.txt'
            if os.path.exists(nolig_file): os.remove(nolig_file)
            systems = np.unique(ligand_df['file'])
            
            # For testing, filter the systems by what's present in the test data
            systems = [system_file for system_file in systems if os.path.exists(f'{test_dir}/benchmark_data_dir/{dataset_name}/cleaned_pdb/{system_file}')]
            
            for system in systems:
                process_p2rank_chains(f'{test_dir}/benchmark_data_dir/{dataset_name}/cleaned_pdb/{system}', ligand_df, data_dir=data_dir)

            # Compare the outputs of the test_data to the expected outputs
            expected_mol2                   = sorted(glob(f'{source_data_dir}/standards/mol2/*'))
            expected_raw                    = sorted(glob(f'{source_data_dir}/standards/raw/*'))
            expected_ready_to_parse_mol2    = sorted(glob(f'{source_data_dir}/standards/ready_to_parse_mol2/**/*'))
            expected_split_pdb              = sorted(glob(f'{source_data_dir}/standards/split_pdb/*'))
            
            output_mol2                     = sorted(glob(f'{test_dir}/benchmark_data_dir/{dataset_name}/mol2/*'))
            output_raw                      = sorted(glob(f'{test_dir}/benchmark_data_dir/{dataset_name}/raw/*'))
            output_ready_to_parse_mol2      = sorted(glob(f'{test_dir}/benchmark_data_dir/{dataset_name}/ready_to_parse_mol2/**/*'))
            output_split_pdb                = sorted(glob(f'{source_data_dir}/standards/split_pdb/*'))
            
            # Check number of files is correct
            assert len(expected_mol2) == len(output_mol2), "Number of mol2 files is incorrect"
            assert len(expected_raw) == len(output_raw), "Number of raw files is incorrect"
            assert len(expected_ready_to_parse_mol2) == len(output_ready_to_parse_mol2), "Number of ready_to_parse_mol2 files is incorrect"
            assert len(expected_split_pdb) == len(os.listdir(f'{test_dir}/benchmark_data_dir/{dataset_name}/split_pdb')), "Number of split_pdb files is incorrect"
            
            # Check file name equality
            for i in range(len(expected_mol2)):
                assert expected_mol2[i].split('/')[-1] == output_mol2[i].split('/')[-1], \
                    f"mol2 file name mismatch: {expected_mol2[i].split('/')[-1]} != {output_mol2[i].split('/')[-1]}"
            for i in range(len(expected_raw)):
                assert expected_raw[i].split('/')[-1] == output_raw[i].split('/')[-1], \
                    f"raw file name mismatch: {expected_raw[i].split('/')[-1]} != {output_raw[i].split('/')[-1]}"
            for i in range(len(expected_ready_to_parse_mol2)):
                assert expected_ready_to_parse_mol2[i].split('/')[-1] == output_ready_to_parse_mol2[i].split('/')[-1], \
                    f"ready_to_parse_mol2 file name mismatch: {expected_ready_to_parse_mol2[i].split('/')[-1]} != {output_ready_to_parse_mol2[i].split('/')[-1]}"
            for i in range(len(expected_split_pdb)):
                assert expected_split_pdb[i].split('/')[-1] == output_split_pdb[i].split('/')[-1], \
                    f"split_pdb file name mismatch: {expected_split_pdb[i].split('/')[-1]} != {output_split_pdb[i].split('/')[-1]}"
                    
            # Iterate over the files to ensure they are the same
            for expected, output in zip(expected_mol2, output_mol2):
                mol_equality_test(expected, output)
            for expected, output in zip(expected_ready_to_parse_mol2, output_ready_to_parse_mol2):
                mol_equality_test(expected, output)
            for expected, output in zip(expected_raw, output_raw):
                preprocessed_npz_equality_test(expected, output)
            for expected, output in zip(expected_split_pdb, sorted(glob(f'{test_dir}/benchmark_data_dir/{dataset_name}/split_pdb/*'))):
                mol_equality_test(expected, output)
            

        finally:
            if os.path.isdir(test_dir):
                shutil.rmtree(test_dir)
                    
def test_data_processing_coach420_regression():
    pytest_benchmark_ds_regression('coach420')
    
def test_data_processing_coach420_mlig_regression():
    pytest_benchmark_ds_regression('coach420_mlig')

def test_data_processing_coach420_intersect_regression():
    pytest_benchmark_ds_regression('coach420_intersect')
    
def test_data_processing_holo4k_regression():
    pytest_benchmark_ds_regression('holo4k')
    
def test_data_processing_holo4k_mlig_regression():
    pytest_benchmark_ds_regression('holo4k_mlig')
    
 
def test_data_processing_holo4k_chains_regression():
    pytest_benchmark_ds_regression('holo4k_chains')
    
def test_data_processing_holo4k_intersect_regression():
    pytest_benchmark_ds_regression('holo4k_intersect')