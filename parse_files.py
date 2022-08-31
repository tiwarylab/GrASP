from dataclasses import field
from turtle import end_fill
from featurize_protein import process_system
from merge import write_fragment
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
import os
import numpy as np
import pandas as pd
import shutil
import openbabel
from tqdm import tqdm
from glob import glob
import re
import argparse

allowed_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'C', 'G', 'A', 'U', 'I', 'DC', 'DG', 'DA', 'DU', 'DT', 'DI']
selection_str = "".join(["resname " + x + " or " for x in list(allowed_residues)[:-1]]) + "resname " + str(allowed_residues[-1])
exclusion_list = ['HOH', 'DOD', 'WAT', 'NAG', 'MAN', 'UNK', 'GLC', 'ABA', 'MPD', 'GOL', 'SO4', 'PO4']


def add_chains_from_frags(univ):
    univ.add_TopologyAttr('chainID')
    for i, frag in enumerate(univ.atoms.fragments):
        univ.add_Segment(segid=str(i))
        for atom in frag.atoms:
            atom.chainID = str(i)


def label_sites_given_ligands(path_to_mol2, extension='mol2'):
    protein = mda.Universe(os.path.join(path_to_mol2, f'protein.{extension}'))
    add_chains_from_frags(protein)
    protein_no_h = protein.select_atoms("not type H")
    
    all_sites = mda.AtomGroup([], protein) # empty AtomGroup
    for file_path in sorted(glob(path_to_mol2+ '/*')):
        if 'protein' in file_path:
            # This is the main structure, we already have it
            pass
        elif 'ligand' in file_path and not 'site' in file_path:
            # This is a ligand file
            ligand = mda.Universe(file_path)
            site_resid_list = []
            site_chain_list = []
            for atom in ligand.atoms:
                x,y,z = atom.position
                atoms_to_label = protein_no_h.select_atoms("point {} {} {} 6.5".format(x, y, z))
                resids_to_label = list(atoms_to_label.resids)
                chains_to_label = list(atoms_to_label.chainIDs)
                site_resid_list += resids_to_label
                site_chain_list += chains_to_label
            
            resid_chain_pairs = set(zip(site_resid_list, site_chain_list))
            site_selection_str = " or ".join([f'(resid {x} and chainID {y})' for x, y in resid_chain_pairs])

            this_ligands_site = protein.select_atoms(site_selection_str)
            all_sites += this_ligands_site
            ligand_idx = int(re.findall("\d+",file_path.split('/')[-1])[0])
            this_ligands_site.atoms.write(os.path.join(path_to_mol2, f"site_for_ligand_{ligand_idx}.{extension}"))

        else:
            # This is an unexpected file
            pass
    
    all_sites.write(os.path.join(path_to_mol2, f"site.{extension}"))


def pdb2mol2(pdb_file, structure_name, out_directory, addH=True, out_name='protein', cleanup=True):
    # print("Starting")
    output_path = out_directory + structure_name
    if not os.path.isdir(output_path): os.makedirs(output_path)
    
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "mol2")

    output_mol2_path  = output_path + '/' + out_name +'.mol2'

    mol = openbabel.OBMol()

    obConversion.ReadFile(mol, pdb_file)
    mol.DeleteHydrogens()
    if addH: 
        mol.CorrectForPH()
        mol.AddHydrogens()

    obConversion.WriteFile(mol, output_mol2_path)
    
    if cleanup:
        # Use MDA to remove clean file
        univ = mda.Universe(output_mol2_path)
        res_names = univ.residues.resnames
        new_names = [ "".join(re.findall(".*[a-zA-Z]+", name)).upper() for name in res_names]
        univ.residues.resnames = new_names
        univ = univ.select_atoms(selection_str)
        mda.coordinates.MOL2.MOL2Writer(output_mol2_path).write(univ)


def protein2mol2(pdb_file, structure_name, out_directory, min_size=256, addH=True, out_name='protein', cleanup=True):
    # print("Starting")
    output_path = out_directory + structure_name
    if not os.path.isdir(output_path): os.makedirs(output_path)
    
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "mol2")

    output_mol2_path  = output_path + '/' + out_name +'.mol2'

    mol = openbabel.OBMol()

    obConversion.ReadFile(mol, pdb_file)
    mol.StripSalts(min_size)
    mol.DeleteHydrogens()
    if addH: 
        mol.CorrectForPH()
        mol.AddHydrogens()
    

    obConversion.WriteFile(mol, output_mol2_path)
    
    if cleanup:
        # Use MDA to remove clean file
        univ = mda.Universe(output_mol2_path)
        res_names = univ.residues.resnames
        new_names = [ "".join(re.findall(".*[a-zA-Z]+", name)).upper() for name in res_names]
        univ.residues.resnames = new_names
        univ = univ.select_atoms(selection_str)
        mda.coordinates.MOL2.MOL2Writer(output_mol2_path).write(univ)


def rebond_mol2(i, infile, structure_name, outfile, addH=False, strip_size=256):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol2", "pdb")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, infile)
    mol.DeleteHydrogens()
    mol.StripSalts(strip_size)
    obConversion.WriteFile(mol, 'temp{}.pdb'.format(i))
    pdb2mol2('temp{}.pdb'.format(i), structure_name, outfile, addH=addH)

    # Delete mol2 file
    os.remove('temp{}.pdb'.format(i))
    return None


def convert_all_pdb(structure_name, out_directory, addH=True, cleanup=True):
    output_path = out_directory + structure_name
    files = os.listdir(output_path)
    for file in files:
        if file.split('.')[-1] == 'pdb':
            pdb2mol2(f'{output_path}/{file}', structure_name, out_directory, addH=addH, out_name=file[:-4], cleanup=cleanup)
            os.remove(f'{output_path}/{file}')


def load_p2rank_set(file):
    df = pd.read_csv(file, sep='\s+', names=['path'], index_col=False)
    df['path'] = ['benchmark_data_dir/'+'/unprocessed_pdb/'.join(file.split('/')) for file in df['path']]
    
    return df


def load_p2rank_mlig(file, skiprows):
    df = pd.read_csv(file, sep='\s+', names=['path', 'ligands'], index_col=False, skiprows=skiprows)
    df = df[df['ligands'] != '<CONFLICTS>']
    df['path'] = ['benchmark_data_dir/'+'/unprocessed_pdb/'.join(file.split('/')) for file in df['path']]
    df['ligands'] = [l.split(',') for l in df['ligands']]
    
    return df


def write_mlig(df, out_file):
    with open(out_file, 'w') as f:
        f.write('HEADER: protein ligand_codes\n\n')
        for val in df.values:
            val[0] = '/'.join(val[0].split('/')[1:4:2])
            f.write(f'{val[0]}  {",".join(val[1])}\n')


def extract_residues_p2rank(mol_directory, univ_extension='pdb'):
    univ = mda.Universe(f'{mol_directory}/system.{univ_extension}')
    lig_ind = 0

    sel = univ.select_atoms('record_type HETATM and around 4 protein')
    for res in sel.residues:
        if (res.resname[:3] not in exclusion_list) and (res.atoms.n_atoms >= 5):
            com = res.atoms.center_of_mass()
            com_string = ' '.join(com.astype(str).tolist())
            not_protruding = univ.select_atoms(f'protein and not type H and point {com_string} 5.5').n_atoms > 0
            if not_protruding:
                write_fragment(res.atoms, univ, f'{mol_directory}/ligand_{lig_ind}.{univ_extension}', check_overlap=False)
                lig_ind += 1


def extract_residues_from_list(mol_directory, lig_resnames, univ_extension='pdb'):
    univ = mda.Universe(f'{mol_directory}/system.{univ_extension}')
    lig_ind = 0

    sel = univ.select_atoms('record_type HETATM and around 4 protein')
    for res in sel.residues:
        if (res.resname[:3] not in exclusion_list) and (res.resname[:3] in lig_resnames) and (res.atoms.n_atoms >= 5):
            com = res.atoms.center_of_mass()
            com_string = ' '.join(com.astype(str).tolist())
            not_protruding = univ.select_atoms(f'protein and not type H and point {com_string} 5.5').n_atoms > 0
            if not_protruding:
                write_fragment(res.atoms, univ, f'{mol_directory}/ligand_{lig_ind}.{univ_extension}', check_overlap=False)
                lig_ind += 1


def process_train_openbabel(i, file, output_dir):
    strip_size = 98 # Max number of atoms in a ligand in scPDB excluding hydrogens
    
    try:
        prepend = os.getcwd()
        structure_name = file
        if not os.path.isdir(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name)): 
            os.makedirs(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name))
        # print(os.path.join(prepend, '/scPDB_data_dir/unprocessed_mol2/',file,'/protein.mol2'))
        # print(prepend+'/scPDB_data_dir/unprocessed_mol2/'+file+'/protein.mol2')
        rebond_mol2(i,prepend+'/scPDB_data_dir/unprocessed_mol2/'+file+'/protein.mol2', structure_name, prepend+'/'+output_dir+"/ready_to_parse_mol2/",addH=True, strip_size=strip_size)
        
        for file_path in glob(prepend + '/scPDB_data_dir/unprocessed_mol2/' + file + '/*'):
            if 'ligand' in file_path:
                shutil.copyfile(file_path, prepend+'/'+ output_dir+"/ready_to_parse_mol2/"+file+'/'+file_path.split('/')[-1])
        
        label_sites_given_ligands(prepend + '/scPDB_data_dir/ready_to_parse_mol2/' + file)
        
        process_system((prepend + '/' + output_dir + '/ready_to_parse_mol2/' + structure_name), save_directory='./' + output_dir)
        
    except AssertionError as e: 
        print("Failed to find ligand in", file)
    except Exception as e:
        # print("ERROR", file, e)
        raise e
        

def process_train_classic(i, structure_name, output_dir, unprocessed_dir = 'unprocessed_scPDB_mol2'):
    # print("Processing", structure_name, flush=True)
    try:
        process_system(os.path.join('./',output_dir, unprocessed_dir, structure_name), save_directory='./' + output_dir)
    except AssertionError as e: 
        print("Failed to find ligand in", structure_name)
    except Exception as e:
        # print(e)
        raise e
        

def process_p2rank_set(path, data_dir="benchmark_data_dir", min_size=256):
    try:
        prepend = os.getcwd()
        structure_name = path.split('/')[-1].split('.')[0]
        mol2_dir = f'./{data_dir}/ready_to_parse_mol2/'
        protein2mol2(f'{prepend}/{path}', structure_name, mol2_dir, min_size=min_size, out_name='protein', cleanup=True)
        shutil.copyfile(f'{prepend}/{path}', f'{mol2_dir}{structure_name}/system.pdb') # copying pdb for ligand extraction
        extract_residues_p2rank(f'{mol2_dir}{structure_name}') # parsing pdb avoids selection issues
        label_sites_given_ligands(f'{mol2_dir}{structure_name}')
        convert_all_pdb(structure_name, mol2_dir, cleanup=False) # converting system and ligand pdbs to mol2s
        
        process_system(mol2_dir + structure_name, save_directory='./'+data_dir)
        # break
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        # print(e)
        raise e


def process_mlig_set(path, lig_resnames, data_dir="benchmark_data_dir", min_size=256):
    try:
        prepend = os.getcwd()
        structure_name = path.split('/')[-1].split('.')[0]
        mol2_dir = f'./{data_dir}/ready_to_parse_mol2/'
        protein2mol2(f'{prepend}/{path}', structure_name, mol2_dir, min_size=min_size, out_name='protein', cleanup=True)
        shutil.copyfile(f'{prepend}/{path}', f'{mol2_dir}{structure_name}/system.pdb') # copying pdb for ligand extraction
        extract_residues_from_list(f'{mol2_dir}{structure_name}', lig_resnames) # parsing pdb avoids selection issues
        label_sites_given_ligands(f'{mol2_dir}{structure_name}')
        convert_all_pdb(structure_name, mol2_dir, cleanup=False) # converting system and ligand pdbs to mol2s
        
        process_system(mol2_dir + structure_name, save_directory='./'+data_dir)
        # break
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        # print(e)
        raise e


if __name__ == "__main__":   
    num_cores = 24
    prepend = os.getcwd()
    from joblib.externals.loky import set_loky_pickler
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser(description="Prepare datasets for GNN inference.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", choices=["train_openbabel", "train_classic", "coach420", "coach420_mlig", "holo4k", "holo4k_mlig"], help="Dataset to prepare.")
    args = parser.parse_args()
    dataset = args.dataset
 
    if dataset == "train_openbabel":
        print("Parsing the standard train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/scPDB_data_dir/unprocessed_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_openbabel)(i, filename, 'scPDB_data_dir') for i, filename in enumerate(tqdm(mol2_files[:]))) 
        # for i, filename in enumerate(mol2_files[1800+360+380+250:]):
        #     process_train(i,filename, 'regular_data_dir')

    elif dataset == "train_classic":
        print("Parsing the standard train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/data_dir/unprocessed_scPDB_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_classic)(i, filename, 'data_dir') for i, filename in enumerate(tqdm(mol2_files[:]))) 
        # for i, filename in enumerate(mol2_files[1800+360+380+250:]):
        #     process_train(i,filename, 'regular_data_dir')

    elif dataset == "coach420":
        full_df = load_p2rank_set(f'{prepend}/benchmark_data_dir/coach420.ds')
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(full_df['path'][i], data_dir='/benchmark_data_dir/coach420', min_size=5) for i in tqdm(full_df.index))
    
    elif dataset == "coach420_mlig":
        full_df = load_p2rank_mlig(f'{prepend}/benchmark_data_dir/coach420(mlig).ds', skiprows=4)
        Parallel(n_jobs=num_cores)(delayed(process_mlig_set)(full_df['path'][i], full_df['ligands'][i], data_dir='/benchmark_data_dir/coach420_mlig', min_size=5) for i in tqdm(full_df.index))
    
    elif dataset == "holo4k":
        full_df = load_p2rank_set(f'{prepend}/benchmark_data_dir/holo4k.ds')
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(full_df['path'][i], data_dir='/benchmark_data_dir/holo4k', min_size=5) for i in tqdm(full_df.index))
    
    elif dataset == "holo4k_mlig":
        full_df = load_p2rank_mlig(f'{prepend}/benchmark_data_dir/holo4k(mlig).ds', skiprows=2)
        Parallel(n_jobs=num_cores)(delayed(process_mlig_set)(full_df['path'][i], full_df['ligands'][i], data_dir='/benchmark_data_dir/holo4k_mlig', min_size=5) for i in tqdm(full_df.index))
