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

allowed_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
allowed_elements = ['C', 'N', 'O', 'S']
res_selection_str = " or ".join([f'resname {x}' for x in allowed_residues])
atom_selection_str = " or ".join([f'element {x}' for x in allowed_elements]) # ignore post-translational modification
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
    protein_no_h = protein.select_atoms("not element H")
    
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
            try:
                this_ligands_site.atoms.write(os.path.join(path_to_mol2, f"site_for_ligand_{ligand_idx}.{extension}"))
            except IndexError as e:
                print(f'Unable to write site for {file_path}')

        else:
            # This is an unexpected file
            pass
    
    all_sites.write(os.path.join(path_to_mol2, f"site.{extension}"))


def cleanup_residues(univ):
        res_names = univ.residues.resnames
        new_names = [ "".join(re.findall(".*[a-zA-Z]+", name)).upper() for name in res_names]
        univ.residues.resnames = new_names
        
        return univ

def undo_se_modification(univ):
    se_atoms = univ.select_atoms('protein and element Se')
    se_atoms.elements = 'S'
    
    return univ

def clean_alternate_positions(input_dir, output_dir):
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    
    res_group = '|'.join(allowed_residues)
    r = re.compile(f'^ATOM.*([2-9]|[B-Z])({res_group})')
    
    for file_path in glob(f'{input_dir}/*.pdb'):
        with open(file_path, 'r') as infile:
            lines = infile.readlines()
            
        pdb_name = file_path.split('/')[-1]
        out_path = f'{output_dir}/{pdb_name}'
        
        with open(out_path, 'w') as outfile:
            outfile.writelines([line for line in lines if not re.search(r, line)])
    
def convert_to_mol2(in_file, structure_name, out_directory, addH=True, in_format='pdb', out_name='protein', parse_prot=True):
    ob_input = in_file
    output_path = out_directory + structure_name
    if not os.path.isdir(output_path): os.makedirs(output_path)
    
    if parse_prot:
        univ = mda.Universe(in_file)
        univ = cleanup_residues(univ)
        univ = undo_se_modification(univ)
        prot_atoms = univ.select_atoms(f'protein and ({res_selection_str}) and ({atom_selection_str})')
        ob_input = f'{output_path}/{out_name}.{in_format}'
        mda.coordinates.writer(ob_input).write(prot_atoms)
        
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats(in_format, "mol2")

    output_mol2_path  = f'{output_path}/{out_name}.mol2'

    mol = openbabel.OBMol()

    obConversion.ReadFile(mol, ob_input)
    mol.DeleteHydrogens()
    if addH: 
        mol.CorrectForPH()
        mol.AddHydrogens()
    
    obConversion.WriteFile(mol, output_mol2_path)
    
    if parse_prot:
        if ob_input != output_mol2_path:
            os.remove(ob_input) # cleaning temp file
        univ = mda.Universe(output_mol2_path) 
        univ = cleanup_residues(univ)
        mda.coordinates.writer(output_mol2_path).write(univ.atoms)


def convert_all_pdb(structure_name, out_directory, addH=True, parse_prot=True):
    output_path = out_directory + structure_name
    files = os.listdir(output_path)
    for file in files:
        if file.split('.')[-1] == 'pdb':
            convert_to_mol2(f'{output_path}/{file}', structure_name, out_directory, addH=addH, out_name=file[:-4], parse_prot=parse_prot)
            os.remove(f'{output_path}/{file}')


def load_p2rank_set(file, skiprows=0, pdb_dir='unprocessed_pdb', joined_style=False):
    df = pd.read_csv(file, sep='\s+', names=['path'], index_col=False, skiprows=skiprows)
    if joined_style: df['path'] = ['/'.join(file.split('/')[::2]) for file in df['path']] #removing subset directory
    df['path'] = ['benchmark_data_dir/'+f'/{pdb_dir}/'.join(file.split('/')) for file in df['path']]
    
    return df


def load_p2rank_mlig(file, skiprows, pdb_dir='unprocessed_pdb'):
    df = pd.read_csv(file, sep='\s+', names=['path', 'ligands'], index_col=False, skiprows=skiprows)
    df = df[df['ligands'] != '<CONFLICTS>']
    df['path'] = ['benchmark_data_dir/'+f'/{pdb_dir}/'.join(file.split('/')) for file in df['path']]
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
    univ.dimensions = None # this prevents PBC bugs in distance calculation
    lig_ind = 0

    sel = univ.select_atoms('record_type HETATM and around 4 protein')
    for res in sel.residues:
        if (res.resname[:3] not in exclusion_list) and (res.atoms.n_atoms >= 5):
            com = res.atoms.center_of_mass()
            com_string = ' '.join(com.astype(str).tolist())
            not_protruding = univ.select_atoms(f'protein and not element H and point {com_string} 5.5').n_atoms > 0
            if not_protruding:
                write_fragment(res.atoms, univ, f'{mol_directory}/ligand_{lig_ind}.{univ_extension}', check_overlap=False)
                lig_ind += 1


def extract_residues_from_list(mol_directory, lig_resnames, univ_extension='pdb'):
    univ = mda.Universe(f'{mol_directory}/system.{univ_extension}')
    univ.dimensions = None # this prevents PBC bugs in distance calculation
    lig_ind = 0

    sel = univ.select_atoms('record_type HETATM and around 4 protein')
    for res in sel.residues:
        if (res.resname[:3] not in exclusion_list) and (res.resname[:3] in lig_resnames) and (res.atoms.n_atoms >= 5):
            com = res.atoms.center_of_mass()
            com_string = ' '.join(com.astype(str).tolist())
            not_protruding = univ.select_atoms(f'protein and not element H and point {com_string} 5.5').n_atoms > 0
            if not_protruding:
                write_fragment(res.atoms, univ, f'{mol_directory}/ligand_{lig_ind}.{univ_extension}', check_overlap=False)
                lig_ind += 1


def check_p2rank_criteria(prot_univ, lig_univ):
    prot_univ.add_TopologyAttr('record_type')
    prot_univ.atoms.record_types = 'protein'
    lig_univ.add_TopologyAttr('record_type')
    lig_univ.atoms.record_types = 'ligand'


    univ = mda.Merge(prot_univ.atoms, lig_univ.atoms)
    univ.dimensions = None # this prevents PBC bugs in distance calculation

    valid_resnames = np.all([res.resname not in exclusion_list for res in lig_univ.residues])
    not_prot = np.all([res.resname not in allowed_residues for res in lig_univ.residues])
    nearby = univ.select_atoms('record_type ligand and around 4 protein').n_atoms > 0
    if valid_resnames and not_prot and nearby and (lig_univ.atoms.n_atoms >= 5):
        com = lig_univ.atoms.center_of_mass()
        com_string = ' '.join(com.astype(str).tolist())
        not_protruding = univ.select_atoms(f'protein and not element H and point {com_string} 5.5').n_atoms > 0
        if not_protruding:
            return True

    return False


def process_train_p2rank(file, output_dir): 
    try:
        prepend = os.getcwd()
        structure_name = file

        if not os.path.isdir(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name)): 
            os.makedirs(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name))

        convert_to_mol2(f'{prepend}/scPDB_data_dir/unprocessed_mol2/{file}/protein.mol2', structure_name, f'{prepend}/{output_dir}/ready_to_parse_mol2/', addH=True, in_format='mol2')
        
        for file_path in glob(f'{prepend}/scPDB_data_dir/unprocessed_mol2/{file}/*'):
            if 'ligand' in file_path:
                prot_univ = mda.Universe(f'{prepend}/{output_dir}/ready_to_parse_mol2/{file}/protein.mol2') 
                lig_univ = mda.Universe(file_path)
                passing_p2rank = check_p2rank_criteria(prot_univ, lig_univ)
                if passing_p2rank:
                    shutil.copyfile(file_path, f'{prepend}/{output_dir}/ready_to_parse_mol2/{file}/{file_path.split("/")[-1]}')
        
        n_ligands = np.sum(['ligand' in file for file in os.listdir(f'{prepend}/{output_dir}/ready_to_parse_mol2/{file}/')])
        if n_ligands > 0:
            label_sites_given_ligands(f'{prepend}/{output_dir}/ready_to_parse_mol2/{structure_name}')
            if not os.path.isdir(f'{prepend}/{output_dir}/raw'): os.makedirs(f'{prepend}/{output_dir}/raw')
            if not os.path.isdir(f'{prepend}/{output_dir}/mol2'): os.makedirs(f'{prepend}/{output_dir}/mol2')
            process_system(f'{prepend}/{output_dir}/ready_to_parse_mol2/{structure_name}', save_directory=f'./{output_dir}')
        else:
            with open(f'{prepend}/{output_dir}/no_ligands.txt', 'a') as f:
                f.write(f'{structure_name}\n')
        
    except AssertionError as e: 
        print("Failed to find ligand in", file)
    except Exception as e:
        raise e

def process_train_openbabel(file, output_dir):
    try:
        prepend = os.getcwd()
        structure_name = file
        if not os.path.isdir(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name)): 
            os.makedirs(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name))
        convert_to_mol2(prepend+'/scPDB_data_dir/unprocessed_mol2/'+file+'/protein.mol2', structure_name, prepend+'/'+output_dir+"/ready_to_parse_mol2/", addH=True, in_format='mol2')
        
        for file_path in glob(prepend + '/scPDB_data_dir/unprocessed_mol2/' + file + '/*'):
            if 'ligand' in file_path:
                shutil.copyfile(file_path, prepend+'/'+ output_dir+"/ready_to_parse_mol2/"+file+'/'+file_path.split('/')[-1])
        
        label_sites_given_ligands(prepend + '/scPDB_data_dir/ready_to_parse_mol2/' + file)
        
        process_system((prepend + '/' + output_dir + '/ready_to_parse_mol2/' + structure_name), save_directory='./' + output_dir)
        
    except AssertionError as e: 
        print("Failed to find ligand in", file)
    except Exception as e:
        raise e
        

def process_train_classic(structure_name, output_dir, unprocessed_dir = 'unprocessed_scPDB_mol2'):
    try:
        process_system(os.path.join('./',output_dir, unprocessed_dir, structure_name), save_directory='./' + output_dir)
    except AssertionError as e: 
        print("Failed to find ligand in", structure_name)
    except Exception as e:
        raise e
        

def process_p2rank_set(path, data_dir="benchmark_data_dir", chen_fix=False):
    print(f'\n PARSING {path} \n', flush=True)
    try:
        prepend = os.getcwd()
        structure_name = '.'.join(path.split('/')[-1].split('.')[:-1])
        mol2_dir = f'{prepend}/{data_dir}/ready_to_parse_mol2/'
        convert_to_mol2(f'{prepend}/{path}', structure_name, mol2_dir, out_name='protein', parse_prot=True)
        shutil.copyfile(f'{prepend}/{path}', f'{mol2_dir}{structure_name}/system.pdb') # copying pdb for ligand extractio
        if chen_fix:
            univ = mda.Universe(f'{mol2_dir}{structure_name}/system.pdb')
            mda.coordinates.writer(f'{mol2_dir}{structure_name}/system.pdb').write(univ.atoms)
        extract_residues_p2rank(f'{mol2_dir}{structure_name}') # parsing pdb avoids selection issues

        n_ligands = np.sum(['ligand' in file for file in os.listdir(f'{mol2_dir}{structure_name}')])
        if n_ligands > 0:
            label_sites_given_ligands(f'{mol2_dir}{structure_name}')
            if not os.path.isdir(f'{prepend}/{data_dir}/raw'): os.makedirs(f'{prepend}/{data_dir}/raw')
            if not os.path.isdir(f'{prepend}/{data_dir}/mol2'): os.makedirs(f'{prepend}/{data_dir}/mol2')
            process_system(mol2_dir + structure_name, save_directory='./'+data_dir)
        else:
            with open(f'{prepend}/{data_dir}/no_ligands.txt', 'a') as f:
                f.write(f'{structure_name}\n')
                
        convert_all_pdb(structure_name, mol2_dir, parse_prot=False) # converting system and ligand pdbs to mol2s
        
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        raise e


def process_mlig_set(path, lig_resnames, data_dir="benchmark_data_dir"):
    try:
        prepend = os.getcwd()
        structure_name = path.split('/')[-1].split('.')[0]
        mol2_dir = f'{prepend}/{data_dir}/ready_to_parse_mol2/'
        convert_to_mol2(f'{prepend}/{path}', structure_name, mol2_dir, out_name='protein', parse_prot=True)
        shutil.copyfile(f'{prepend}/{path}', f'{mol2_dir}{structure_name}/system.pdb') # copying pdb for ligand extraction
        extract_residues_from_list(f'{mol2_dir}{structure_name}', lig_resnames) # parsing pdb avoids selection issues

        n_ligands = np.sum(['ligand' in file for file in os.listdir(f'{mol2_dir}{structure_name}')])
        if n_ligands > 0:
            label_sites_given_ligands(f'{mol2_dir}{structure_name}')
            if not os.path.isdir(f'{prepend}/{data_dir}/raw'): os.makedirs(f'{prepend}/{data_dir}/raw')
            if not os.path.isdir(f'{prepend}/{data_dir}/mol2'): os.makedirs(f'{prepend}/{data_dir}/mol2')
            process_system(mol2_dir + structure_name, save_directory='./'+data_dir)
        else:
            with open(f'{prepend}/{data_dir}/no_ligands.txt', 'a') as f:
                f.write(f'{structure_name}\n')

        convert_all_pdb(structure_name, mol2_dir, parse_prot=False) # converting system and ligand pdbs to mol2s
        
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        raise e


if __name__ == "__main__":
    num_cores = 24
    prepend = os.getcwd()
    from joblib.externals.loky import set_loky_pickler
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser(description="Prepare datasets for GNN inference.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", choices=["train_p2rank", "train_openbabel", "train_classic", "coach420", "coach420_mlig", "holo4k", "holo4k_mlig", "chen11", "joined"], help="Dataset to prepare.")
    args = parser.parse_args()
    dataset = args.dataset
 
    if dataset == "train_p2rank":
        print("Parsing the standard train set with p2rank criteria")
        nolig_file = f'{prepend}/scPDB_data_dir/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/scPDB_data_dir/unprocessed_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_p2rank)(filename, 'scPDB_data_dir') for filename in tqdm(mol2_files[:])) 

    elif dataset == "train_openbabel":
        print("Parsing the standard train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/scPDB_data_dir/unprocessed_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_openbabel)(filename, 'scPDB_data_dir') for filename in tqdm(mol2_files[:])) 

    elif dataset == "train_classic":
        print("Parsing the standard train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/data_dir/unprocessed_scPDB_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_classic)(filename, 'data_dir') for filename in tqdm(mol2_files[:])) 

    elif dataset == "coach420":
        full_df = load_p2rank_set(f'{prepend}/benchmark_data_dir/coach420.ds')
        nolig_file = f'{prepend}/benchmark_data_dir/coach420/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(full_df['path'][i], data_dir='/benchmark_data_dir/coach420') for i in tqdm(full_df.index))
    
    elif dataset == "coach420_mlig":
        full_df = load_p2rank_mlig(f'{prepend}/benchmark_data_dir/coach420(mlig).ds', skiprows=4)
        nolig_file = f'{prepend}/benchmark_data_dir/coach420_mlig/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        Parallel(n_jobs=num_cores)(delayed(process_mlig_set)(full_df['path'][i], full_df['ligands'][i], data_dir='/benchmark_data_dir/coach420_mlig') for i in tqdm(full_df.index))
    
    elif dataset == "holo4k":
        print('Cleaning alternate positions...')
        clean_alternate_positions(f'{prepend}/benchmark_data_dir/holo4k/unprocessed_pdb/', f'{prepend}/benchmark_data_dir/holo4k/cleaned_pdb/')
        full_df = load_p2rank_set(f'{prepend}/benchmark_data_dir/holo4k.ds', pdb_dir='cleaned_pdb')
        nolig_file = f'{prepend}/benchmark_data_dir/holo4k/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(full_df['path'][i], data_dir='/benchmark_data_dir/holo4k') for i in tqdm(full_df.index))
    
    elif dataset == "holo4k_mlig":
        print('Cleaning alternate positions...')
        clean_alternate_positions(f'{prepend}/benchmark_data_dir/holo4k/unprocessed_pdb/', f'{prepend}/benchmark_data_dir/holo4k/cleaned_pdb/')
        full_df = load_p2rank_mlig(f'{prepend}/benchmark_data_dir/holo4k(mlig).ds', skiprows=2, pdb_dir='cleaned_pdb')
        nolig_file = f'{prepend}/benchmark_data_dir/holo4k_mlig/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        Parallel(n_jobs=num_cores)(delayed(process_mlig_set)(full_df['path'][i], full_df['ligands'][i], data_dir='/benchmark_data_dir/holo4k_mlig') for i in tqdm(full_df.index))

    elif dataset == "chen11":
        full_df = load_p2rank_set(f'{prepend}/benchmark_data_dir/chen11.ds', skiprows=5)
        nolig_file = f'{prepend}/benchmark_data_dir/chen11/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(full_df['path'][i], data_dir='/benchmark_data_dir/chen11', chen_fix=True) for i in tqdm(full_df.index))

    elif dataset == "joined":
        print('Cleaning alternate positions...')
        clean_alternate_positions(f'{prepend}/benchmark_data_dir/joined/unprocessed_pdb/', f'{prepend}/benchmark_data_dir/joined/cleaned_pdb/')
        full_df = load_p2rank_set(f'{prepend}/benchmark_data_dir/joined.ds', pdb_dir='cleaned_pdb', joined_style=True)
        nolig_file = f'{prepend}/benchmark_data_dir/joined/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(full_df['path'][i], data_dir='/benchmark_data_dir/joined', chen_fix=True) for i in tqdm(full_df.index))
        