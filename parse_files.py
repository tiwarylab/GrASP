from dataclasses import field
from turtle import end_fill
from parsing_regular import process_system
from merge import write_fragment
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import os
import sys
import numpy as np
import pandas as pd
import shutil
import openbabel
from tqdm import tqdm
from glob import glob
import re

allowed_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'C', 'G', 'A', 'U', 'I', 'DC', 'DG', 'DA', 'DU', 'DT', 'DI']
selection_str = "".join(["resname " + x + " or " for x in list(allowed_residues)[:-1]]) + "resname " + str(allowed_residues[-1])


def label_sites_given_ligands(path_to_mol2, use_pdb=False):
    if use_pdb: extension = 'pdb'
    else: extension = 'mol2'
    protein = mda.Universe(os.path.join(path_to_mol2, f'protein.{extension}'))
    protein_no_h = protein.select_atoms("not type H")
    ligand_idx = 0
    all_site_resids = []
    for file_path in sorted(glob(path_to_mol2+ '/*')):
        if 'protein' in file_path:
            # This is the main structure, we already have it
            pass
        elif 'ligand' in file_path and not 'site' in file_path:
            # This is a ligand file
            ligand = mda.Universe(file_path)
            site_resid_list = []
            for atom in ligand.atoms:
                x,y,z = atom.position
                resids_to_label = list(protein_no_h.select_atoms("point {} {} {} 6.5".format(x, y, z)).resids)
                site_resid_list += resids_to_label
                all_site_resids += resids_to_label
            
            site_resid_list = list(set(site_resid_list))
            site_selection_str = "".join(["resid " + str(x) + " or " for x in site_resid_list[:-1]] + ["resid " + str(site_resid_list[-1])])

            this_ligands_site = protein.select_atoms(site_selection_str)
            this_ligands_site.atoms.write(os.path.join(path_to_mol2,f"site_for_ligand_{ligand_idx}.{extension}"))
            ligand_idx += 1
        else:
            # This is an unexpected file
            pass
    site_resid_list = list(set(all_site_resids))
    
    # site_selection_str = "".join(["resid " + str(x) + " or " for x in site_resid_list[:-1]] + ["resid " + str(site_resid_list[-1])])
    # protein.select_atoms(site_selection_str).atoms.write(os.path.join(path_to_mol2,"site.mol2"))
    protein.residues[np.in1d(protein.residues.resids, site_resid_list)].atoms.write(os.path.join(path_to_mol2, f"site.{extension}"))

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
        new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
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
        new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
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


def rebond_pdb(pdb_file, structure_name, out_directory, min_size=256, addH=True, out_name='protein', cleanup=True):
    output_path = out_directory + structure_name
    if not os.path.isdir(output_path): os.makedirs(output_path)
    
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdb")

    output_pdb_path  = output_path + '/' + out_name +'.pdb'

    mol = openbabel.OBMol()

    obConversion.ReadFile(mol, pdb_file)
    mol.StripSalts(min_size)
    mol.DeleteHydrogens()
    if addH: 
        mol.CorrectForPH()
        mol.AddHydrogens()
    

    obConversion.WriteFile(mol, output_pdb_path)
    
    if cleanup:
        # Use MDA to remove clean file
        univ = mda.Universe(output_pdb_path)
        res_names = univ.residues.resnames
        new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
        univ.residues.resnames = new_names
        univ = univ.select_atoms(selection_str)
        mda.coordinates.PDB.PDBWriter(output_pdb_path).write(univ)

def convert_all_pdb(structure_name, out_directory, addH=True, cleanup=True):
    output_path = out_directory + structure_name
    files = os.listdir(output_path)
    for file in files:
        if file.split('.')[-1] == 'pdb':
            pdb2mol2(f'{output_path}/{file}', structure_name, out_directory, addH=addH, out_name=file[:-4], cleanup=cleanup)
            os.remove(f'{output_path}/{file}')

 
# def process_train_openbabel(i, file, output_dir):
#     print("Processing", file, flush=True)
#     try:
#         prepend = os.getcwd()
#         structure_name = file
        
#         rebond_mol2(i ,os.path.join(prepend, ',data_dir/unprocessed_mol2/', structure_name,'/protein.mol2'), structure_name, prepend+'/'+output_dir,add_H=True)
        
#         label_protein_site(prepend + '/data_dir/unprocessed_mol2/' + file, structure_name, out_directory=prepend +'/data_dir')
        
        
#         path_to_pdb = prepend +'/' + output_dir + '/unprocessed_scPDB_mol2/'+structure_name +"/"
#         for file_name in os.listdir(path_to_pdb):
#             if 'protein' not in file_name:
#                 # Do not add hydrogens to sites, they will not be used for labeling and moreover will  mess up comparison between 'ground truth' and predictions
#                 pdb2mol2(path_to_pdb+file_name, structure_name,prepend+'/data_dir',addH=False) 
#             else:
        
        
#         rebond_mol2(i,path_to_pdb+'protein.mol2',structure_name,prepend+'/' + output_dir, addH=True)
        
#         # mol22mol2(path_to_pdb+"protein.mol2", structure_name, prepend+output_dir,addH=True, out_name='protein')
#         if not os.path.isdir(prepend+'/' + output_dir + '/unprocessed_mol2/'+structure_name): 
#             os.makedirs(prepend+'/' + output_dir + '/unprocessed_mol2/'+structure_name)
#         shutil.copyfile(path_to_pdb+'site.mol2', prepend+'/' + output_dir + '/unprocessed_mol2/'+structure_name+'/site.mol2')
#         # print("processing system")
#         process_system('./' + output_dir + '/unprocessed_mol2/' + structure_name, save_directory='./' + output_dir)
#         # break
#     except AssertionError as e: 
#         print("Failed to find ligand in", file)
#     except Exception as e:
#         print(e)

def load_ligand_list(file, skiprows):
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

def deeppocket_mlig(dp_file, full_df, out_file):
    df = pd.read_csv(dp_file, sep=',', names=['path', 'pockets'], index_col=False)
    df['path'] = [f'benchmark_data_dir/{"/unprocessed_pdb/".join(file.split("/"))}.pdb' for file in df['path']]
    df = df[df['pockets'] > 0]

    dp_lig_df = full_df[full_df['path'].isin(df['path'])]
    write_mlig(dp_lig_df, out_file)

def extract_ligands(mol_directory):
    univ = mda.Universe(f'{mol_directory}/system.mol2')
    lig_ind = 0
    for frag in univ.atoms.fragments:
        if  3 < frag.n_atoms < 256:
            write_fragment(frag, univ, f'{mol_directory}/ligand_{lig_ind}.mol2')
            lig_ind += 1

def extract_ligands_from_list(mol_directory, lig_resnames):
    univ = mda.Universe(f'{mol_directory}/system.mol2') 
    prot = mda.Universe(f'{mol_directory}/protein.mol2')
    lig_ind = 0
    for frag in univ.atoms.fragments:
        res_set = {res.resname[:3] for res in frag.residues}
        if  res_set.issubset(set(lig_resnames)) and not res_set.issubset(set(allowed_residues)):
            frag_dist = np.min(distance_array(frag.positions, prot.select_atoms("not type H").positions))
            if frag_dist <= 6.5: 
                write_fragment(frag, univ, f'{mol_directory}/ligand_{lig_ind}.mol2')
                lig_ind += 1
    
    if lig_ind == 0: # if no ligands were found, check if they are bonded
        print(f'Residue-level selection needed for {mol_directory}', flush=True)
        extract_residues_from_list(mol_directory, lig_resnames)



def extract_residues_from_list(mol_directory, lig_resnames):
    univ = mda.Universe(f'{mol_directory}/system.mol2')
    prot = mda.Universe(f'{mol_directory}/protein.mol2')
    lig_ind = 0
    for res in univ.atoms.residues:
        if (res.resname[:3] in lig_resnames) and (res.resname[:3] not in allowed_residues):
            res_dist = np.min(distance_array(res.atoms.positions, prot.select_atoms("not type H").positions))
            if res_dist <= 6.5: 
                write_fragment(res.atoms, univ, f'{mol_directory}/ligand_{lig_ind}.mol2')
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
        
    
def move_SC6K(num_cores, verbose=False):
    if not os.path.isdir(os.path.join(prepend,'SC6K_data_dir/SC6K/unprocessed_mol2')): os.makedirs(os.path.join(prepend,'SC6K_data_dir/SC6K/unprocessed_mol2'))
    
    prot_pattern = re.compile("([a-zA-Z0-9]{4}_[0-9]+).*PROT\.pdb")         # Although we have the mol2 versions of these files, OB doesn't seem to like them
    site_pattern = re.compile("([a-zA-Z0-9]{4}_[0-9]+).*SITE\.mol2")
    
    if not verbose: 
        lst = tqdm(sorted(os.listdir(os.path.join(prepend,'SC6K_data_dir/SC6K/'))))
    else:
        lst = sorted(os.listdir(os.path.join(prepend,'SC6K_data_dir/SC6K/'))) 
        
    def move_files(rcsb_id):
        if verbose: print(rcsb_id)
        for file in os.listdir(os.path.join(prepend,'SC6K_data_dir/SC6K/',rcsb_id)):
            prot_name = re.fullmatch(prot_pattern, file)
            if prot_name is not None:
                prot_name = prot_name.groups()[0]
            site_name = re.fullmatch(site_pattern, file)   
            if site_name is not None:
                site_name = site_name.groups()[0]
            
            if prot_name is not None:
                output_path = os.path.join(prepend,'SC6K_data_dir/unprocessed_mol2/',prot_name)
                if not os.path.isdir(output_path): os.makedirs(output_path)
                
                mol = openbabel.OBMol()
                obConversion = openbabel.OBConversion()
                obConversion.SetInAndOutFormats("pdb", "mol2")
                obConversion.ReadFile(mol, os.path.join(prepend,'SC6K_data_dir/SC6K/',rcsb_id,file))
                mol.DeleteHydrogens()
                mol.CorrectForPH()
                mol.AddHydrogens()
                
                obConversion.WriteFile(mol, os.path.join(output_path, "protein.mol2"))
                
            elif site_name is not None:
                output_path = os.path.join(prepend,'SC6K_data_dir/unprocessed_mol2/',site_name)
                if not os.path.isdir(output_path): os.makedirs(output_path)
                shutil.copyfile(os.path.join(prepend,'SC6K_data_dir/SC6K/',rcsb_id,file), os.path.join(output_path,'site.mol2'))
    
    Parallel(n_jobs=num_cores)(delayed(move_files)(rcsb_id) for rcsb_id in lst) 
        
    
                
            
            
        
# import threading, time, random
# def process_train_helper(i, file):
#     thread = threading.Thread(target=process_train, args=(i, file))
#     thread.start()
#     thread.join()

def process_chen(file, data_dir="benchmark_data_dir"):
    try:
        prepend = os.getcwd()
        structure_name = file.split('_')[1][:-4]
        mol2_dir = f'./{data_dir}/ready_to_parse_mol2/' 
        path_to_pdb = prepend +'/'+data_dir+'/unprocessed_pdb/'
        pdb2mol2(path_to_pdb+file, structure_name, mol2_dir, out_name='system', cleanup=False)
        protein2mol2(path_to_pdb+file, structure_name, mol2_dir, min_size=231, out_name='protein', cleanup=True)
        extract_ligands(f'{mol2_dir}{structure_name}/')
        label_sites_given_ligands(f'{mol2_dir}{structure_name}')
        
        process_system(mol2_dir + structure_name, save_directory='./'+data_dir)
        # break
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        # print(e)
        raise e

def process_mlig(path, lig_resnames, data_dir="benchmark_data_dir", min_size=256):
    print(f'start {path}', flush=True)
    try:
        prepend = os.getcwd()
        structure_name = path.split('/')[-1].split('.')[0]
        mol2_dir = f'./{data_dir}/ready_to_parse_mol2/'
        pdb2mol2(f'{prepend}/{path}', structure_name, mol2_dir , out_name='system', cleanup=False)
        rebond_pdb(f'{prepend}/{path}', structure_name, mol2_dir, min_size=min_size, out_name='protein', cleanup=True)
        pdb2mol2(f'{mol2_dir}{structure_name}/protein.pdb', structure_name, mol2_dir , out_name='protein', cleanup=False)
        extract_ligands_from_list(f'{mol2_dir}{structure_name}/', lig_resnames)
        label_sites_given_ligands(f'{mol2_dir}{structure_name}', use_pdb=True)
        convert_all_pdb(structure_name, mol2_dir) # we get sites w/ pdb then convert to avoid issues w/ duplicate resids
        
        process_system(mol2_dir + structure_name, save_directory='./'+data_dir)
        # break
        print(f'end {path}', flush=True)
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        # print(e)
        raise e

if __name__ == "__main__":   
    num_cores = 1
    prepend = os.getcwd()
    from joblib.externals.loky import set_loky_pickler
    from joblib import Parallel, delayed
 
    if str(sys.argv[1]) == "chen":
        pdb_files = [filename for filename in sorted(list(os.listdir(prepend +'/benchmark_data_dir/chen/unprocessed_pdb')))]
        Parallel(n_jobs=num_cores)(delayed(process_chen)(filename, data_dir='/benchmark_data_dir/chen') for _, filename in enumerate(tqdm(pdb_files)))
    elif str(sys.argv[1]) == "train_openbabel":
        print("Parsing the standard train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/scPDB_data_dir/unprocessed_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_openbabel)(i, filename, 'scPDB_data_dir') for i, filename in enumerate(tqdm(mol2_files[:]))) 
        # for i, filename in enumerate(mol2_files[1800+360+380+250:]):
        #     process_train(i,filename, 'regular_data_dir')
    elif str(sys.argv[1]) == "train_classic":
        print("Parsing the standard train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/data_dir/unprocessed_scPDB_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_classic)(i, filename, 'data_dir') for i, filename in enumerate(tqdm(mol2_files[:]))) 
        # for i, filename in enumerate(mol2_files[1800+360+380+250:]):
        #     process_train(i,filename, 'regular_data_dir')
    elif str(sys.argv[1]) == "train_trimmed":
        raise DeprecationWarning ("This pipeline is deprecated")
        # print("Parsing the standard train set")
        # mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/data_dir/unprocessed_scPDB_mol2')))]
        # Parallel(n_jobs=num_cores)(delayed(process_train_trimmed)(i, filename, 'regular_data_dir') for i, filename in enumerate(tqdm(mol2_files[4000:])))               # POINTS TO SAME DIR AS OPEN BABEL TO SAVE SPACE BE CAREFUL
        # for i, filename in enumerate(mol2_files[1800+360+380+250:]):
        #     process_train(i,filename, 'regular_data_dir')
    elif str(sys.argv[1]) == "train_hetro":
        print("Parsing the heterogeneous train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/hetro_data_dir/unprocessed_scPDB_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_classic)(i, filename, 'hetro_data_dir') for i, filename in enumerate(tqdm(mol2_files[:]))) 
        # for i, filename in enumerate(mol2_files[:]):
        #     process_train(i,filename, 'hetro_data_dir')
    elif str(sys.argv[1]) == "coach420":
        full_df = load_ligand_list(f'{prepend}/benchmark_data_dir/coach420(mlig)-deeppocket.ds', skiprows=2)
        Parallel(n_jobs=num_cores)(delayed(process_mlig)(full_df['path'][i], full_df['ligands'][i], data_dir='/benchmark_data_dir/coach420', min_size=4) for i in tqdm(full_df.index))
    elif str(sys.argv[1]) == "holo4k":
        full_df = load_ligand_list(f'{prepend}/benchmark_data_dir/holo4k(mlig)-deeppocket.ds', skiprows=2)
        Parallel(n_jobs=num_cores)(delayed(process_mlig)(full_df['path'][i], full_df['ligands'][i], data_dir='/benchmark_data_dir/holo4k', min_size=4) for i in tqdm(full_df.index[1868:1918]))
    elif str(sys.argv[1] == "SC6K"):
        if len(sys.argv) < 3: 
            move_SC6K(num_cores, verbose=False)
        elif str(sys.argv[2]) == "-v":
            move_SC6K(num_cores, verbose=True)
        else:
            raise IOError()
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/SC6K_data_dir/unprocessed_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_classic)(i, filename, 'SC6K_data_dir', unprocessed_dir='unprocessed_mol2') for i, filename in enumerate(tqdm(mol2_files[:]))) 
    else:
        print("Expected first argument to be 'test', 'train', or 'train_hetro' instead got " + str(sys.argv[1]))
