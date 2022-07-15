from dataclasses import field
from turtle import end_fill
from featurize_protein import process_system
from merge import write_fragment
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
from scipy.spatial import ConvexHull
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
        new_names = [ "".join(re.findall(".*[a-zA-Z]+", name)).upper() for name in res_names]
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
    prot = mda.Universe(f'{mol_directory}/protein.mol2')
    lig_ind = 0
    for frag in univ.atoms.fragments:
        if  3 < frag.n_atoms < 256:
            frag_dist = np.min(distance_array(frag.positions, prot.select_atoms("not type H").positions))
            if frag_dist <= 6.5: 
                write_fragment(frag, univ, f'{mol_directory}/ligand_{lig_ind}.mol2')
                lig_ind += 1


# def extract_ligands_from_list(mol_directory, lig_resnames):
#     univ = mda.Universe(f'{mol_directory}/system.mol2') 
#     prot = mda.Universe(f'{mol_directory}/protein.mol2')
#     lig_ind = 0
#     for frag in univ.atoms.fragments:
#         res_set = {res.resname[:3] for res in frag.residues}
#         if  res_set.issubset(set(lig_resnames)) and not res_set.issubset(set(allowed_residues)):
#             frag_dist = np.min(distance_array(frag.positions, prot.select_atoms("not type H").positions))
#             if frag_dist <= 6.5: 
#                 write_fragment(frag, univ, f'{mol_directory}/ligand_{lig_ind}.mol2')
#                 lig_ind += 1
    
#     if lig_ind == 0: # if no ligands were found, check if they are bonded
#         print(f'Residue-level selection needed for {mol_directory}', flush=True)
#         extract_residues_from_list(mol_directory, lig_resnames)


def extract_residues_from_list(mol_directory, lig_resnames, univ_extension='mol2', prot_extension='mol2'):
    univ = mda.Universe(f'{mol_directory}/system.{univ_extension}')
    prot = mda.Universe(f'{mol_directory}/protein.{prot_extension}')
    if univ_extension == 'mol2': add_chains_from_frags(univ)

    lig_ind = 0
    for res in univ.atoms.residues:
        if (res.resname[:3] in lig_resnames) and (res.resname[:3] not in allowed_residues):
            chains = np.unique(res.atoms.chainIDs)
            for chain in chains: # this catches mol2 files where a residue is multiple fragments in error
                res_atoms = res.atoms.select_atoms(f'chainID {chain}')
                res_dist = np.min(distance_array(res_atoms.positions, prot.select_atoms("not type H").positions))
                if res_dist <= 6.5: 
                    write_fragment(res_atoms, univ, f'{mol_directory}/ligand_{lig_ind}.{univ_extension}', check_overlap=False)
                    lig_ind += 1


def inside_hull(convex_hull, point):
    vertices = convex_hull.points[convex_hull.vertices]
    
    concat_points = np.row_stack([vertices, point])
    cat_hull = ConvexHull(concat_points)
    cat_vertices = cat_hull.points[cat_hull.vertices]
    
    inside = np.array_equal(vertices, cat_vertices)
    
    return inside


def fraction_inside(convex_hull, ligand):
    lig_coords = ligand.atoms.positions
    inside = [inside_hull(convex_hull, coord) for coord in lig_coords]
    
    return np.mean(inside)


def cavity_distances(cavity, ligand):
    all_distances = distance_array(cavity.atoms.positions, ligand.atoms.positions)
    min_distances = np.min(all_distances, axis=0)
    
    return min_distances


def write_matched_ligand(cavity_path, ligand_path, output_dir):
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    dp_cavity_name = cavity_path.split('/')[-2]
    dp_cavity_num = dp_cavity_name.split('ligand')[-1]
    if 'new' in dp_cavity_num: 
        dp_cavity_num = dp_cavity_num.split('new')[-1]
    with open(f'{output_dir}cavity_association.txt', 'a') as outfile:
        outfile.write(f'{dp_cavity_name}: {ligand_path}\n')
        
    shutil.copyfile(ligand_path, f'{output_dir}ligand_{dp_cavity_num}.mol2')


def match_ligands(dp_data_path, orig_data_path, new_data_path, structure_name):
    cavity_dir = f'{dp_data_path}{structure_name}/'
    ligand_dir = f'{orig_data_path}{structure_name}/'
    output_dir = f'{new_data_path}{structure_name}/'
    
    cavity_paths = [cavity_dir + path for path in os.listdir(cavity_dir) if os.path.isdir(cavity_dir + path)]
    for i, path in enumerate(cavity_paths):
        contents = os.listdir(path)
        mol2_count = 0
        for file in contents:
            if file.split('.')[-1] == 'mol2':
                cavity_paths[i] += f'/{file}'
                mol2_count += 1
        if mol2_count > 1: print(f'Warning: Multiple MOL2 found in {path}!')
    
    ligand_paths = [ligand_dir + path for path in os.listdir(ligand_dir) if 'ligand' in path and not 'site' in path]
    
    with open(f'{output_dir}cavity_association.txt', 'w') as outfile:
        outfile.write('Link between DeepPocket cavities and our ligands.\nThe format is cavity name:ligand path.\n\n')
    
    for cavity_path in cavity_paths:
        cavity = mda.Universe(cavity_path)
        if len(cavity.atoms) >= 4:
            cav_hull = ConvexHull(cavity.atoms.positions)
        
            ligand_fractions = []
            for ligand_path in ligand_paths:
                ligand = mda.Universe(ligand_path)
                ligand_fractions.append(fraction_inside(cav_hull, ligand))
        else: 
            print(f'Warning: The cavity at {cavity_path} only has {len(cavity.atoms)} points.')
            ligand_fractions = 0
            
        if np.max(ligand_fractions) < 1e-4:
            print(f'Warning: No ligand found within the cavity at {cavity_path}, using closest ligand.')
            average_dists = []
            for ligand_path in ligand_paths:
                ligand = mda.Universe(ligand_path)
                average_dists.append(np.mean(cavity_distances(cavity, ligand)))
                
            closest_avg = np.min(average_dists)
            print(f'Closest ligand found with average distance of {closest_avg:.2f} A.')
            selected_ligand = ligand_paths[np.argmin(average_dists)]
            
        elif np.max(ligand_fractions) < .5:
            print(f'Warning: Ligand is less than half inside {cavity_path}')
            selected_ligand = ligand_paths[np.argmax(ligand_fractions)]
            
        else:
            selected_ligand = ligand_paths[np.argmax(ligand_fractions)]
            
        write_matched_ligand(cavity_path, selected_ligand, output_dir)


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
    try:
        prepend = os.getcwd()
        structure_name = path.split('/')[-1].split('.')[0]
        mol2_dir = f'./{data_dir}/ready_to_parse_mol2/'
        protein2mol2(f'{prepend}/{path}', structure_name, mol2_dir, min_size=min_size, out_name='protein', cleanup=True)
        shutil.copyfile(f'{prepend}/{path}', f'{mol2_dir}{structure_name}/system.pdb') # copying pdb for ligand extraction
        extract_residues_from_list(f'{mol2_dir}{structure_name}', lig_resnames, univ_extension='pdb') # parsing pdb avoids selection issues
        label_sites_given_ligands(f'{mol2_dir}{structure_name}')
        convert_all_pdb(structure_name, mol2_dir, cleanup=False) # converting system and ligand pdbs to mol2s
        
        process_system(mol2_dir + structure_name, save_directory='./'+data_dir)
        # break
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        # print(e)
        raise e

def remake_deeppocket(dp_data_dir, orig_data_dir, new_data_dir, structure_name):
    orig_mol2_dir = f'{orig_data_dir}ready_to_parse_mol2/'
    mol2_dir = f'{new_data_dir}ready_to_parse_mol2/'
    output_path = f'{mol2_dir}{structure_name}'
    if not os.path.isdir(f'{output_path}'): os.makedirs(f'{output_path}')

    shutil.copyfile(f'{orig_mol2_dir}{structure_name}/system.mol2', f'{output_path}/system.mol2')
    shutil.copyfile(f'{orig_mol2_dir}{structure_name}/protein.mol2', f'{output_path}/protein.mol2')
    match_ligands(dp_data_dir, orig_mol2_dir, mol2_dir, structure_name)
    label_sites_given_ligands(f'{output_path}')

    process_system(mol2_dir + structure_name, save_directory=new_data_dir)


if __name__ == "__main__":   
    num_cores = 24
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
        Parallel(n_jobs=num_cores)(delayed(process_mlig)(full_df['path'][i], full_df['ligands'][i], data_dir='/benchmark_data_dir/holo4k', min_size=4) for i in tqdm(full_df.index))
    elif str(sys.argv[1]) == "coach420_dp" or str(sys.argv[1]) == "holo4k_dp":
        set_name = sys.argv[1][:-3]
        dp_data_dir =  f'{prepend}/benchmark_data_dir/dp_cavities/{set_name}/'
        orig_data_dir =  f'{prepend}/benchmark_data_dir/{set_name}/'
        new_data_dir = f'{prepend}/benchmark_data_dir/{set_name}_dp/'
        dp_systems = [system for system in sorted(os.listdir(dp_data_dir)) if os.path.isdir(dp_data_dir + system)]
        Parallel(n_jobs=num_cores)(delayed(remake_deeppocket)(dp_data_dir, orig_data_dir, new_data_dir, i) for i in tqdm(dp_systems))
    else:
        print("Expected first argument to be 'test', 'train', or 'train_hetro' instead got " + str(sys.argv[1]))
