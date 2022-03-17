from parsing_atoms_w_atom_feats_parallel_new_feats import process_system
import MDAnalysis as mda
import os
import sys
import shutil
import openbabel
from tqdm import tqdm

def label_protein_site(bound_structure, structure_name, out_directory):
    output_path = out_directory + '/ready_for_mol2_conversion/' + structure_name
    if not os.path.isdir(output_path): os.makedirs(output_path)
    # print(bound_structure)
    protein = mda.Universe(bound_structure,format='PDB')
    bound_structure = protein.select_atoms("protein")
    ligand_structure = protein.select_atoms("not protein")
    
    assert len(bound_structure.atoms) > 0
    assert len(ligand_structure.atoms) > 0
    assert len(bound_structure.select_atoms('type H').atoms) == 0
    assert len(ligand_structure.select_atoms('type H').atoms) == 0
        
    # Get residues that are considered a site
    site_resid_list = []
    for atom in ligand_structure.atoms:
        x,y,z = atom.position
        site_resid_list += (list(bound_structure.select_atoms("point {} {} {} 6.5".format(x, y, z)).resids))
        
    site_resid_list = list(set(site_resid_list))
        
    site_selection_str = "".join(["resid " + str(x) + " or " for x in site_resid_list[:-1]] + ["resid " + str(site_resid_list[-1])])

    bound_site = bound_structure.select_atoms(site_selection_str)

    # Write bound_site to mol2 
    bound_structure.write(output_path + '/protein.pdb')
    bound_site.write(output_path + '/site.pdb')

    # Write bound site of reach ligand to mol2
    for idx, segment in enumerate(bound_site.segments):
        resid_list = segment.residues.resids
        seg_selection_str = "".join(["resid " + str(x) + " or " for x in resid_list[:-1]] + ["resid " + str(resid_list[-1])])
        site = bound_structure.select_atoms(seg_selection_str)
        site.write(output_path + '/site_for_ligand_{}.pdb'.format(idx))
    return None

def pdb2mol2(pdb_file, structure_name, out_directory, addH=True, out_name='protein'):
    # print("Starting")
    output_path = out_directory + '/unprocessed_mol2/' + structure_name
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
    return None

def rebond_mol2(i, infile, structure_name, outfile, addH=False):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol2", "pdb")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, infile)
    mol.DeleteHydrogens()
    obConversion.WriteFile(mol, 'temp{}.pdb'.format(i))
    pdb2mol2('temp{}.pdb'.format(i), structure_name, outfile, addH=addH)
    os.remove('temp{}.pdb'.format(i))
    return None
 

def process_train(i, file):
    print("Processing", file, flush=True)
    try:
        prepend = os.getcwd()
        structure_name = file
        #label_protein_site(prepend + '/data_dir/unprocessed_mol2/' + file, structure_name, out_directory=prepend +'/data_dir')
        path_to_pdb = prepend +'/data_dir/unprocessed_scPDB_mol2/'+structure_name +"/"
        # for file_name in os.listdir(path_to_pdb):
        #     # if 'protein' not in file_name:
        #     #     # Do not add hydrogens to sites, they will not be used for labeling and moreover will  mess up comparison between 'ground truth' and predictions
        #     #     #pdb2mol2(path_to_pdb+file_name, structure_name,prepend+'/data_dir',addH=False) 
        #     # else:
        rebond_mol2(i,path_to_pdb+'protein.mol2',structure_name,prepend+'/data_dir', addH=True)
        if not os.path.isdir(prepend+'/data_dir/unprocessed_mol2/'+structure_name): 
            os.makedirs(prepend+'/data_dir/unprocessed_mol2/'+structure_name)
        shutil.copyfile(path_to_pdb+'site.mol2', prepend+'/data_dir/unprocessed_mol2/'+structure_name+'/site.mol2')
        # print("processing system")
        process_system('./data_dir/unprocessed_mol2/' + structure_name, save_directory='./data_dir')
        # break
    except AssertionError as e: 
        print("Failed to find ligand in", file)
    except Exception as e:
        print(e)
    finally:
        # I'm not sure what is happening, it seems like OpenBabel is killing the process if there is an issue in the mol2 files. 
        # I'm going to try this in order to see if we can preven it from killing all of the processes 
        return None
        
# import threading, time, random
# def process_train_helper(i, file):
#     thread = threading.Thread(target=process_train, args=(i, file))
#     thread.start()
#     thread.join()

def process_val(file):
    try:
        prepend = os.getcwd()
        structure_name = file.split('_')[1][:-4]
        label_protein_site(prepend + '/benchmark_data_dir/unprocessed_pdb/' + file, structure_name, out_directory=prepend +'/benchmark_data_dir')
        path_to_pdb = prepend +'/benchmark_data_dir/ready_for_mol2_conversion/'+structure_name +"/"
        for file_name in os.listdir(path_to_pdb):
            if 'protein' not in file_name:
                # Do not add hydrogens to sites, they will not be used for labeling and moreover will  mess up comparison between 'ground truth' and predictions
                pdb2mol2(path_to_pdb+file_name, structure_name,prepend+'/benchmark_data_dir',addH=False, out_name=file_name.split('/')[-1][:-4]) 
            else:
                pdb2mol2(path_to_pdb+file_name, structure_name,prepend+'/benchmark_data_dir', out_name='protein')
        # print("processing system")
        process_system('./benchmark_data_dir/unprocessed_mol2/' + structure_name, save_directory='./benchmark_data_dir')
        # break
    except AssertionError as e:
        print("Failed to find ligand in", file)
    except Exception as e:
        print(e)

if __name__ == "__main__":   
    num_cores = 1
    prepend = os.getcwd()
    from joblib.externals.loky import set_loky_pickler
    from joblib import Parallel, delayed

    # set_loky_pickler("dill")
    ######
    ######  TODO: RESET THE CORES GENIUS
    ######
 
    if str(sys.argv[1]) == "test":
        pdb_files = [filename for filename in sorted(list(os.listdir(prepend +'/benchmark_data_dir/unprocessed_pdb')))]
        Parallel(n_jobs=num_cores)(delayed(process_train)(filename) for filename in enumerate(pdb_files))
            
    elif str(sys.argv[1]) == "train":
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/data_dir/unprocessed_scPDB_mol2')))]
        # Parallel(n_jobs=1)(delayed(process_train)(i, filename) for i, filename in enumerate(tqdm(mol2_files[1800+360+380+250:]))) 
        for i, filename in enumerate(mol2_files[1800+360+380+250:]):
            process_train(i,filename)
    else:
        print("Expected first argument to be 'test' or 'train' instead got " + str(sys.argv[1]))
