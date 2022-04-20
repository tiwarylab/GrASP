from dataclasses import field
from turtle import end_fill
from parsing_regular import process_system
from parsing_trimmed import process_trimmed
import MDAnalysis as mda
import os
import sys
import shutil
from openbabel import openbabel
from tqdm import tqdm
from glob import glob
import re

allowed_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'C', 'G', 'A', 'U', 'I', 'DC', 'DG', 'DA', 'DU', 'DT', 'DI']
selection_str = "".join(["resname " + x + " or " for x in list(allowed_residues)[:-1]]) + "resname " + str(allowed_residues[-1])


def label_protein_site(bound_structure, structure_name, out_directory):
    output_path = out_directory + '/ready_for_mol2_conversion/' + structure_name
    if not os.path.isdir(output_path): os.makedirs(output_path)
    # print(bound_structure)
    protein = mda.Universe(bound_structure,format='PDB')
    bound_structure = protein.select_atoms("protein")
    bound_structure.select_atoms(selection_str)
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
    for idx, segment in enumerate(ligand_structure.segments):
        segment.atoms.write(output_path + '/ligand_{}.pdb'.format(idx))
        site_resid_list = []
        for atom in ligand_structure.segments[idx].atoms:
            x,y,z = atom.position
            site_resid_list += (list(bound_structure.select_atoms("point {} {} {} 6.5".format(x, y, z)).resids))
            
        site_resid_list = list(set(site_resid_list))
            
        site_selection_str = "".join(["resid " + str(x) + " or " for x in site_resid_list[:-1]] + ["resid " + str(site_resid_list[-1])])

        site = bound_structure.select_atoms(site_selection_str)
        
        # resid_list = segment.residues.resids
        # seg_selection_str = "".join(["resid " + str(x) + " or " for x in resid_list[:-1]] + ["resid " + str(resid_list[-1])])
        # site = bound_structure.select_atoms(seg_selection_str)
        site.write(output_path + '/site_for_ligand_{}.pdb'.format(idx))
    return None

def label_sites_given_ligands(path_to_mol2):
    protein = mda.Universe(os.path.join(path_to_mol2, 'protein.mol2'))
    protein_no_h = protein.select_atoms("not type H")
    ligand_idx = 0
    all_site_resids = []
    for file_path in sorted(glob(path_to_mol2+ '/*')):
        if 'protein' in file_path:
            # This is the main structure, we already have it
            pass
        elif 'ligand' and not 'site' in file_path:
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
            this_ligands_site.atoms.write(os.path.join(path_to_mol2,"site_for_ligand_{}.mol2".format(ligand_idx)))
            ligand_idx += 1
        else:
            # This is an unexpected file
            pass
    site_resid_list = list(set(all_site_resids))
    
    site_selection_str = "".join(["resid " + str(x) + " or " for x in site_resid_list[:-1]] + ["resid " + str(site_resid_list[-1])])
    protein.select_atoms(site_selection_str).atoms.write(os.path.join(path_to_mol2,"site.mol2"))

def pdb2mol2(pdb_file, structure_name, out_directory, addH=True, out_name='protein', cleanup=True):
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
    
    if cleanup:
        # Use MDA to remove clean file
        univ = mda.Universe(output_mol2_path)
        res_names = univ.residues.resnames
        new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
        univ.residues.resnames = new_names
        univ = univ.select_atoms(selection_str)
        mda.coordinates.MOL2.MOL2Writer(output_mol2_path).write(univ)

    
# def mol22mol2(infile, structure_name, out_directory, addH=True, out_name="protein"):
#     output_path = out_directory + '/unprocessed_mol2/' + structure_name
#     if not os.path.isdir(output_path): os.makedirs(output_path)
    
#     obConversion = openbabel.OBConversion()
#     obConversion.SetInAndOutFormats("mol2", "mol2")

#     mol = openbabel.OBMol()
#     obConversion.ReadFile(mol, infile)
#     mol.DeleteHydrogens()
#     if addH:
#         mol.CorrectForPH()
#         mol.AddHydrogens()
#     obConversion.WriteFile(mol, output_mol2_path)

def rebond_mol2(i,infile, structure_name, outfile, addH=False):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol2", "pdb")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, infile)
    mol.DeleteHydrogens()
    mol.StripSalts(256)
    obConversion.WriteFile(mol, 'temp{}.pdb'.format(i))
    pdb2mol2('temp{}.pdb'.format(i), structure_name, outfile, addH=addH)

    # Delete mol2 file
    os.remove('temp{}.pdb'.format(i))
    return None
 

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
        
def process_train_openbabel(i, file, output_dir):
    try:
        prepend = os.getcwd()
        structure_name = file
        if not os.path.isdir(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name)): 
            os.makedirs(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name))
        # print(os.path.join(prepend, '/scPDB_data_dir/unprocessed_mol2/',file,'/protein.mol2'))
        # print(prepend+'/scPDB_data_dir/unprocessed_mol2/'+file+'/protein.mol2')
        rebond_mol2(i,prepend+'/scPDB_data_dir/unprocessed_mol2/'+file+'/protein.mol2', structure_name, prepend+'/'+output_dir+"/ready_to_parse_mol2/",addH=True)
        
        for file_path in glob(prepend + '/scPDB_data_dir/unprocessed_mol2/' + file + '/*'):
            if 'ligand' in file_path:
                shutil.copyfile(file_path, prepend+'/'+ output_dir+"/ready_to_parse_mol2/"+file+'/'+file_path.split('/')[-1])
        
        label_sites_given_ligands(prepend + '/scPDB_data_dir/ready_to_parse_mol2/' + file)
        
        process_system((prepend + '/' + output_dir + '/ready_to_parse_mol2/' + structure_name), save_directory='./' + output_dir)
        
    except AssertionError as e: 
        print("Failed to find ligand in", file)
    except Exception as e:
        print("ERROR", file, e)
        
def process_train_classic(i, structure_name, output_dir, unprocessed_dir = 'unprocessed_scPDB_mol2'):
    # print("Processing", structure_name, flush=True)
    try:
        process_system(os.path.join('./',output_dir, unprocessed_dir, structure_name), save_directory='./' + output_dir)
    except AssertionError as e: 
        print("Failed to find ligand in", structure_name)
    except Exception as e:
        print(e)
        
def process_train_trimmed(i, structure_name, output_dir):
    # print("Processing", structure_name, flush=True)
    try:
        process_trimmed('./' + output_dir + '/unprocessed_scPDB_mol2/' + structure_name, save_directory='./' + output_dir)
    except AssertionError as e: 
        print("Failed to find ligand in", structure_name)
    except Exception as e:
        # print(e)
        raise(e)
    
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

def process_val(file, data_dir="benchmark_data_dir"):
    try:
        prepend = os.getcwd()
        structure_name = file.split('_')[1][:-4]
        label_protein_site(prepend + '/'+data_dir+'/unprocessed_pdb/' + file, structure_name, out_directory=prepend +'/benchmark_data_dir')
        path_to_pdb = prepend +'/'+data_dir+'/ready_for_mol2_conversion/'+structure_name +"/"
        for file_name in os.listdir(path_to_pdb):
            if 'protein' not in file_name:
                # Do not add hydrogens to sites, they will not be used for labeling and moreover will  mess up comparison between 'ground truth' and predictions
                pdb2mol2(path_to_pdb+file_name, structure_name,prepend+'/'+data_dir+'',addH=False, out_name=file_name.split('/')[-1][:-4], cleanup=False) 
            else:
                pdb2mol2(path_to_pdb+file_name, structure_name,prepend+'/'+data_dir+'', out_name='protein')
        # print("processing system")
        process_system('./'+data_dir+'/unprocessed_mol2/' + structure_name, save_directory='./'+data_dir)
        # break
    except AssertionError as e:
        print("Failed to find ligand in", file)
    except Exception as e:  
        print(e)

if __name__ == "__main__":   
    num_cores = 128
    prepend = os.getcwd()
    from joblib.externals.loky import set_loky_pickler
    from joblib import Parallel, delayed
 
    if str(sys.argv[1]) == "val":
        pdb_files = [filename for filename in sorted(list(os.listdir(prepend +'/benchmark_data_dir/unprocessed_pdb')))]
        Parallel(n_jobs=num_cores)(delayed(process_val)(filename) for _, filename in enumerate(tqdm(pdb_files)))
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
        print("Parsing the standard train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/data_dir/unprocessed_scPDB_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_trimmed)(i, filename, 'regular_data_dir') for i, filename in enumerate(tqdm(mol2_files[4000:])))               # POINTS TO SAME DIR AS OPEN BABEL TO SAVE SPACE BE CAREFUL
        # for i, filename in enumerate(mol2_files[1800+360+380+250:]):
        #     process_train(i,filename, 'regular_data_dir')
    elif str(sys.argv[1]) == "train_hetro":
        print("Parsing the heterogeneous train set")
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/hetro_data_dir/unprocessed_scPDB_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_classic)(i, filename, 'hetro_data_dir') for i, filename in enumerate(tqdm(mol2_files[:]))) 
        # for i, filename in enumerate(mol2_files[:]):
        #     process_train(i,filename, 'hetro_data_dir')
    elif str(sys.argv[1]) == "coach420":
        pdb_files = [filename for filename in sorted(list(os.listdir(prepend +'/coach420_data_dir/unprocessed_pdb')))]
        Parallel(n_jobs=num_cores)(delayed(process_val)(filename) for _, filename in enumerate(tqdm(pdb_files)))
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
