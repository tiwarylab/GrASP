from parsing_atoms_w_atom_feats_parallel_new_feats import process_system
import MDAnalysis as mda
import os
import openbabel

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

# def pdb2mol2(protein_pdb_file, site_pdb_file, structure_name, out_directory, addH=True):
#     output_path = out_directory + '/unprocessed_mol2/' + structure_name
#     if not os.path.isdir(output_path): os.makedirs(output_path)
    
#     obConversion = openbabel.OBConversion()
#     obConversion.SetInAndOutFormats("pdb", "mol2")
    
#     protein_mol2_file_path  = output_path + '/protein.mol2'
#     site_mol2_file_path = output_path + '/site.mol2'
    
#     mol = openbabel.OBMol()
#     if addH: mol.AddHydrogens()
#     obConversion.ReadFile(mol, protein_pdb_file)
#     obConversion.WriteFile(mol, protein_mol2_file_path)
    
#     obConversion.ReadFile(mol, site_pdb_file)
#     obConversion.WriteFile(mol, site_mol2_file_path)
#     return None

def pdb2mol2(pdb_file, structure_name, out_directory, addH=True):
    # print("Starting")
    output_path = out_directory + '/unprocessed_mol2/' + structure_name
    if not os.path.isdir(output_path): os.makedirs(output_path)
    
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "mol2")
    
    output_mol2_path  = output_path + '/' + pdb_file.split('/')[-1][:-4] +'.mol2'
    
    mol = openbabel.OBMol()
    
    obConversion.ReadFile(mol, pdb_file)
    # print( mol.NumAtoms())
    if addH: mol.AddHydrogens()
    # print( mol.NumAtoms())
    
    obConversion.WriteFile(mol, output_mol2_path)
    # print("Done")
    return None

if __name__ == "__main__":
    prepend = os.getcwd()
    pdb_files = [filename for filename in sorted(list(os.listdir(prepend +'/benchmark_data_dir/unprocessed_pdb')))]
    for file in pdb_files:
        try:
            structure_name = file.split('_')[1][:-4]
            label_protein_site(prepend + '/benchmark_data_dir/unprocessed_pdb/' + file, structure_name, out_directory=prepend +'/benchmark_data_dir')
            path_to_pdb = prepend +'/benchmark_data_dir/ready_for_mol2_conversion/'+structure_name +"/"
            for file_name in os.listdir(path_to_pdb):
                if 'protein' not in file_name:
                    # Do not add hydrogens to sites, they will not be used for labeling and moreover will  mess up comparison between 'ground truth' and predictions
                    pdb2mol2(path_to_pdb+file_name, structure_name,prepend+'/benchmark_data_dir',addH=False) 
                else:
                    pdb2mol2(path_to_pdb+file_name, structure_name,prepend+'/benchmark_data_dir')
            # print("processing system")
            process_system('./benchmark_data_dir/unprocessed_mol2/' + structure_name, save_directory='./benchmark_data_dir')
            # break
        except AssertionError as e:
            print("Failed to find ligand in", file)
        except Exception as e:
            print(e)
            continue

# if __name__ == "__main__":
#     prepend = os.getcwd()
#     pdb_files = [filename for filename in sorted(list(os.listdir(prepend +'/benchmark_data_dir/unprocessed_pdb')))]
#     for file in pdb_files:
#         try:
#             structure_name = file.split('_')[1][:-4]
#             label_protein_site(prepend + '/benchmark_data_dir/unprocessed_pdb/' + file, structure_name, out_directory=prepend +'/benchmark_data_dir')
#             path_to_pdb = prepend +'/benchmark_data_dir/ready_for_mol2_conversion/'+structure_name
#             pdb2mol2(path_to_pdb+'/protein.pdb', path_to_pdb+'/site.pdb',structure_name,prepend+'/benchmark_data_dir')
#         except AssertionError as e:
#             print("Failed to find ligand in", file)
#         except Exception as e:
#             raise e
#             continue
        
    
    # Example file names:
    # - e.003.001.001_1fswa_unbound.pdb
    # - g.050.001.002_2vnfa.pdb
    
    # Get 'code' for the path, use it to find the path
    
    # Send structure path and ligand path to pdb2mol
    
    # Get and save site.mol2 from label_protein_site
    
    # Pass data to parsing code

    # 1. Convert pdb files to mol2 files                                        |   Done
    # 2. Read mol2 for bound structure                                          |   Done
    # 3. Read mol2 for ligand                                                   |   Done
    # 4. Label site in bound stucture                                           |   Done
    # 5. Read mol2 for unbound structure                                            Deferred
    # 6. Align unbound structure to bound structure based on site                   Deferred
    # 7. Label bound structure based on proximity to ligand (see scPDB paper)       Done
