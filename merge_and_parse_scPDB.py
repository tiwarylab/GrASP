import MDAnalysis as mda
from  MDAnalysis.analysis.rms import RMSD
import mdtraj
from mdtraj import shrake_rupley
from tqdm import tqdm
import glob
import os
import numpy as np
from itertools import combinations
from joblib  import Parallel, delayed
import networkx as nx
import csv

import MDAnalysis as mda
import MDAnalysis.analysis.rms

SASA_RATIO_CUTOFF = 0.685
RMSD_CUTOFF = 1e-4

ALLOWED_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'C', 'G', 'A', 'U', 'I', 'DC', 'DG', 'DA', 'DU', 'DT', 'DI']
ALLOWED_RESIUDES_SELECTION_STR = "".join(["resname " + x + " or " for x in list(ALLOWED_RESIDUES)[:-1]]) + "resname " + str(ALLOWED_RESIDUES[-1])

'''
/ = Done, ? = Question, - = TODO
/ Split rcsbID_*'s by RMSD on only carbon alphas
/ Merge all known ligands if they don't conflict. 
? What should we do if known ligands conflict? seems unlikely enough to just raise an exception? right now they're ignored
/ Merge unknown ligands with same resname if they don't conflict. If they do, they're ignored
/ Merge protein in with ligand
? Which protein file shold we use when building a final structure? Does it matter? I'm currently using the first: see line 93
/ Remove any ligands that don't fit our SASA ratio
/ Save file


- If there are multiple of the same known ligands with the same resname, use the highest SASA ratio for the dynamic cutoff

'''

def remove_salts(universe: mda.Universe, threshold=256):
    '''
    Removes any fragments from the universe that have 
    fewer atoms than the threshold. Fragments are 
    determined by bond conectivity. Returns an MDAnalysis.AtomGroup.
    
    Parameters
    ------------
    universe : MDAnalysis.Universe
        The MDAnalysis Universe whose's salts will be removed
    threshold : int, optional
        The minimum number of atoms in any fragment of the returned universe. 
        Default value is 256 which is larger than any ligand in the sc-PDB databse.
    '''
    to_return = []
    for fragment in universe.atoms.fragments:
        if len(fragment.atoms) >= threshold:
            to_return.append(fragment)
    return mda.Merge(*to_return)

def write_fragment(atom_group: mda.AtomGroup, parent_universe:mda.Universe, output_path: str):
    '''
    Writes an atom group to a mol2 with a specified output path.add()
    
    Merged atom groups cannot typically be written to mol2 files. This method 
    identifies the atoms in the parent protein by coordinates so that they can be written.add()
    
    Parameters
    ------------
    atom_group : MDAnalysis.AtomGroup
        The atom group that will be written to the output path.add()
    parent_universe : MDAnalysis.Universe
        The universe containing the atoms in atom_group
    output_path : str
        The path that the atom_group will be written to
    '''
    indices = []
    for atom in atom_group.atoms:
        x, y, z = atom.position
        sel = parent_universe.select_atoms("point {} {} {} 0.1".format(x, y, z))
        assert(len(sel) > 0), "Expected an atom at point {} {} {} 0.1 but it was not found!".format(x, y, z)
        assert(len(sel) == 1), "Two ligand atoms were found in nearly the same position!"
        indices.append(sel[0].index)
    sel_str = "".join(["index " + str(x) + " or " for x in indices[:-1]]) + "index " + str(indices[-1])
    parent_universe.select_atoms(sel_str).write(output_path)
    return None

def generate_structures(i, pdbID, directory, save_directory):
    # Get similarity between all pdbID_*.mol2 files
    similar_structure_paths =  sorted(list(glob.glob(os.path.join(directory, pdbID + '*')) ))
    similar_structures = {x:remove_salts(mda.Universe(x+'/protein.mol2', format='mol2')) for x in similar_structure_paths}
    try:
        similarity_dict = {(x,y):RMSD(similar_structures[x],similar_structures[y],select="name CA").run() if (len(similar_structures[x].select_atoms("name CA")) == len(similar_structures[y].select_atoms("name CA"))) else 999 for (x,y) in combinations(similar_structure_paths,2)}
    except AttributeError as e: 
        raise  AttributeError(str(i) + str(pdbID) + str(similar_structures))
        print(i)
        print(pdbID)
        print(similar_structures, flush=True)
        raise e
    except TypeError as e:
        raise  TypeError(str(i) + str(pdbID) + str(similar_structures))
        print(i)
        print(pdbID)
        print(similar_structures, flush=True)
        raise e
    # Create a graph where the maximal cliques have a RMSD similarity < RMSD_CUTOFF
    G = nx.Graph()
    G.add_nodes_from(similar_structure_paths)
    try:
        G.add_edges_from([k for k, v in similarity_dict.items() if type(v)!=int and v.results['rmsd'][0,2] < RMSD_CUTOFF])
    except Exception as e:
        raise TypeError (str(i) + str(pdbID) + str(similarity_dict.values()))
    components = nx.find_cliques(G) # returns a list of sets in which all paths to protein structures have 0 RMSD on the CA atoms. Files with no partner are returned as a set with one element 
    
    # Get all resnames from all known ligands
    #all_ligand_paths = [os.path.join(x,'ligand.mol2') for x in similar_structure_paths]
    #all_ligand_structures = [mda.Universe(x, format='mol2') for x in all_ligand_paths]
    # ligand_resnames = np.unique([resname[:3] for lig in all_ligand_structures for resname in lig.residues.resnames ])
    # ligand_sel_str = "".join(["resname " + x + " or " for x in list(ligand_resnames)[:-1]]) + "resname " + str(list(ligand_resnames)[-1])
    #del all_ligand_structures, all_ligand_paths
    
    for structure_idx, identical_structure_paths in enumerate(components):
        ligand_paths = [os.path.join(x,'ligand.mol2') for x in identical_structure_paths]
        protein_paths = [os.path.join(x,'protein.mol2') for x in identical_structure_paths]
        
        protein_structures       = [mda.Universe(x, format='mol2') for x in protein_paths]
        ligand_structures        = [mda.Universe(x, format='mol2') for x in ligand_paths]
        ligand_structures_no_H   = [x.select_atoms('not type H') for x in ligand_structures]
        ligand_len_list          = [len(x.atoms) for x in ligand_structures_no_H]
        
        # Create a universe with all of the ligands only
        ligand_universe = mda.Universe.empty(0)
        # Add all known ligands
        for prot_idx in range(len(protein_structures)):
            ligand = ligand_structures[prot_idx]
            # res_names = ligand.residues.resnames
            # new_resnames = [name[:3] for name in res_names]
            # ligand.residues.resnames = new_resnames
            
            # Check if ligand is already in universe or in it's place
            safe = True
            for atom in ligand.atoms:
                x, y, z = atom.position
                if len(ligand_universe.select_atoms("point {} {} {} 0.1".format(x, y, z))) != 0:
                    safe = False
            if safe: 
                if len(ligand_universe.atoms) == 0:
                    ligand_universe = ligand.atoms
                else:
                    ligand_universe = mda.Merge(ligand_universe.atoms, ligand.atoms)
                    
        # Add all unknown ligands
        for prot_idx in range(len(protein_structures)):
            protein = protein_structures[prot_idx]
            # res_names = protein.residues.resnames
            # new_resnames = [ name[:3] for name in res_names]
            # protein.residues.resnames = new_resnames

            # Iterate through fragments in each protein. If they match our criteria, we will add them to our ligand universe
            for fragment in protein.atoms.fragments:
                sel = fragment.select_atoms("not type H")
                if len(sel.atoms) in ligand_len_list:
                    if np.any([np.all(sel.atoms.types == lig.atoms.types) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H]):
                        # Types and sybyl types are the same and ordered correctly
                        # Check if ligand is already in universe or in it's place
                        safe = True
                        for atom in fragment:
                            x, y, z = atom.position
                            if len(ligand_universe.select_atoms("point {} {} {} 0.1".format(x, y, z))) != 0:
                                safe = False
                        if safe: ligand_universe = mda.Merge(ligand_universe.atoms, fragment.atoms)  
                    elif  np.any([np.all([x.split('.')[0] for x in sel.atoms.types] == [x.split('.')[0] for x in lig.atoms.types] ) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H]):
                        # Types are the same and ordered corrrectly
                        # Check if ligand is already in universe or in it's place
                        safe = True
                        for atom in fragment:
                            x, y, z = atom.position
                            if len(ligand_universe.select_atoms("point {} {} {} 0.1".format(x, y, z))) != 0:
                                safe = False
                        if safe: ligand_universe = mda.Merge(ligand_universe.atoms, fragment.atoms) 
                    elif np.any([np.all(sorted(sel.atoms.types) == sorted(lig.atoms.types)) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H]):
                        # Types and sybyl types are the same but misordered
                        # Check if ligand is already in universe or in it's place
                        safe = True
                        for atom in fragment:
                            x, y, z = atom.position
                            if len(ligand_universe.select_atoms("point {} {} {} 0.1".format(x, y, z))) != 0:
                                safe = False
                        if safe: ligand_universe = mda.Merge(ligand_universe.atoms, fragment.atoms) 
                    elif np.any([np.all(sorted([x.split('.')[0] for x in sel.atoms.types]) == sorted([x.split('.')[0] for x in lig.atoms.types])) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H]):
                        # Types are the same but misordered
                        # Check if ligand is already in universe or in it's place
                        safe = True
                        for atom in fragment:
                            x, y, z = atom.position
                            if len(ligand_universe.select_atoms("point {} {} {} 0.1".format(x, y, z))) != 0:
                                safe = False
                        if safe: ligand_universe = mda.Merge(ligand_universe.atoms, fragment.atoms) 
                    else:
                        # Nothing matches but the ligand lengths are the same
                        pass
                    
            
        ligand_universe.atoms.write("temp_ligand_{}.pdb".format(i))
        # Using the first protein in the universe because they're all guarenteed to be identical (in terms of RMSD on the CA atoms)
        first_protein = mda.Universe(protein_paths[0], format='mol2')
        
        first_protein = remove_salts(first_protein)       

        assert(len(first_protein.atoms) != 0)

        complete_universe = mda.Merge(first_protein.atoms, ligand_universe.atoms)
        complete_universe.atoms.write("temp_complex_{}.pdb".format(i))
        
        ligand_traj  = mdtraj.load("temp_ligand_{}.pdb".format(i))
        complete_traj = mdtraj.load("temp_complex_{}.pdb".format(i))
        
        known_ligand_SASA_ratios = []
        known_ligand_indices_ligand_universe = []  
        known_ligand_indices_complete_universe = []   
        
        try:
            ligand_SASA = shrake_rupley(ligand_traj, mode='atom')[0]
            complete_SASA = shrake_rupley(complete_traj, mode='atom')[0]
            
        except KeyError as e:
            print("MDTraj Key Error in " + str(identical_structure_paths))
            return 

        if not os.path.isdir(os.path.join(save_directory,"{}_{}".format(pdbID, structure_idx))): os.makedirs(os.path.join(save_directory,"{}_{}".format(pdbID, structure_idx)))

        # File that will contain info about this pdb id and its residues
        with open(os.path.join(save_directory,"{}_{}/about.csv".format(pdbID, structure_idx)),"w") as about_csv:
            csvwriter = csv.writer(about_csv) 
            csvwriter.writerow(['ligand_number','is_known','SASA_ratio','SASA_ratio_of_known_ligand','confidence_level']) 
            
            # Calculate indices of known ligand atoms in both the ligand only universe and the protein universe  
            ligand_index = 0
            for ligand in ligand_structures:
                this_ligands_indices_ligand_universe = []
                this_ligands_indices_complete_universe = []
                for atom in ligand.atoms:
                    x, y, z = atom.position
                    # This is a prime location that two atoms in the same place could really mess things up
                    ligand_atom_sel = ligand_universe.select_atoms("point {} {} {} 0.1".format(x, y, z))
                    assert(len(ligand_atom_sel) == 1), "Two ligand atoms were found in nearly the same position!"
                    known_ligand_indices_ligand_universe.append(ligand_atom_sel[0].index) 
                    this_ligands_indices_ligand_universe.append(ligand_atom_sel[0].index)
                    
                    known_index = complete_universe.select_atoms("point {} {} {} 0.1".format(x, y, z))[0].index
                    known_ligand_indices_complete_universe.append(known_index)
                    this_ligands_indices_complete_universe.append(known_index)
                SASA_ratio = np.mean(complete_SASA[this_ligands_indices_complete_universe])/np.mean(ligand_SASA[this_ligands_indices_ligand_universe])
                known_ligand_SASA_ratios.append(SASA_ratio)
                ligand.atoms.write(os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                csvwriter.writerow([ligand_index,True,SASA_ratio,SASA_ratio,5])
                ligand_index += 1
            
            
            # Find all ligand atoms that are not already labeled and have a sasa ratio above our cutoff
            delete_list = []    
            for fragment in complete_universe.atoms.fragments:
                sel = fragment.select_atoms("not type H")
                if len(sel.atoms) in ligand_len_list:
                    if np.any([np.all(sel.atoms.types == lig.atoms.types) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H]):
                        # Types and sybyl types are the same and ordered correctly
                        indices = list(set(fragment.indices) - set(known_ligand_indices_complete_universe))
                        if len(indices) > 0:   # Wont be greater than zero if this is a known ligand
                            matched_ligand_index = np.argmax([np.all(sel.atoms.types == lig.atoms.types) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H])
                            matched_known_SASA_ratio = known_ligand_SASA_ratios[matched_ligand_index]
                            this_ligand_traj = complete_traj.atom_slice(indices)
                            ligand_SASA = shrake_rupley(this_ligand_traj, mode='atom')[0]
                            SASA_ratio = np.mean(complete_SASA[indices])/np.mean(ligand_SASA)
                            
                            if SASA_ratio < SASA_RATIO_CUTOFF or SASA_ratio < matched_known_SASA_ratio + 0.1:
                                # fragment.atoms.write(os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                                write_fragment(fragment, mda.Universe(protein_paths[0], format='mol2'), os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                                csvwriter.writerow([ligand_index,False,SASA_ratio,matched_known_SASA_ratio,4])
                                ligand_index += 1
                    elif  np.any([np.all([x.split('.')[0] for x in sel.atoms.types] == [x.split('.')[0] for x in lig.atoms.types] ) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H]):
                        # Types and sybyl types are the same and ordered correctly
                        indices = list(set(fragment.indices) - set(known_ligand_indices_complete_universe))
                        if len(indices) > 0:   # Wont be greater than zero if this is a known ligand  
                            matched_ligand_index = np.argmax([np.all(sel.atoms.types == lig.atoms.types) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H])
                            matched_known_SASA_ratio = known_ligand_SASA_ratios[matched_ligand_index]
                            this_ligand_traj = complete_traj.atom_slice(indices)
                            ligand_SASA = shrake_rupley(this_ligand_traj, mode='atom')[0]
                            SASA_ratio = np.mean(complete_SASA[indices])/np.mean(ligand_SASA)
                            
                            if SASA_ratio < SASA_RATIO_CUTOFF or SASA_ratio < matched_known_SASA_ratio + 0.1:
                                # fragment.atoms.write(os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                                write_fragment(fragment, mda.Universe(protein_paths[0], format='mol2'), os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                                csvwriter.writerow([ligand_index,False,SASA_ratio,matched_known_SASA_ratio,3])
                                ligand_index += 1
                    elif np.any([np.all(sorted(sel.atoms.types) == sorted(lig.atoms.types)) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H]):    
                        # Types and sybyl types are the same and ordered correctly
                        indices = list(set(fragment.indices) - set(known_ligand_indices_complete_universe))
                        if len(indices) > 0:   # Wont be greater than zero if this is a known ligand  
                            matched_ligand_index = np.argmax([np.all(sel.atoms.types == lig.atoms.types) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H])
                            matched_known_SASA_ratio = known_ligand_SASA_ratios[matched_ligand_index]
                            this_ligand_traj = complete_traj.atom_slice(indices)
                            ligand_SASA = shrake_rupley(this_ligand_traj, mode='atom')[0]
                            SASA_ratio = np.mean(complete_SASA[indices])/np.mean(ligand_SASA)
                            
                            if SASA_ratio < SASA_RATIO_CUTOFF or SASA_ratio < matched_known_SASA_ratio + 0.1:
                                # fragment.atoms.write(os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                                write_fragment(fragment, mda.Universe(protein_paths[0], format='mol2'), os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                                csvwriter.writerow([ligand_index,False,SASA_ratio,matched_known_SASA_ratio,2])
                                ligand_index += 1
                    elif np.any([np.all(sorted([x.split('.')[0] for x in sel.atoms.types]) == sorted([x.split('.')[0] for x in lig.atoms.types])) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H]):
                        # Types and sybyl types are the same and ordered correctly
                        indices = list(set(fragment.indices) - set(known_ligand_indices_complete_universe))
                        if len(indices) > 0:   # Wont be greater than zero if this is a known ligand  
                            matched_ligand_index = np.argmax([np.all(sel.atoms.types == lig.atoms.types) if len(sel.atoms) == len(lig.atoms) else False for lig in ligand_structures_no_H])
                            matched_known_SASA_ratio = known_ligand_SASA_ratios[matched_ligand_index]
                            this_ligand_traj = complete_traj.atom_slice(indices)
                            ligand_SASA = shrake_rupley(this_ligand_traj, mode='atom')[0]
                            SASA_ratio = np.mean(complete_SASA[indices])/np.mean(ligand_SASA)
                            
                            if SASA_ratio < SASA_RATIO_CUTOFF or SASA_ratio < matched_known_SASA_ratio + 0.1:
                                # fragment.atoms.write(os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                                write_fragment(fragment, mda.Universe(protein_paths[0], format='mol2'), os.path.join(save_directory,"{}_{}/ligand_{}.mol2".format(pdbID, structure_idx, ligand_index)))
                                csvwriter.writerow([ligand_index,False,SASA_ratio,matched_known_SASA_ratio,1])
                                ligand_index += 1
                        
            about_csv.close()         
            
        os.remove("temp_ligand_{}.pdb".format(i))
        os.remove("temp_complex_{}.pdb".format(i))
        
        # delete_str = "".join(["not index " + x + " or " for x in list(delete_list)[:-1]]) + "not index " + str(list(delete_list)[-1])
        # complete_universe = complete_universe.select_atoms(delete_str)
        # It turns out the you can't write merged proteins as mol2 files so we're going to have to write an uncleaned protein and clean it up later
        protein_to_output = mda.Universe(protein_paths[0], format='mol2')
        assert len(protein_to_output.atoms) > 0
        protein_to_output.atoms.write(os.path.join(save_directory,"{}_{}/protein.mol2".format(pdbID, structure_idx)))
        
def main():
    directory = './scPDB_data_dir/unprocessed_scPDB_mol2/'
    save_directory = './scPDB_data_dir/unprocessed_mol2'
    num_cores = 1#24
    pdbID_i_list = os.listdir(directory)
    pdbID_list = np.unique(sorted([x[:4] for x in pdbID_i_list]))    
    
    # groups = init_groups(directory)
    # groups = combine_pdbs(directory, groups)

    try:
        Parallel(n_jobs=num_cores)(delayed(generate_structures)(i, pdbID, directory, save_directory) for i, pdbID in enumerate(tqdm(pdbID_list[:]))) 
    finally:
        for i in range(len(directory)): # directory is a little more than we need but it's the only way to do this cleanly
            if os.path.isfile("temp_ligand_{}.mol2".format(i)): os.remove("temp_ligand_{}.mol2".format(i))
            if os.path.isfile("temp_complex_{}.mol2".format(i)): os.remove("temp_complex_{}.mol2".format(i))

if __name__=="__main__":
    main()


# import numpy as np

# import os
# import re
# import sys
# def ca_rmsd(file1, file2):
#     selection_str = ALLOWED_RESIUDES_SELECTION_STR
    
#     u1 = mda.Universe(file1)
#     u2 = mda.Universe(file2)
    
#     res_names = u1.residues.resnames
#     new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
#     u1.residues.resnames = new_names
#     res_names = u2.residues.resnames
#     new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
#     u2.residues.resnames = new_names
    
#     if len(u1.select_atoms(selection_str + ' and name CA')) == len(u2.select_atoms(selection_str + ' and name CA')):
#         R = mda.analysis.rms.RMSD(u1, u2, selection_str + ' and name CA')
#         R.run()
#         rmsd = R.results['rmsd'][0,2]
#     else:
#         rmsd = 9999
    
#     return rmsd
# def init_groups(path):
#     subdirs = [f for f in os.listdir(path)]
#     pdbs = [s.split('_')[0] for s in subdirs]
#     pdbs = np.unique(pdbs).tolist()
            
#     groups = {}
#     for pdb in pdbs:
#         groups[pdb] = []
#     for s in subdirs:
#         groups[s.split('_')[0]].append([s])
        
#     return groups
# def combine_groups(path, pdb_list):
#     for i in range(len(pdb_list)-1):
#         for j in range(i+1, len(pdb_list)):
#             group1 = pdb_list[i]
#             group2 = pdb_list[j]
#             if len(group1) > 0 and len(group2) > 0:
#                 rmsd = ca_rmsd(f'{path}/{group1[0]}/protein.mol2', f'{path}/{group2[0]}/protein.mol2')
#                 if rmsd < 1e-4:
#                     pdb_list[i].extend(group2)
#                     pdb_list[j] = []
#     pdb_list = [group for group in pdb_list if group] # removing empty lists
#     return pdb_list
# def combine_pdbs(path, groups):
#     for pdb in groups.keys():
#         pdb_list = groups[pdb]
#         if len(pdb_list) > 1:
#             pdb_list = combine_groups(path, pdb_list)
#             groups[pdb] = pdb_list
#     return groups
# if __name__ == "__main__":
#     path = './scPDB_data_dir/unprocessed_scPDB_mol2'
#     groups = init_groups(path)
#     groups = combine_pdbs(path, groups)
#     np.save('pdb_groups.npy', groups)
