from MDAnalysis.analysis import distances
import re
import pandas as pd
import MDAnalysis as mda
import numpy as np
import pandas as pd
import glob as glob
import os
# from IPython.display import clear_output
import scipy
from fast_distance_computation import get_distance_matrix

# imports for bond features
from rdkit import Chem
from mdtraj import shrake_rupley#, baker_hubbard, kabsch_sander, wernet_nilsson
from mdtraj import load as mdtrajload
import mdtraj as md
from collections import defaultdict
# from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def process_system(filename):
    warnings.filterwarnings("ignore")

    # Adjacency Matrix
    path_to_files = './scPDB_raw_data/' + filename
    # filename_lst = [filename for filename in os.listdir(path_to_files) if 'site' in filename or 'protein' in filename]
    
    protein_w_H = mda.Universe(path_to_files + '/protein.mol2', format='mol2')
    # protein_w_H.ids = np.arange(0,len(protein_w_H.atoms))       # I'll be abusing the id attributes as by own settable index
    # print(protein_w_H.ids)
    # n_atoms = len(protein_w_H.atoms)
    # the_ids = np.arange(0,len(protein_w_H.atoms)) 
    # set_ids = protein_w_H.ids

    res_names = protein_w_H.residues.resnames
    new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
    protein_w_H.residues.resnames = new_names
    
    protein_w_H = protein_w_H.select_atoms(selection_str)
    # if np.max(protein_w_H.ids) >= n_atoms: 
    #     raise Exception
    indices_not_dropped = protein_w_H.indices

    # Calculate SAS for each atom, this needs to be done before hydrogens are dropped
    try:
        traj = mdtrajload(path_to_files + '/protein.mol2')
        SAS = shrake_rupley(traj, mode='atom')
        if len(SAS) > 1:
            # Sanity check, I'm pretty sure this should never happen
            raise Exception("Did not expect more than one list of SAS values")   
        # SAS_org = SAS
        SAS = SAS[0]
    except KeyError as e:
        print("Value not included in dictionary \"{}\" while parsing {}.".format(e, path_to_files))
        failed_list.append([path_to_files, "Value not included in dictionary \"{}\" while parsing {}.".format(e, path_to_files)])
        return

    mapping = defaultdict(lambda: -1)                           # A mapping from atom indices to position in SAS
    for i in range(len(protein_w_H.atoms)):
        mapping[protein_w_H.atoms.indices[i]] = i

    # if np.max(protein_w_H.ids) >= n_atoms: 
    #     raise Exception
    # original_ids = protein_w_H.ids
    # Keeping track of the ids is getting weird so I'm going to make a comment about it:
    # Every time we select atoms, we will need to know which atoms are saved and which get dropped
    # In this case, original_ids stores what atoms are left over after applying our selection string.
    # We'll use this to mask the mdtraj SAS output so we have the same atoms from each. We'll use it 
    # again later when we drop hydrogens
    # if len(SAS) > len(temp):
    #     raise Exception ("HOW")
    # try:
    #     SAS = [SAS[mapping[i]] for i in indices_not_dropped]                        # Sync SAS atoms with mda atoms
    # except Exception as e:
    #     # print("Max of org ids:", np.max(the_ids))
    #     # print("Max of ids:", np.max(set_ids))
    #     print("Index Error at:", filename)
    #     print("Number of atoms read by mdtraj:", traj.n_atoms)
    #     # print("Number of atoms read by MDA:", n_atoms)
    #     raise e

    # protein_w_H.ids = np.arange(0,len(protein_w_H.atoms))       # Reset ids to contiguous values

    # Add SAS from hydrogen to bonded atom, create number of bonded hydrogens feature
    num_bonded_H = np.zeros(traj.n_atoms)
    for atom in protein_w_H:
        is_bonded_to_H = [re.search("^[a-zA-Z]+", bond.type).group().upper() == 'H' for _, bond in atom.bonds]
        num_bonded_H[atom.index] = sum(is_bonded_to_H)
        # Because the bonds have the old ids, we use our map to the new ids to access the SAS value, if the value is -1
        # it means that the bonded atom no longer exists in our universe (i.e., it was dropped). If this happens it will
        # be a very rare occasion as must things other than solvents are not droppped.
        local_SAS = np.array([SAS[atom_idx[1]]  for atom_idx in atom.bonds.indices])    
        SAS[atom.index] = np.sum(local_SAS * is_bonded_to_H)       # Only take the values from hydrogens

    # Drop Hydrogens
    protein_w_H.ids = np.arange(0, len(protein_w_H.atoms))
    protein = protein_w_H.select_atoms("not type H")
    protein.ids = np.arange(0, len(protein.atoms))              # Abusing atoms ids to make them zero-indexed which makes our life easier

    
    # original_ids = protein.ids                                  # Save the ids so we know what got dropped
    # protein.ids = np.arange(0,len(protein.atoms))               # Reset ids to contiguous values
    # SAS = [SAS[i] for i in protein.atoms.indices]                        # Remove Hydrogens From SAS
    # num_bonded_H = [num_bonded_H[i] for i in protein.atoms.indices]      # Remove hydroges from count of bonded hydrogens to each atom

    trimmed = scipy.sparse.lil_matrix((len(protein.atoms.positions), len(protein.atoms.positions)), dtype='float')
    get_distance_matrix(protein.atoms.positions, trimmed, 7)
    trimmed = trimmed.tocsr()

    # Feature Matrix
    feature_array = []  # This will contain all of the features for a given molecule
    flag = False

    bins = np.arange(0,6)
    pi_4 = 4 * np.pi
    for atom in protein.atoms:                                                              # Iterate through residues and create vectors of features
        name = "".join(re.findall("^[a-zA-Z]+", atom.resname)).upper()                      # Remove numbers from the name string
        element = re.search("^[a-zA-Z]+", atom.type).group().upper()
        try:
            # rdf calculation where dr = 1 and r_max = 5
            d = trimmed[np.where(protein.ids == atom.id)[0][0]]
            n, bins = np.histogram(d[d>0], bins =bins)
            r = bins[1:]
            g = n/(pi_4 * r ** 2)
            # Add feature vector with [residue level feats] [atom level feats] [rdf] [SAS] [num bonded hydrogens per atom]
            feature_array.append(np.concatenate((residue_properties.iloc[resname_dict[name]].values, atom_properties.iloc[atom_dict[element]].values, g, [SAS[atom.index]], [num_bonded_H[atom.index]])))                              # Add corresponding features to feature array
        except Exception as e:
            print("Value not included in dictionary \"{}\" while parsing {}.".format(name, path_to_files))
            failed_list.append([path_to_files, "Value not included in dictionary \"{}\" while parsing {}.".format(name, path_to_files)])
            flag = True
            return
            # raise ValueError ("Value not included in dictionary \"{}\" while parsing {}.".format(name, path_to_files)) from e

    if flag:
        return                                                                                   # This flag will skip creating the class matrix and will prevent the file from being saved

    if trimmed.shape[0] != len(feature_array):  # Sanity Check
        raise ValueError ("Adjacency matrix shape ({}) did not match feature array shape ({}). {}".format(np.array(trimmed).shape, np.array(feature_array).shape, filename))

    # Classes
    site = mda.Universe(path_to_files + '/site.mol2', format='mol2')
    res_names = site.residues.resnames
    new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
    site.residues.resnames = new_names
    site = site.select_atoms(selection_str).select_atoms("not type H")

    site.ids = np.arange(0,len(site.atoms))

    binding_site_lst = []

    # Sometimes the site atoms are not found in the protein. This happens when they're a part of a nonstandard residues.
    # For now, we'll skip them and keep a count of how many there are
    try:
        for atom in site.atoms:
            x, y, z = atom.position                                                                       # Get the coordinates of each atom of the binding site
            binding_site_lst.append(protein.select_atoms("point {} {} {} 0".format(x, y, z))[0].id)      # Select that atom in the whole protein
    except Exception as e:
        print("Binding site atom not found in filtered protein. Was it dropped? \n Filename:", str(path_to_files))
        # Okay in theory this is bad and could cause a race condition but it's very unlikely because this case is rarely reached.
        # In conjunction with the print statement I'm really not worried about it.
        # Okay, after reading about GIL I think it's fine... 
        global nonstandard_residue_skip 
        nonstandard_residue_skip += 1 
        return

    if binding_site_lst == []:
        print(site.atoms)
        raise Exception("Binding Site Not Found")

    mask = np.ones(len(protein.atoms), bool)                                                         # Inverse of the atoms in the binding site
    mask[binding_site_lst] = False

    classes = np.zeros((len(protein.atoms),2))
    classes[binding_site_lst, 1] = 1                                                                    # Set all atoms in the binding site to class 2
    classes[mask, 0] = 1                                                                                # Set all atoms not in the binding site to class 1

    # Creating edge_attributes dictionary. Only holds bond types, weights are stored in trimmed
    edge_attributes = {tuple(bond.atoms.ids):{"bond_type":bond_type_dict[bond.order]} for bond in protein.bonds}

    np.savez_compressed('./data_atoms_w_atom_feats/'+ filename, adj_matrix = trimmed, feature_matrix = feature_array, class_array = classes, edge_attributes = edge_attributes)
    protein.atoms.write('./mol2_atoms_w_atom_feats/'+ str(filename) +'.mol2',)


residue_properties = pd.read_csv('./Amino_Acid_Properties_for_Atoms.csv')                                       # A csv containing properties of amino acids and a on-hot encoding of their name
atom_properties = pd.read_csv('./Atom_Properties_for_Atoms.csv')
resname_dict = {'ALA':0, 'ARG':1, 'ASN':2, 'ASP':3, 'CYS':4, 'GLN':5,                                           # A dictionary mapping all of the residue names to an entry in the property dataframe
                'GLU':6, 'GLY':7, 'HIS':8, "ILE":9, "LEU":10, 
                "LYS":11, "MET":12, "PHE":13, "PRO":14, "SER":15, 
                "THR":16, "TRP":17, "TYR":18, "VAL":19, "C":20, "G":21, 
                "A":22, "U":23, "I":24, "DC":25, "DG":26, "DA":27,
                "DU":28, "DT":29, "DI":30}

atom_dict = {'C':0,'N':1,'O':2,'S':3,'H':4,'MG':5,'Z':6,'MN':7,'CA':8,'FE':9,'P':10, 'CL':11, 'F':12, 'I':13, 'Br':14}

bond_type_dict = {'1':[1,0,0,0,0],'2':[0,1,0,0,0],'3':[0,0,1,0,0],'ar':[0,0,0,1,0],'am':[0,0,0,0,1],'un':[0,0,0,0,0]}

selection_str = "".join(["resname " + x + " or " for x in list(resname_dict.keys())[:-1]]) + "resname " + str(list(resname_dict.keys())[-1])
# print(selection_str)
# total_files = len(os.listdir('./scPDB_raw_data'))
index = 1
failed_list = []
nonstandard_residue_skip = 0

# num_cores = multiprocessing.cpu_count()

inputs = [filename for filename in sorted(list(os.listdir('./scPDB_raw_data')))]

if not os.path.isdir('./data_atoms_w_atom_feats'):
    os.makedirs('./data_atoms_w_atom_feats')

try:
    ##########################################
    # Comment me out to run just one file
    num_cores = 8
    if __name__ == "__main__":
        Parallel(n_jobs=num_cores)(delayed(process_system)(i,) for i in tqdm(inputs[:]))

    np.savez('./failed_list', np.array(failed_list))
    ##########################################

    ##########################################
    # Uncomment me to run just one file
    # process_system('1iep_1') 
    ##########################################
finally:
    if nonstandard_residue_skip > 0:
        print("Warning: Number of files skipped due to a nonstandard residue being a part of the site:", nonstandard_residue_skip)
