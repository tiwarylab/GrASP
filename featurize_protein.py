import os
import numpy as np
import pyximport; pyximport.install()


def process_system(path_to_protein_mol2_files, save_directory='./data_dir'):
    # I really wish I didn't have to do this but the way some of these paackages pickle I have no other way. If you know a better alternative feel free to reach out
    import re
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
    from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
    import numpy as np
    from pathlib import Path
    import scipy
    from glob import glob
    from fast_distance_computation import get_distance_matrix

    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig

    from mdtraj import shrake_rupley
    from mdtraj import load as mdtrajload
    from collections import defaultdict

    # import warnings
    # warnings.filterwarnings("ignore") 

    #                     [One hot encoding of residue name     polar Y/N     Acidic,Basic,Neutral  Pos/Neg/Neutral Charge]
    residue_dict = {'ALA':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   0,1,    0,     0,     1,      0,  0,  1], 
                    'ARG':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   1,0,    0,     1,     0,      1,  0,  0], 
                    'ASN':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   1,0,    0,     0,     1,      0,  0,  1], 
                    'ASP':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   1,0,    1,     0,     0,      0,  1,  0], 
                    'CYS':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   1,0,    1,     0,     0,      0,  0,  1], 
                    'GLN':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   1,0,    0,     0,     1,      0,  0,  1],                                           
                    'GLU':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,   0,0,    1,     0,     0,      0,  1,  0], 
                    'GLY':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,   0,1,    0,     0,     1,      0,  0,  1], 
                    'HIS':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,   1,0,    0,     1,     0,      0,  0,  1], 
                    "ILE":[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,   0,1,    0,     0,     1,      0,  0,  1], 
                    "LEU":[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,   0,1,    0,     0,     1,      0,  0,  1], 
                    "LYS":[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,   1,0,    0,     1,     0,      1,  0,  0], 
                    "MET":[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,   0,1,    0,     0,     1,      0,  0,  1], 
                    "PHE":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,   0,1,    0,     0,     1,      0,  0,  1], 
                    "PRO":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,   0,1,    0,     0,     1,      0,  0,  1], 
                    "SER":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,   1,0,    0,     0,     1,      0,  0,  1], 
                    "THR":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,   1,0,    0,     0,     1,      0,  0,  1], 
                    "TRP":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,   0,1,    0,     0,     1,      0,  0,  1], 
                    "TYR":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,   1,0,    1,     0,     0,      0,  0,  1], 
                    "VAL":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,   0,1,    0,     0,     1,      0,  0,  1],
    }
    hybridization_dict = {
        "SP2":        [1,0],
        "SP3":        [0,1], 
    }

    atom_dict = {'C': [1,0,0,0],
                'N': [0,1,0,0],
                'O': [0,0,1,0],
                'S': [0,0,0,1],
    }

    # Leaving an extra bit to denote self loops
    bond_type_dict = {
        '1': [1,0,0,0,0,0],
        '2': [0,1,0,0,0,0],
        'ar':[0,0,1,0,0,0],
        'am':[0,0,0,1,0,0],
        'un':[0,0,0,0,1,0]    # Unknown bond order is set to null/unbonded edges
    }
    selection_str = "".join(["resname " + x + " or " for x in list(residue_dict.keys())[:-1]]) + "resname " + str(list(residue_dict.keys())[-1])
    feature_factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    
    # Adjacency Matrix
    path_to_files = path_to_protein_mol2_files
    structure_name = path_to_protein_mol2_files.split('/')[-1]
    try:
        protein_w_H = mda.Universe(path_to_files + '/protein.mol2', format='mol2')
        rdkit_protein_w_H = Chem.MolFromMol2File(path_to_files + '/protein.mol2', removeHs = False, sanitize=False, cleanupSubstructures=False)
        # rdkit_protein_w_H = Chem.MolFromMol2File(path_to_files + '/protein.mol2', removeHs = False)

    except Exception as e: 
        raise e
        print("Failed to compute charges for the following file due to a structure error. This file will be skipped:", path_to_files + '/protein.mol2', flush=True)
        return
    res_names = protein_w_H.residues.resnames
    new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
    protein_w_H.residues.resnames = new_names
    
    protein_w_H = protein_w_H.select_atoms(selection_str)
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
        print("Value not included in dictionary \"{}\" while calculating SASA {}.".format(e, path_to_files))
        # failed_list.append([path_to_files, "Value not included in dictionary \"{}\" while calculating SASA {}.".format(e, path_to_files)])
        return

    # Add SAS from hydrogen to bonded atom, create number of bonded hydrogens feature
    num_bonded_H = np.zeros(traj.n_atoms)
    for atom in protein_w_H:
        is_bonded_to_H = [re.search("^[a-zA-Z]+", bond.type).group().upper() == 'H' for _, bond in atom.bonds]
        num_bonded_H[atom.index] = sum(is_bonded_to_H)
        # Because the bonds have the old ids, we use our map to the new ids to access the SAS value, if the value is -1
        # it means that the bonded atom no longer exists in our universe (i.e., it was dropped). If this happens it will
        # be a very rare occasion as must things other than solvents are not droppped.
        local_SAS = np.array([SAS[atom_idx[1]]  for atom_idx in atom.bonds.indices])    
        SAS[atom.index] += np.sum(local_SAS * is_bonded_to_H)       # Only take the values from hydrogens
    # Drop Hydrogens
    protein_w_H.ids = np.arange(0, len(protein_w_H.atoms))
    protein = protein_w_H.select_atoms("not type H")
    protein.ids = np.arange(0, len(protein.atoms))              # Abusing atoms ids to make them zero-indexed which makes our life easier

    trimmed = scipy.sparse.lil_matrix((len(protein.atoms.positions), len(protein.atoms.positions)), dtype='float')
    get_distance_matrix(protein.atoms.positions, trimmed, 10)
    trimmed = trimmed.tocsr()

    # Feature Matrix
    feature_array = []  # This will contain all of the features for a given molecule

    try:
        rdkit_protein_w_H.UpdatePropertyCache(strict=False)
        Chem.rdmolops.SetHybridization(rdkit_protein_w_H)
    except Exception as e:
        print("Failed to update property cache while processing", path_to_files)
        return
    acceptor_indices = [x.GetAtomIds()[0] for x in feature_factory.GetFeaturesForMol(rdkit_protein_w_H, includeOnly="Acceptor")]
    donor_indices = [x.GetAtomIds()[0] for x in feature_factory.GetFeaturesForMol(rdkit_protein_w_H, includeOnly="Donor")]
    hydrophobe_indices = [x.GetAtomIds()[0] for x in feature_factory.GetFeaturesForMol(rdkit_protein_w_H, includeOnly="Hydrophobe")]                # Seems to be the slow one, comparitively
    lumped_hydrophobe_indices = [x.GetAtomIds()[0] for x in feature_factory.GetFeaturesForMol(rdkit_protein_w_H, includeOnly="LumpedHydrophobe")]


    bins = np.arange(0,11)
    pi_4 = 4 * np.pi
    for atom in protein.atoms:                                                              # Iterate through residues and create vectors of features
        name = "".join(re.findall("^[a-zA-Z]+", atom.resname)).upper()                      # Remove numbers from the name string
        element = re.search("^[a-zA-Z]+", atom.type).group().upper()
        try:
            # rdf calculation where dr = 1 and r_max = 10
            d = trimmed[np.where(protein.ids == atom.id)[0][0]]
            n, bins = np.histogram(d[d>0], bins =bins)
            r = bins[1:] # using the exterior radius of the shell
            g = n/(pi_4 * r ** 2)
            g = g[1:] # skip the bin from 0 to 1, there shouldn't be any entries

            # RDKit Features
            rdkit_atom = rdkit_protein_w_H.GetAtomWithIdx(int(atom.index))
            
            num_bonds_w_heavy_atoms = [rdkit_atom.GetTotalDegree() - rdkit_atom.GetTotalNumHs(includeNeighbors=True)]
            formal_charge = [rdkit_atom.GetFormalCharge()]
            is_in_ring = [1,0] if rdkit_atom.IsInRing() else [0,1]
            is_aromatic = [1,0] if rdkit_atom.GetIsAromatic() else [0,1]
            mass = [rdkit_atom.GetMass()]
            hybridization = hybridization_dict[str(rdkit_atom.GetHybridization())]
            
            acceptor = [1,0] if atom.index in acceptor_indices else [0,1] 
            donor = [1,0] if atom.index in donor_indices else [0,1]
            hydrophobe = [1,0] if atom.index in hydrophobe_indices else [0,1]
            lumped_hydrophobe = [1,0] if atom.index in lumped_hydrophobe_indices else [0,1]

            if name == 'MET' and element == 'SE': element = 'S' # for featurizing selenomethionine

            assert not np.any(np.isnan(num_bonds_w_heavy_atoms))
            assert not np.any(np.isnan(formal_charge))
            assert not np.any(np.isnan(is_in_ring))
            assert not np.any(np.isnan(is_aromatic))
            assert not np.any(np.isnan(mass))
            assert not np.any(np.isnan(hybridization))
            assert not np.any(np.isnan(acceptor))
            assert not np.any(np.isnan(donor))
            assert not np.any(np.isnan(hydrophobe)) 
            assert not np.any(np.isnan(lumped_hydrophobe))

            # Warning, any change to SAS's index must be reflected in infer_test_set.py
            # Add feature vector with                  0-27               28-31        32-40       41               42                  43               44-45        46-47      48      49-50        51-52    53-54    55-56     57-58 (59  is degree)
            feature_array.append(np.concatenate((residue_dict[name], atom_dict[element], g, [SAS[atom.index]], formal_charge, num_bonds_w_heavy_atoms, is_in_ring, is_aromatic, mass, hybridization, acceptor, donor, hydrophobe, lumped_hydrophobe)))  #,formal_charge     25                       # Add corresponding features to feature array
        except Exception as e:
            print("Error while feautrizing atom for file {}.{}".format(path_to_files,e), flush=True)
            return -2
            # raise ValueError ("Value not included in dictionary \"{}\" while generating feature vector for {}.".format(name, path_to_files)) from e


    if trimmed.shape[0] != len(feature_array):  # Sanity Check
        raise ValueError ("Adjacency matrix shape ({}) did not match feature array shape ({}). {}".format(np.array(trimmed).shape, np.array(feature_array).shape, structure_name))

    # Classes
    lig_coord_list = []
    for file_path in glob(f'{path_to_files}/*'):
        if 'ligand' in file_path and not 'site' in file_path:
            lig_univ = mda.Universe(file_path)
            lig_coord_list.append(lig_univ.select_atoms('not type H').positions)

    prot_coords = protein.positions
    all_lig_coords = np.row_stack(lig_coord_list)
    distance_to_ligand = np.min(distance_array(prot_coords, all_lig_coords), axis=1)

    # Creating edge_attributes dictionary. Only holds bond types, weights are stored in trimmed
    edge_attributes = {tuple(bond.atoms.ids):{"bond_type":bond_type_dict[bond.order]} for bond in protein.bonds}

    np.savez_compressed(save_directory + '/raw/' + structure_name, adj_matrix = trimmed, feature_matrix = feature_array, ligand_distance_array = distance_to_ligand, edge_attributes = edge_attributes)
    protein.atoms.write(save_directory + '/mol2/' + str(structure_name) +'.mol2')

    return None


# Should run when file is called but not imported
# if __name__ == "__main__":
#     from joblib import Parallel, delayed
#     from tqdm import tqdm
    
#     # print(selection_str)
#     # total_files = len(os.listdir('./scPDB_raw_data'))
#     # index = 1 # I have no idea what this was for, got I hope we don't need it
#     # failed_list = []

#     inputs = ['./scPDB_raw_data' + struct_name for struct_name in sorted(list(os.listdir('./scPDB_raw_data')))]

#     if not os.path.isdir('./data_dir'):
#         os.makedirs('./data_dir')
#     ##########################################
#     # Comment me out to run just one file
#     num_cores = 24
    
#     from joblib.externals.loky import set_loky_pickler
#     set_loky_pickler("dill")
    
#     r = Parallel(n_jobs=num_cores)(delayed(process_system)(x, save_directory='./data_dir') for x in tqdm(inputs[:]))
#     # Parallel(n_jobs=2)(delayed(process_system)(x) for x in ['1iep_1','3eky_1'])

#     # np.savez('./failed_list', np.array(failed_list))
    ##########################################

    ##########################################
    # Uncomment me to run just one file
    import time
    # print("Starting")
    # start = time.time()
    # process_system('1iep_1',residue_dict,hybridization_dict,atom_dict,bond_type_dict,selection_str) 
    # print("Finished. Total Time:", str(time.time() - start)) 
    ##########################################
# finally:
#     res, i = zip(*r)
#     if res.count(-1) > 0:
#         print("Warning: Number of files skipped due to a nonstandard residue being a part of the site:", res.count(-1))

if __name__ == "_main__":
    structure_name = '1ds7_2'
    process_system('./data_dir/unprocessed_mol2/' + structure_name, save_directory='./data_dir')
