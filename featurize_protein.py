import pyximport; pyximport.install()

def process_system(path_to_protein_mol2_files, save_directory='./data_dir', parse_ligands=True):
    """Process a protein-ligand complex in MOL2 format.

    This function processes a protein-ligand complex in MOL2 format. It calculates various features for each atom
    and saves the processed data in the specified output directory.

    Parameters
    ----------
    path_to_protein_mol2_files : str
        The path to the directory containing the protein-ligand complex files in MOL2 format.

    save_directory : str, optional
        The directory where the processed data will be saved, by default './data_dir'.

    parse_ligands : bool, optional
        If True, calculate the identity and distance to the closest ligand for each atom, by default True.
    """
    
    # Some of the packages don't pickle nicely and therefore don't work with joblib.Parallel. So, we import them inside the function
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

    import warnings

    # Adding the filter to suppress the mdtraj warning
    warnings.filterwarnings("ignore", category=UserWarning, message="top= kwargs ignored since this file parser does not support it")

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
        'un':[0,0,0,0,1,0]  # Unkown Bond Type
    }
    
    selection_str = " or ".join([f'resname {x}' for x in list(residue_dict.keys())])
    feature_factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    
    # Load File in mdanalysis and rdkit 
    path_to_files = path_to_protein_mol2_files
    structure_name = path_to_protein_mol2_files.split('/')[-1]
    
    protein_w_H = mda.Universe(path_to_files + '/protein.mol2', format='mol2')
    rdkit_protein_w_H = Chem.MolFromMol2File(path_to_files + '/protein.mol2', removeHs = False, sanitize=False, cleanupSubstructures=False)

    res_names = protein_w_H.residues.resnames
    new_names = [ "".join(re.findall("[a-zA-Z]+", name)).upper() for name in res_names]
    protein_w_H.residues.resnames = new_names
    
    protein_w_H = protein_w_H.select_atoms(selection_str)
    
    # Calculate solvent accessible surface area for each atom, this needs to be done before hydrogens are dropped
    try:
        traj = mdtrajload(path_to_files + '/protein.mol2')
        SAS = shrake_rupley(traj, mode='atom')
        if len(SAS) > 1:
            # Sanity Check. This should never happen but the API is unclear
            raise Exception("Did not expect more than one list of SAS values")   
        SAS = SAS[0]
    except KeyError as e:
        print("Value not included in dictionary \"{}\" while calculating SASA {}.".format(e, path_to_files))
        return

    # Add SAS from hydrogen to bonded atom, create number of bonded hydrogens feature
    num_bonded_H = np.zeros(traj.n_atoms)
    for atom in protein_w_H:
        is_bonded_to_H = [bond.element == 'H' for _, bond in atom.bonds]
        num_bonded_H[atom.index] = sum(is_bonded_to_H)
        # Because the bonds have the old ids, we use our map to the new ids to access the SAS value, if the value is -1
        # it means that the bonded atom no longer exists in our universe (i.e., it was dropped). If this happens it will
        # be a very rare occasion as must things other than solvents are not droppped.
        local_SAS = np.array([SAS[atom_idx[1]]  for atom_idx in atom.bonds.indices])    
        SAS[atom.index] += np.sum(local_SAS * is_bonded_to_H)       # Only take the values from hydrogens
    
    # Drop Hydrogens
    protein_w_H.ids = np.arange(0, len(protein_w_H.atoms))
    protein = protein_w_H.select_atoms("not element H")
    protein.ids = np.arange(0, len(protein.atoms))              # Reindex atoms ids to make them zero-indexed and contiguous

    # Compute sparse adjacency matrix with a threshold of 10, leaves room to be trimmed down later
    trimmed = scipy.sparse.lil_matrix((len(protein.atoms.positions), len(protein.atoms.positions)), dtype='float')
    get_distance_matrix(protein.atoms.positions, trimmed, 10)
    trimmed = trimmed.tocsr()

    feature_array = []  # Will contain all of the features for a given molecule
    SASA_array = []

    # Calculate hydrogen donor, acceptor, hydrophobe, and lumped_hydrophobe properties
    try:
        rdkit_protein_w_H.UpdatePropertyCache(strict=False)
        Chem.rdmolops.SetHybridization(rdkit_protein_w_H)
    except Exception as e:
        print("Failed to update property cache while processing", path_to_files)
        return
    acceptor_indices          = [x.GetAtomIds()[0] for x in feature_factory.GetFeaturesForMol(rdkit_protein_w_H, includeOnly="Acceptor")]
    donor_indices             = [x.GetAtomIds()[0] for x in feature_factory.GetFeaturesForMol(rdkit_protein_w_H, includeOnly="Donor")]
    hydrophobe_indices        = [x.GetAtomIds()[0] for x in feature_factory.GetFeaturesForMol(rdkit_protein_w_H, includeOnly="Hydrophobe")]
    lumped_hydrophobe_indices = [x.GetAtomIds()[0] for x in feature_factory.GetFeaturesForMol(rdkit_protein_w_H, includeOnly="LumpedHydrophobe")]


    # Generate a feature vector for each atom
    bins = np.arange(0,11)
    for atom in protein.atoms:                                              # Iterate through atoms and create vectors of features for each atom
        name = "".join(re.findall("^[a-zA-Z]+", atom.resname)).upper()      # Clean resname 
        element = atom.element.upper()
        try:
            # radial density function (rdf) calculation where dr = 1 and r_max = 10
            d = trimmed[np.where(protein.ids == atom.id)[0][0]]
            n, bins = np.histogram(d[d>0], bins =bins)
            r = bins[1:]                                            # using the exterior radius of the shell
            vol = (4/3) * np.pi * r**3
            g = n.cumsum() / vol                                    # this is now density not g(r)
            g = g[1:]                                               # skip the bin from 0 to 1, there shouldn't be any entries

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

            # WARNING: any change to SAS's index must be reflected in infer_test_set.py

            feature_array.append(np.concatenate((residue_dict[name], atom_dict[element], g, [SAS[atom.index]], formal_charge, num_bonds_w_heavy_atoms, is_in_ring, is_aromatic, mass, hybridization, acceptor, donor, hydrophobe, lumped_hydrophobe)))
            SASA_array.append(SAS[atom.index])
        except Exception as e:
            print("Error while feautrizing atom {} for file {}.{}".format(atom.id, path_to_files,e), flush=True)
            return -2


    if trimmed.shape[0] != len(feature_array):  # Sanity Check
        raise ValueError ("Adjacency matrix shape ({}) did not match feature array shape ({}). {}".format(np.array(trimmed).shape, np.array(feature_array).shape, structure_name))

    # Optionally calculate identity of closest ligand for each atom
    if parse_ligands:
        lig_coord_list = []
        for file_path in sorted(glob(f'{path_to_files}/*')):
            if 'ligand' in file_path.split('/')[-1] and not 'site' in file_path.split('/')[-1]:
                lig_univ = mda.Universe(file_path)
                lig_coord_list.append(lig_univ.atoms.positions)

        prot_coords = protein.positions
        all_lig_coords = np.row_stack(lig_coord_list)
        distance_to_ligand = np.min(distance_array(prot_coords, all_lig_coords), axis=1)
        closest_ligand_atom = np.argmin(distance_array(prot_coords, all_lig_coords), axis=1)
        atom_to_ligand_map = np.concatenate([i*np.ones(lig_coord_list[i].shape[0]) for i in range(len(lig_coord_list))])
        closest_ligand = atom_to_ligand_map[closest_ligand_atom]

    else:
        prot_coords = protein.positions
        distance_to_ligand = 99 * np.ones(len(protein.atoms))
        closest_ligand = -1 * np.ones(len(protein.atoms))

    # Creating edge_attributes dictionary. Only holds bond types, weights are stored in trimmed
    edge_attributes = {tuple(bond.atoms.ids):{"bond_type":bond_type_dict[bond.order]} for bond in protein.bonds}

    np.savez_compressed(save_directory + '/raw/' + structure_name, adj_matrix = trimmed,
                        feature_matrix = feature_array, ligand_distance_array = distance_to_ligand,
                        coords = prot_coords, closest_ligand = closest_ligand, 
                        edge_attributes = edge_attributes, SASA_array = SASA_array)
    protein.atoms.write(save_directory + '/mol2/' + str(structure_name) +'.mol2')

    return None
