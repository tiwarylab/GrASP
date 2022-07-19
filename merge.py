from numbers import Rational
import MDAnalysis as mda
from  MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.distances import distance_array
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
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


SASA_RATIO_CUTOFF = 0.3
RMSD_CUTOFF = 1e-4

def remove_salts(universe: mda.Universe, threshold=229):
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
    for fragment in universe.atoms.fragments:   # Note that because of the way fragments are written in MDA, this will ignore any selections
        if len(fragment.atoms) >= threshold:
            to_return.append(fragment)
    return mda.Merge(*to_return)

def write_fragment(atom_group: mda.AtomGroup, parent_universe:mda.Universe, output_path: str, check_overlap=True):
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
        if check_overlap: assert(len(sel) == 1), "Two ligand atoms were found in nearly the same position!"
        indices.append(sel[0].index)
    sel_str = "".join(["index " + str(x) + " or " for x in indices[:-1]]) + "index " + str(indices[-1])
    parent_universe.select_atoms(sel_str).write(output_path)
    return None

def get_SASA_ratio(i:int,protein_structure_file:str, ligand_structure_file:str):
    temp_system_name = "temp_system_{}.pdb".format(i)
    
    protein_univ = mda.Universe(protein_structure_file, format='mol2')
    ligand_univ  = mda.Universe(ligand_structure_file, format='mol2')
    
    system_univ  = mda.Merge(remove_salts(protein_univ).atoms, ligand_univ.atoms)
    system_univ.atoms.write(temp_system_name)
    
    ligand_indices = [system_univ.select_atoms("point {} {} {} 0.1".format(x, y, z))[0].index for x, y, z in ligand_univ.atoms.positions]
    
    system_traj = mdtraj.load(temp_system_name)
    ligand_traj = mdtraj.load(ligand_structure_file)
    system_SASA =  np.sum(np.array(shrake_rupley(system_traj, mode='atom')[0])[ligand_indices])
    ligand_SASA =  np.sum(shrake_rupley(ligand_traj,mode='atom')[0])
    
    os.remove(temp_system_name)
    return system_SASA/ligand_SASA

def get_SASA_ratio(i:int,protein_univ:mda.AtomGroup, ligand_univ:mda.AtomGroup):
    temp_system_name = "temp_system_{}.pdb".format(i)
    temp_ligand_name = "temp_ligand_{}.pdb".format(i)
    
    system_univ  = mda.Merge(remove_salts(protein_univ).atoms, ligand_univ.atoms)
    system_univ.atoms.write(temp_system_name)
    ligand_univ.atoms.write(temp_ligand_name)
    
    ligand_indices = [system_univ.select_atoms("point {} {} {} 0.1".format(x, y, z))[0].index for x, y, z in ligand_univ.atoms.positions]
    
    system_traj = mdtraj.load(temp_system_name)
    ligand_traj = mdtraj.load(temp_ligand_name)
    system_SASA =  np.sum(np.array(shrake_rupley(system_traj, mode='atom')[0])[ligand_indices])
    ligand_SASA =  np.sum(shrake_rupley(ligand_traj,mode='atom')[0])
    
    os.remove(temp_system_name)
    os.remove(temp_ligand_name)
    return system_SASA/ligand_SASA

class pdbID():
    def __init__(self, index, pdbID, directory):
        self.pdbID      = pdbID
        self.directory  = directory
        self.index      = index
        
        similar_protein_paths =  sorted(list(glob.glob(os.path.join(directory, pdbID + '*')) ))
        similar_structures = {x:remove_salts(mda.Universe(x+'/protein.mol2', format='mol2')) for x in similar_protein_paths}
        
        similarity_dict = {}
        for (x,y) in combinations(similar_protein_paths,2):
            R = RMSD(similar_structures[x],similar_structures[y],select="name CA") if (len(similar_structures[x].select_atoms("name CA")) == len(similar_structures[y].select_atoms("name CA"))) else 999
            if R != 999: R.run()
            similarity_dict[(x,y)] = R
            
        # Create a graph where the maximal cliques have a RMSD similarity < RMSD_CUTOFF
        G = nx.Graph()
        G.add_nodes_from(similar_protein_paths)
        G.add_edges_from([k for k, v in similarity_dict.items() if type(v)!=int and v.results['rmsd'][0,2] < RMSD_CUTOFF])
        components = nx.find_cliques(G)

        self.merge_groups = [Protein_Structure(index, paths, self) for index, paths in enumerate(components)]
        for structure in self.merge_groups: structure.find_unknown_ligands()
        
    def write(self, directory:str):
        for structure in self.merge_groups: structure.write(directory + "/" + self.pdbID + "_" + str(structure.index))
        
        
class Protein_Structure():
    def __init__(self,index, paths, parent_pdbID):

        self.index = index
        self.paths = paths
        self.parent_pdbID = parent_pdbID
        
        # Since RMSD ~= 0, we will pick the first protein to represent all proteins in this universe
        self.mda_universe          = mda.Universe(self.paths[0] + '/protein.mol2')
        self.stripped_mda_universe = remove_salts(self.mda_universe)
        
        self.known_ligands = []
        for path in paths:
            if len(self.known_ligands) == 0:
                self.known_ligands.append(Ligand(len(self.known_ligands),mda.Universe(path + '/ligand.mol2'), True, 5, self))
            else:
                lig = Ligand(len(self.known_ligands), mda.Universe(path + '/ligand.mol2'), True, 5, self)
                # Check if ligand being added overlaps with any other ligands added
                if not np.any([lig.overlaps(other) for other in self.known_ligands]):
                    self.known_ligands.append(lig)
    
    def find_unknown_ligands(self):      
        self.unknown_ligands = []
        possible_lens = [len(lig.mda_universe.atoms) for group in self.parent_pdbID.merge_groups for lig in group.known_ligands]
        for fragment in self.mda_universe.atoms.fragments:
            if len(fragment.atoms) not in possible_lens:
                continue
            temp_lig =  Ligand(len(self.unknown_ligands)+len(self.known_ligands), fragment, False, 0, self)
            top_confidence_level = 0
            top_confidence_associated_SASA = 0
            for group in self.parent_pdbID.merge_groups:  # Must check all ligands associated with the same pdb id 
                confidence_levels = [temp_lig.compare_to(lig) for lig in group.known_ligands]
                matched_ligand_index = np.argmax(confidence_levels)
                if confidence_levels[matched_ligand_index] > top_confidence_level:
                    top_confidence_level = confidence_levels[matched_ligand_index]
                    top_confidence_associated_SASA = group.known_ligands[matched_ligand_index].SASA_ratio
            temp_lig.confidence_level = top_confidence_level
            if top_confidence_level:
                if np.all([not temp_lig.overlaps(lig) for lig in self.known_ligands]) and np.all([not temp_lig.overlaps(lig) for lig in self.unknown_ligands]):
                    if temp_lig.SASA_ratio < SASA_RATIO_CUTOFF or temp_lig.SASA_ratio < top_confidence_associated_SASA + 0.1:
                        self.unknown_ligands.append(temp_lig)         

    def write(self, directory:str):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        with open(directory + "/about.csv","w") as about_csv:
            csvwriter = csv.writer(about_csv) 
            csvwriter.writerow(['ligand_number','is_known','SASA_ratio','SASA_ratio_of_known_ligand','confidence_level']) 
            for lig in self.known_ligands:      lig.write(directory, csvwriter)
            for lig in self.unknown_ligands:    lig.write(directory, csvwriter)
               
        # We would like to save the stripped universe, but MDA can't do that right now 
        self.mda_universe.atoms.write(directory + "/protein.mol2")
        
class Ligand():
    def __init__(self,index:int,mda_universe, is_known:bool, confidence_level:int, parent_protein:Protein_Structure):
        self.index              = index
        self.parent_protein     = parent_protein        
        self.mda_universe       = mda_universe
        self.is_known           = is_known
        self.confidence_level   = confidence_level
        
        self.num_atoms = len(self.mda_universe.atoms)
        self.num_heavy_atoms = len(self.mda_universe.select_atoms("not type H"))
        
        unique_id = self.parent_protein.parent_pdbID.index * 10000 + self.parent_protein.index * 100 + self.index
        
        self.SASA_ratio = get_SASA_ratio(unique_id, self.parent_protein.stripped_mda_universe, self.mda_universe)
        
    def compare_to(self, other):
        if type(other) is not Ligand:
            return 0
        if self.num_heavy_atoms != other.num_heavy_atoms:
            return 0
        this_no_H = self.mda_universe.select_atoms("not type H")
        other_no_H = other.mda_universe.select_atoms("not type H")
        
        if np.all(this_no_H.atoms.types == other_no_H.atoms.types):
            return 4
        if np.all([x.split('.')[0] for x in this_no_H.atoms.types] == [x.split('.')[0] for x in other_no_H.atoms.types] ):
            return 3
        if np.all(sorted(this_no_H.atoms.types) == sorted(other_no_H.atoms.types)):
            return 2
        if np.all(sorted([x.split('.')[0] for x in this_no_H.atoms.types]) == sorted([x.split('.')[0] for x in other_no_H.atoms.types])):
            return 1
        return 0
        
    def overlaps(self, other):
        dist_arr = distance_array(self.mda_universe.atoms.positions,
                                                        other.mda_universe.atoms.positions,
                                                        backend='openMP')
        return np.any(dist_arr < 1e-4)

    def write(self, directory:str, csvwriter):
        csvwriter.writerow([self.index, self.is_known, self.SASA_ratio, self.confidence_level])
        self.mda_universe.atoms.write(directory + "/ligand_{}.mol2".format(self.index))

def generate_structures(index, id, directory, save_directory):
    structures = pdbID(index, id, directory)
    structures.write(save_directory)
    
def main():
    directory       = './scPDB_data_dir/unprocessed_scPDB_mol2/'
    save_directory  = './scPDB_data_dir/unprocessed_mol2'
    num_cores = 24
    
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)
    
    pdbID_i_list = os.listdir(directory)
    pdbID_list = np.unique(sorted([x[:4] for x in pdbID_i_list]))    
    
    try:
        Parallel(n_jobs=num_cores)(delayed(generate_structures)(i, id, directory, save_directory) for i, id in enumerate(tqdm(pdbID_list[:])))
    finally:
        for i in range(len(directory)): # directory is a little more than we need but it's the only way to do this cleanly
            if os.path.isfile("temp_ligand_{}.mol2".format(i)): os.remove("temp_ligand_{}.mol2".format(i))
            if os.path.isfile("temp_complex_{}.mol2".format(i)): os.remove("temp_complex_{}.mol2".format(i))

        
if __name__=="__main__":
    main()
