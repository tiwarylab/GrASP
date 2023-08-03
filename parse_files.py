from featurize_protein import process_system
import MDAnalysis as mda
from MDA_fix.MOL2Parser import MOL2Parser # fix added in MDA development build
import os
import numpy as np
import pandas as pd
import networkx as nx
import shutil
import openbabel
from tqdm import tqdm
from glob import glob
import re
import argparse

ALLOWED_RESIUDES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
ALLOWED_ELEMENTS = ['C', 'N', 'O', 'S']
RES_SELECTION_STR = " or ".join([f'resname {x}' for x in ALLOWED_RESIUDES])
ATOM_SELECTION_STR = " or ".join([f'element {x}' for x in ALLOWED_ELEMENTS]) # ignore post-translational modification
EXCLUSION_LIST = ['HOH', 'DOD', 'WAT', 'NAG', 'MAN', 'UNK', 'GLC', 'ABA', 'MPD', 'GOL', 'SO4', 'PO4']


def cleanup_residues(univ):
    """Clean up residue names in a Universe object.

    This function modifies the residue names in the provided Universe object
    by removing any numeric characters and converting the residue names to uppercase.

    Parameters
    ----------
    univ : MDAnalysis.core.groups.Universe
        The Universe object representing the molecular system.

    Returns
    -------
    MDAnalysis.core.groups.Universe
        A modified Universe object with cleaned up residue names.
    """
    res_names = univ.residues.resnames
    new_names = [ "".join(re.findall(".*[a-zA-Z]+", name)).upper() for name in res_names]
    univ.residues.resnames = new_names
    
    return univ


def undo_se_modification(univ):
    """Modify selenium (Se) atoms to sulfur (S) atoms in a Universe object.

    This function  changes selenium (Se) atoms in the provided Universe object
    to sulfur (S) atoms. This is done to revert the selenomethionine post translational modification.

    Parameters
    ----------
    univ : MDAnalysis.core.groups.Universe
        The Universe object representing the molecular system.

    Returns
    -------
    MDAnalysis.core.groups.Universe
        A modified Universe object with selenium (Se) atoms converted back to sulfur (S) atoms.
    """
    se_atoms = univ.select_atoms('protein and element Se')
    se_atoms.elements = 'S'
    
    return univ


def clean_alternate_positions(input_dir:str, output_dir:str):
    """Clean PDB files in the input directory by removing alternate positions of residues.

    This function reads all the PDB files in the input directory, removes the alternate positions of residues
    (keeping only the first position), and writes the cleaned PDB files to the output directory.

    Parameters
    ----------
    input_dir : str
        The path to the directory containing the input PDB files.

    output_dir : str
        The path to the directory where cleaned PDB files will be written.
    """
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    
    res_group = '|'.join(ALLOWED_RESIUDES)
    r = re.compile(f'^ATOM.*([2-9]|[B-Z])({res_group})')
    
    for file_path in glob(f'{input_dir}/*.pdb'):
        with open(file_path, 'r') as infile:
            lines = infile.readlines()
            
        pdb_name = file_path.split('/')[-1]
        out_path = f'{output_dir}/{pdb_name}'
        
        with open(out_path, 'w') as outfile:
            outfile.writelines([line for line in lines if not re.search(r, line)])
    

def convert_to_mol2(in_file, structure_name, out_directory, addH=True, in_format='pdb', out_name='protein', parse_prot=True):
    """Convert a molecular structure file to the MOL2 format with optional additional preprocessing steps.

    This function takes a molecular structure file in a specified format and converts it to the MOL2 format.
    Optionally, it performs preprocessing steps on the molecular structure, such as adding hydrogens,
    cleaning up residue names, and undoing selenium (Se) to sulfur (S) modifications.

    Parameters
    ----------
    in_file : str
        The path to the input molecular structure file.

    structure_name : str
        The name of the structure or system being processed.

    out_directory : str
        The path to the output directory where the MOL2 file will be saved.

    addH : bool, optional
        If True, remove and readd hydrogen atoms to the structure before converting to MOL2.
        If False, remove all hydrogens, by default True.

    in_format : str, optional
        The format of the input molecular structure file. For example, 'pdb', 'mol2', etc., by default 'pdb'.

    out_name : str, optional
        The base name of the output MOL2 file (excluding the extension), by default 'protein'.

    parse_prot : bool, optional
        If True, perform residue name cleanup on mol2 file.

    """
    ob_input = in_file                                          # File to be input into openbabel for conversion to mol2
    output_path = out_directory + structure_name
    if not os.path.isdir(output_path): os.makedirs(output_path)
    
    # Optionally clean protein
    if parse_prot:
        univ = mda.Universe(in_file)
        univ = cleanup_residues(univ)
        univ = undo_se_modification(univ)
        prot_atoms = univ.select_atoms(f'protein and ({RES_SELECTION_STR}) and ({ATOM_SELECTION_STR})')
        ob_input = f'{output_path}/{out_name}.{in_format}'
        mda.coordinates.writer(ob_input).write(prot_atoms)
        
    # Use OpenBabel to convert protein to mol2 format
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats(in_format, "mol2")
    output_mol2_path  = f'{output_path}/{out_name}.mol2'
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, ob_input)
    
    # Remove all hydrogens and optionally add them with OpenBabel so that they're added in a unified way
    mol.DeleteHydrogens()
    if addH: 
        mol.CorrectForPH()
        mol.AddHydrogens()
    
    obConversion.WriteFile(mol, output_mol2_path)
    
    # Optionally cleanup residue names
    if parse_prot:
        if ob_input != output_mol2_path:
            os.remove(ob_input) # cleaning temp file
        univ = mda.Universe(output_mol2_path) 
        univ = cleanup_residues(univ)
        mda.coordinates.writer(output_mol2_path).write(univ.atoms)


def load_p2rank_set(file, skiprows=0, pdb_dir='unprocessed_pdb', joined_style=False):
    """Load P2Rank dataset file and convert paths to proper file paths.

    This function reads the P2Rank dataset file and converts the paths in the file to proper file paths
    by prepending the root directory (benchmark_data_dir) and the specified pdb_dir to each path.

    Parameters
    ----------
    file : str
        The path to the P2Rank dataset file to be loaded.

    skiprows : int, optional
        The number of rows to skip while reading the dataset file, by default 0.

    pdb_dir : str, optional
        The directory where the PDB files are located, by default 'unprocessed_pdb'.

    joined_style : bool, optional
        If True, the P2Rank dataset file contains paths in a joined-style format, where subset directories are removed.
        The function will then append 'benchmark_data_dir' to the path before adding 'pdb_dir', by default False.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the proper file paths after preprocessing.
    """
    df = pd.read_csv(file, sep='\s+', names=['path'], index_col=False, skiprows=skiprows)
    if joined_style: df['path'] = ['/'.join(file.split('/')[::2]) for file in df['path']] #removing subset directory
    df['path'] = ['benchmark_data_dir/'+f'/{pdb_dir}/'.join(file.split('/')) for file in df['path']]
    
    return df


def load_p2rank_mlig(file, skiprows, pdb_dir='unprocessed_pdb'):
    """Load P2Rank MLigands dataset file and preprocess ligands information.

    This function reads the P2Rank MLigands dataset file and preprocesses the ligands information by removing rows
    with '<CONFLICTS>' as ligands. It also converts the paths in the file to proper file paths by prepending the root
    directory (benchmark_data_dir) and the specified pdb_dir to each path.

    Parameters
    ----------
    file : str
        The path to the P2Rank MLigands dataset csv to be loaded.

    skiprows : int or list-like or callable, optional
        The number of rows to skip while reading the dataset file or list of row indices to skip,
        or a callable that takes a row index as input and returns True if the row should be skipped, by default 0.

    pdb_dir : str, optional
        The directory where the PDB files are located, by default 'unprocessed_pdb'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the proper file paths and preprocessed ligands information.
    """
    df = pd.read_csv(file, sep='\s+', names=['path', 'ligands'], index_col=False, skiprows=skiprows)
    df = df[df['ligands'] != '<CONFLICTS>']
    df['path'] = ['benchmark_data_dir/'+f'/{pdb_dir}/'.join(file.split('/')) for file in df['path']]
    df['ligands'] = [l.split(',') for l in df['ligands']]
    
    return df


def write_mlig(df, out_file):
    """Write ligands information for each pdb file to an output file.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing ligands information. It must have two columns: 'path' and 'ligands'.
        The 'path' column contains file paths to protein structures, and the 'ligands' column contains lists of ligands.

    out_file : str
        The path to the output file where ligands information for each pdb file will be written.
    """
    with open(out_file, 'w') as f:
        f.write('HEADER: protein ligand_codes\n\n')
        for val in df.values:
            val[0] = '/'.join(val[0].split('/')[1:4:2])
            f.write(f'{val[0]}  {",".join(val[1])}\n')


def load_p2rank_test_ligands(p2rank_file):
    """Load P2Rank test ligands information from a CSV file.

    This function reads the P2Rank test ligands information from a CSV file and processes the data into a Pandas dataframe.

    Parameters
    ----------
    p2rank_file : str
        The path to the CSV file containing P2Rank test ligands information.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing preprocessed P2Rank test ligands information.

    """
    ligand_df = pd.read_csv(p2rank_file, delimiter=', ', engine='python')
    ligand_df['atomIds'] = [[int(atom) for atom in row.split(' ')] for row in ligand_df['atomIds']]
    ligand_df['ligand'] = [row.split('&') for row in ligand_df['ligand']]
    
    return ligand_df


def extract_ligands_from_p2rank_df(path, name, out_dir, ligand_df):
    """Extract ligands from a P2Rank DataFrame and save them as individual PDB files.

        This function extracts ligands from a molecular structure file stored in 'path' based on the information
        provided in the 'ligand_df' DataFrame. The extracted ligands are saved as individual PDB files in the 'out_dir'.

        Parameters
        ----------
        path : str
            The path to the molecular structure file from which to extract ligands.

        name : str
            The name of the file (without extension) from which to extract ligands.

        out_dir : str
            The path to the output directory where the extracted ligands will be saved as individual PDB files.

        ligand_df : pandas.DataFrame
            A DataFrame containing ligand information, such as 'file', 'ligand', '#atoms', and 'atomIds'.
    """
    univ = mda.Universe(path)
    selected_rows = ligand_df[ligand_df['file'] == f'{name}.pdb']
    
    lig_ind = 0
    for i in selected_rows.index:
        resnames, n_atoms, ids = selected_rows['ligand'][i], selected_rows['#atoms'][i], selected_rows['atomIds'][i]
        selection = np.isin(univ.atoms.resnames, resnames) * np.isin(univ.atoms.ids, ids)
        ligand = univ.atoms[selection]
        
        if ligand.n_atoms == n_atoms:
            ligand.write(f'{out_dir}/ligand_{lig_ind}.pdb')
            lig_ind += 1
        else:
            print(f'Error: Ligand match for {resnames} not found in {name}.pdb.', flush=True)


def select_ligands_from_p2rank_df(univ, name, ligand_df):
    """Select ligands from a P2Rank DataFrame based on the molecular structure.

        This function selects ligands from the mdanalysis universe 'univ'
        based on the information provided in the 'ligand_df' DataFrame.

        Parameters
        ----------
        univ : MDAnalysis.Universe
            The MDAnalysis Universe object representing the molecular structure.

        name : str
            The name of the molecular structure (without extension) for which to select ligands.

        ligand_df : pandas.DataFrame
            A DataFrame containing ligand information, such as 'file', 'ligand', '#atoms', and 'atomIds'.

        Returns
        -------
        list of MDAnalysis.AtomGroup
            A list of MDAnalysis AtomGroup objects, each representing a selected ligand.
    """
    selected_rows = ligand_df[ligand_df['file'] == f'{name}.pdb']
    
    ligands = []
    for i in selected_rows.index:
        resnames, n_atoms, ids = selected_rows['ligand'][i], selected_rows['#atoms'][i], selected_rows['atomIds'][i]
        selection = np.isin(univ.atoms.resnames, resnames) * np.isin(univ.atoms.ids, ids)
        ligand = univ.atoms[selection]
        
        if ligand.n_atoms == n_atoms:
            ligands.append(ligand)
        else:
            print(f'Error: Ligand match for {resnames} not found in {name}.pdb.', flush=True)

    return ligands


def chains_bound_to(univ, ligands):
    """Find chains bound to the specified ligands in a molecular structure.

        This function identifies the chains that are bound to the specified ligands in the molecular structure
        represented by the MDAnalysis Universe object 'univ'.

        Parameters
        ----------
        univ : MDAnalysis.Universe
            The MDAnalysis Universe object representing the molecular structure.

        ligands : list of MDAnalysis.AtomGroup
            A list of MDAnalysis AtomGroup objects, each representing a ligand to which chains are bound.

        Returns
        -------
        list of numpy.ndarray
            A list of NumPy arrays, each containing the unique chainIDs of chains bound to the corresponding ligand.
    """
    chain_sets = []
    for ligand in ligands:
        nearby = univ.select_atoms('protein and around 4 group ligand', ligand=ligand)
        chain_sets.append(np.unique(nearby.chainIDs))
    
    return chain_sets


def chain_graph_components(chain_sets):
    """Find connected components of the chain graph.

    This function creates a chain graph from the provided chain sets and finds the connected components in the graph.

    Parameters
    ----------
    chain_sets : list of numpy.ndarray
        A list of NumPy arrays, where each array contains the unique chainIDs of chains bound to a ligand.

    Returns
    -------
    comp : generator of sets
        A generator of connected components in the chain graph.
    """
    bound_chains = np.unique([chain for subset in chain_sets for chain in subset])
    chain_graph = nx.Graph()

    chain_graph.add_nodes_from(bound_chains)

    for chains in chain_sets:
        if len(chains) > 1:
            for start in range(len(chains)-1):
                chain_graph.add_edge(chains[start], chains[start+1])

    return nx.connected_components(chain_graph)


def p2rank_df_intersect(mlig_df, p2rank_df):
    """Intersect two P2Rank DataFrames to find common ligands.

        This function intersects two P2Rank DataFrames, 'mlig_df' and 'p2rank_df', to find common ligands
        based on their 'ligand', '#atoms', and 'atomIds' attributes.

        Parameters
        ----------
        mlig_df : pandas.DataFrame
            The first P2Rank DataFrame containing ligand information.

        p2rank_df : pandas.DataFrame
            The second P2Rank DataFrame containing ligand information.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing the common ligands between 'mlig_df' and 'p2rank_df'.

    """
    concat_list = [pd.DataFrame(columns=mlig_df.columns)]

    for i in mlig_df.index:
        row = mlig_df.iloc[i]
        resname, n_atoms, ids = row['ligand'], row['#atoms'], row['atomIds']

        resname_match = np.array([res == resname for res in p2rank_df['ligand']])
        n_atoms_match = p2rank_df['#atoms'] == n_atoms
        ids_match = np.array([atoms == ids for atoms in p2rank_df['atomIds']])

        match = resname_match * n_atoms_match * ids_match
        if np.any(match):
            concat_list.append(row.to_frame().T)

    intersect_df = pd.concat(concat_list, ignore_index=True)
    
    return intersect_df


def check_p2rank_criteria(prot_univ, lig_univ):
    """Check if a protein-ligand complex meets P2Rank criteria.

        This function checks if a protein-ligand complex meets the criteria required by P2Rank.
        This criteria seeks to eliminate crawling ligands from the dataset.

        Parameters
        ----------
        prot_univ : MDAnalysis.Universe
            The MDAnalysis Universe object representing the protein in the complex.

        lig_univ : MDAnalysis.Universe
            The MDAnalysis Universe object representing the ligand in the complex.

        Returns
        -------
        bool
            True if the complex meets the P2Rank criteria, False otherwise.
    """
    prot_univ.add_TopologyAttr('record_type')
    prot_univ.atoms.record_types = 'protein'
    lig_univ.add_TopologyAttr('record_type')
    lig_univ.atoms.record_types = 'ligand'


    univ = mda.Merge(prot_univ.atoms, lig_univ.atoms)
    univ.dimensions = None # this prevents PBC bugs in distance calculation

    valid_resnames = np.all([res.resname not in EXCLUSION_LIST for res in lig_univ.residues])
    not_prot = np.all([res.resname not in ALLOWED_RESIUDES for res in lig_univ.residues])
    nearby = univ.select_atoms('record_type ligand and around 4 protein').n_atoms > 0
    if valid_resnames and not_prot and nearby and (lig_univ.atoms.n_atoms >= 5):
        com = lig_univ.atoms.center_of_mass()
        com_string = ' '.join(com.astype(str).tolist())
        not_protruding = univ.select_atoms(f'protein and not element H and point {com_string} 5.5').n_atoms > 0
        if not_protruding:
            return True

    return False


def process_train_p2rank_style(file, output_dir): 
    """Process a protein-ligand complex in P2Rank style for the training set.

        This function processes a protein-ligand complex in P2Rank style for the training set. It converts the protein and ligand
        structures into MOL2 format, checks if the ligand meets the P2Rank criteria, and saves the valid ligands in the
        appropriate directories.

        Parameters
        ----------
        file : str
            The name of the protein-ligand complex to process.

        output_dir : str
            The directory where the processed files will be saved.

        Raises
        ------
        AssertionError
            If the function fails to find a ligand in the provided file.
    """
    try:
        prepend = os.getcwd()
        structure_name = file

        if not os.path.isdir(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name)): 
            os.makedirs(os.path.join(prepend,output_dir,"ready_to_parse_mol2",structure_name))

        convert_to_mol2(f'{prepend}/scPDB_data_dir/unprocessed_mol2/{file}/protein.mol2', structure_name, f'{prepend}/{output_dir}/ready_to_parse_mol2/', addH=True, in_format='mol2')
        
        for file_path in glob(f'{prepend}/scPDB_data_dir/unprocessed_mol2/{file}/*'):
            if 'ligand' in file_path:
                prot_univ = mda.Universe(f'{prepend}/{output_dir}/ready_to_parse_mol2/{file}/protein.mol2') 
                lig_univ = mda.Universe(file_path)
                passing_p2rank = check_p2rank_criteria(prot_univ, lig_univ)
                if passing_p2rank:
                    shutil.copyfile(file_path, f'{prepend}/{output_dir}/ready_to_parse_mol2/{file}/{file_path.split("/")[-1]}')
        
        n_ligands = np.sum(['ligand' in file for file in os.listdir(f'{prepend}/{output_dir}/ready_to_parse_mol2/{file}/')])
        if n_ligands > 0:
            if not os.path.isdir(f'{prepend}/{output_dir}/raw'): os.makedirs(f'{prepend}/{output_dir}/raw')
            if not os.path.isdir(f'{prepend}/{output_dir}/mol2'): os.makedirs(f'{prepend}/{output_dir}/mol2')
            process_system(f'{prepend}/{output_dir}/ready_to_parse_mol2/{structure_name}', save_directory=f'./{output_dir}')
        else:
            with open(f'{prepend}/{output_dir}/no_ligands.txt', 'a') as f:
                f.write(f'{structure_name}\n')
        
    except AssertionError as e: 
        print("Failed to find ligand in", file)
    except Exception as e:
        raise e

        

def process_p2rank_set(path, ligand_df, data_dir="benchmark_data_dir", chen_fix=False):
    """Process a protein-ligand complex in P2Rank style.

        This function processes a protein-ligand complex in P2Rank style. It converts the protein and ligand structures into MOL2 format,
        extracts valid ligands based on the provided ligand DataFrame, and saves the valid ligands in the appropriate directories.

        Parameters
        ----------
        path : str
            The path to the protein-ligand complex in P2Rank set format.

        ligand_df : pandas.DataFrame
            The DataFrame containing ligand information, typically obtained from loading the ligand file in P2Rank format.

        data_dir : str, optional
            The directory where the processed files will be saved, by default "benchmark_data_dir".

        chen_fix : bool, optional
            Use mdanalysis to fix pdb formatting for chen dataset.

        Raises
        ------
        AssertionError
            If the function fails to find a ligand in the provided structure_name.
    """
    try:
        prepend = os.getcwd()
        structure_name = '.'.join(path.split('/')[-1].split('.')[:-1])
        mol2_dir = f'{prepend}/{data_dir}/ready_to_parse_mol2/'
        convert_to_mol2(f'{prepend}/{path}', structure_name, mol2_dir, out_name='protein', parse_prot=True)

        if chen_fix:
            univ = mda.Universe(f'{mol2_dir}{structure_name}/system.pdb')
            mda.coordinates.writer(f'{mol2_dir}{structure_name}/system.pdb').write(univ.atoms)

        extract_ligands_from_p2rank_df(path, structure_name, f'{mol2_dir}/{structure_name}', ligand_df)

        n_ligands = np.sum(['ligand' in file for file in os.listdir(f'{mol2_dir}{structure_name}')])
        if n_ligands > 0:
            if not os.path.isdir(f'{prepend}/{data_dir}/raw'): os.makedirs(f'{prepend}/{data_dir}/raw')
            if not os.path.isdir(f'{prepend}/{data_dir}/mol2'): os.makedirs(f'{prepend}/{data_dir}/mol2')
            process_system(mol2_dir + structure_name, save_directory='./'+data_dir)
        else:
            with open(f'{prepend}/{data_dir}/no_ligands.txt', 'a') as f:
                f.write(f'{structure_name}\n')
        
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        raise e


def process_p2rank_chains(path, ligand_df, data_dir):
    """Process a protein-ligand complex with multiple chains in P2Rank format.

    This function processes a protein-ligand complex with multiple chains in P2Rank format. It splits the complex into
    interconnected subsystems based on the ligand-protein interactions and processes each subsystem separately. The
    chains are not processed seperately if there is a bound ligand near the interface.

    Parameters
    ----------
    path : str
        The path to the protein-ligand complex in P2Rank format.

    ligand_df : pandas.DataFrame
        The DataFrame containing ligand information.

    data_dir : str
        The directory where the processed files will be saved.

    Raises
    ------
    AssertionError
        If the function fails to find a ligand in the provided structure.

    """
    try:
        prepend = os.getcwd()
        structure_name = '.'.join(path.split('/')[-1].split('.')[:-1])
        mol2_dir = f'{prepend}/{data_dir}/ready_to_parse_mol2/'
        pdb_dir = f'{prepend}/{data_dir}/split_pdb/'
        if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir)

        univ = mda.Universe(path)
        univ.dimensions = None # this prevents PBC bugs in distance calculation
        ligands = select_ligands_from_p2rank_df(univ, structure_name, ligand_df)

        if len(ligands) == 0:
            with open(f'{prepend}/{data_dir}/no_ligands.txt', 'a') as f:
                f.write(f'{structure_name}\n')
            return

        chain_sets = chains_bound_to(univ, ligands) # calculating which chains are associated with each ligand
        connected_chains = chain_graph_components(chain_sets) # connecting chains with interfacial ligands

        for c in connected_chains:
            chains = list(c)
            chains.sort()
            subsystem_atoms = univ.select_atoms(f'protein and (chainID {" or chainID ".join(c)})')
            subsystem_name = f'{structure_name}{"".join(chains)}'
            subsystem_ligands = []
            for i in range(len(ligands)):
                if set(chain_sets[i]).issubset(c): subsystem_ligands.append(ligands[i])
            subsystem = mda.Merge(subsystem_atoms, *subsystem_ligands)
            
            subsystem.atoms.write(f'{pdb_dir}/{subsystem_name}.pdb') # saving split pdb for p2rank inference
            convert_to_mol2(f'{pdb_dir}/{subsystem_name}.pdb', subsystem_name, mol2_dir, out_name='protein', parse_prot=True)

            for lig_ind in range(len(subsystem_ligands)):
                sub_lig = subsystem_ligands[lig_ind]
                sub_lig.write(f'{mol2_dir}/{subsystem_name}/ligand_{lig_ind}.pdb')
                
            if not os.path.isdir(f'{prepend}/{data_dir}/raw'): os.makedirs(f'{prepend}/{data_dir}/raw')
            if not os.path.isdir(f'{prepend}/{data_dir}/mol2'): os.makedirs(f'{prepend}/{data_dir}/mol2')
            process_system(f'{mol2_dir}{subsystem_name}', save_directory='./'+data_dir)
        
    except AssertionError as e:
        print("Failed to find ligand in", structure_name)
    except Exception as e:  
        raise e


def process_production_set(path, data_dir="benchmark_data_dir/production", skip_hydrogen_cleanup=False):
    """Process a protein-ligand complex in production set format.

    This function processes a protein. It converts the complex to mol2 format
    and saves the processed files in the specified output directory.

    Parameters
    ----------
    path : str
        The path to the protein-ligand complex in production set format.

    data_dir : str, optional
        The directory where the processed files will be saved, by default "benchmark_data_dir/production".

    skip_hydrogen_cleanup: bool, optional
        If true: reuse existing hydrogents, if false: remove and readd hydrogens with OpenBabel, false by default.
        
    Raises
    ------
    Exception
        If any error occurs during processing.
    """
    try:
        prepend = os.getcwd()
        structure_name = path.split('/')[-1].split('.')[0]
        mol2_dir = f'{prepend}/{data_dir}/ready_to_parse_mol2/'
        format = path.split('.')[-1]
        addH = not skip_hydrogen_cleanup
        convert_to_mol2(f'{prepend}/{path}', structure_name, mol2_dir, addH=addH, in_format=format, out_name='protein', parse_prot=True)
        if not os.path.isdir(f'{prepend}/{data_dir}/raw'): os.makedirs(f'{prepend}/{data_dir}/raw')
        if not os.path.isdir(f'{prepend}/{data_dir}/mol2'): os.makedirs(f'{prepend}/{data_dir}/mol2')
        process_system(mol2_dir + structure_name, save_directory=f'{prepend}/{data_dir}', parse_ligands=False)
    except Exception as e:
        raise e


if __name__ == "__main__":
    num_cores = 24
    prepend = os.getcwd()
    from joblib.externals.loky import set_loky_pickler
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser(description="Prepare datasets for GNN inference.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", choices=["scpdb", "coach420", "coach420_mlig", "coach420_intersect",
    "holo4k", "holo4k_mlig", "holo4k_intersect", "holo4k_chains", "production"], help="Dataset to prepare.")
    parser.add_argument("-sh", "--skip_hydrogen_cleanup", help="Remove and re-add hydrogens for the production set.", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset
    
    if args.dataset != "production" and args.skip_hydrogen_cleanup:
        print("Hydrogen cleanup arg is only supported for the productiond dataset.")
        exit(1)
 
    if dataset == "scpdb":
        print("Parsing the standard train set with p2rank criteria")
        nolig_file = f'{prepend}/scPDB_data_dir/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        mol2_files = [filename for filename in sorted(list(os.listdir(prepend +'/scPDB_data_dir/unprocessed_mol2')))]
        Parallel(n_jobs=num_cores)(delayed(process_train_p2rank_style)(filename, 'scPDB_data_dir') for filename in tqdm(mol2_files[:])) 

    elif dataset == "coach420" or dataset == "coach420_mlig":
        ligand_df = load_p2rank_test_ligands(f'{prepend}/test_metrics/{dataset}/p2rank/cases/ligands.csv')
        data_dir = f'benchmark_data_dir/{dataset}'
        nolig_file = f'{prepend}/{data_dir}/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        systems = np.unique(ligand_df['file'])
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(f'benchmark_data_dir/coach420/unprocessed_pdb/{system}',
         ligand_df, data_dir=data_dir) for system in tqdm(systems))
    
    elif dataset == "holo4k" or dataset == "holo4k_mlig":
        print('Cleaning alternate positions...')
        clean_alternate_positions(f'{prepend}/benchmark_data_dir/holo4k/unprocessed_pdb/', f'{prepend}/benchmark_data_dir/holo4k/cleaned_pdb/')
        ligand_df = load_p2rank_test_ligands(f'{prepend}/test_metrics/{dataset}/p2rank/cases/ligands.csv')
        data_dir = f'benchmark_data_dir/{dataset}'
        nolig_file = f'{prepend}/{data_dir}/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        systems = np.unique(ligand_df['file'])
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(f'benchmark_data_dir/holo4k/cleaned_pdb/{system}',
         ligand_df, data_dir=data_dir) for system in tqdm(systems))
    
    elif dataset == "coach420_intersect":
        p2rank_df = load_p2rank_test_ligands(f'{prepend}/test_metrics/coach420/p2rank/cases/ligands.csv')
        mlig_df =  load_p2rank_test_ligands(f'{prepend}/test_metrics/coach420_mlig/p2rank/cases/ligands.csv')
        ligand_df = p2rank_df_intersect(mlig_df, p2rank_df)

        data_dir = f'benchmark_data_dir/{dataset}'
        nolig_file = f'{prepend}/{data_dir}/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        systems = np.unique(ligand_df['file'])
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(f'benchmark_data_dir/coach420/unprocessed_pdb/{system}',
         ligand_df, data_dir=data_dir) for system in tqdm(systems))

    elif dataset == "holo4k_intersect":
        print('Cleaning alternate positions...')
        clean_alternate_positions(f'{prepend}/benchmark_data_dir/holo4k/unprocessed_pdb/', f'{prepend}/benchmark_data_dir/holo4k/cleaned_pdb/')        

        p2rank_df = load_p2rank_test_ligands(f'{prepend}/test_metrics/holo4k/p2rank/cases/ligands.csv')
        mlig_df =  load_p2rank_test_ligands(f'{prepend}/test_metrics/holo4k_mlig/p2rank/cases/ligands.csv')
        ligand_df = p2rank_df_intersect(mlig_df, p2rank_df)

        data_dir = f'benchmark_data_dir/{dataset}'
        nolig_file = f'{prepend}/{data_dir}/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        systems = np.unique(ligand_df['file'])
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_set)(f'benchmark_data_dir/holo4k/cleaned_pdb/{system}',
         ligand_df, data_dir=data_dir) for system in tqdm(systems))

    elif dataset == "holo4k_chains":
        print('Cleaning alternate positions...')
        clean_alternate_positions(f'{prepend}/benchmark_data_dir/holo4k/unprocessed_pdb/', f'{prepend}/benchmark_data_dir/holo4k/cleaned_pdb/')        

        p2rank_df = load_p2rank_test_ligands(f'{prepend}/test_metrics/holo4k/p2rank/cases/ligands.csv')
        mlig_df =  load_p2rank_test_ligands(f'{prepend}/test_metrics/holo4k_mlig/p2rank/cases/ligands.csv')
        ligand_df = p2rank_df_intersect(mlig_df, p2rank_df)

        data_dir = f'benchmark_data_dir/{dataset}'
        nolig_file = f'{prepend}/{data_dir}/no_ligands.txt'
        if os.path.exists(nolig_file): os.remove(nolig_file)
        systems = np.unique(ligand_df['file'])
        Parallel(n_jobs=num_cores)(delayed(process_p2rank_chains)(f'benchmark_data_dir/holo4k/cleaned_pdb/{system}',
         ligand_df, data_dir=data_dir) for system in tqdm(systems))

    elif dataset == "production":
        prod_dir = f'benchmark_data_dir/production'
        paths = glob(f'{prod_dir}/unprocessed_inputs/*')
        Parallel(n_jobs=num_cores)(delayed(process_production_set)(path, prod_dir, args.skip_hydrogen_cleanup) for path in tqdm(paths))
