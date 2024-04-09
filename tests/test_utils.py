import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.sparse import csr_matrix

def mol_equality_test(ref_mol2:str, target_mol2:str):
    """
    Test the equality of two mol2 files.

    Parameters
    ----------
    ref_mol2 : str
        Path to the reference mol2 file.
    target_mol2 : str
        Path to the target mol2 file.

    Raises
    ------
    AssertionError
        If the atom positions, atom types, or bonds do not match between the two mol2 files.
    """
    ref_universe = mda.Universe(ref_mol2)
    target_universe = mda.Universe(target_mol2)
    
    # Compare atom positions
    assert ref_universe.atoms.n_atoms == target_universe.atoms.n_atoms, f"Number of atoms do not match for files {ref_mol2} and {target_mol2}"
    assert_almost_equal(ref_universe.atoms.positions, target_universe.atoms.positions)

    # Compare atom types
    assert np.all(ref_universe.atoms.types == target_universe.atoms.types), f"Atom types do not match for files {ref_mol2} and {target_mol2}"

    # Compare bonds (if they exist)
    ref_contains_bonds = False
    try:
        if len(ref_universe.bonds) > 0:
            ref_contains_bonds = True
    except mda.exceptions.NoDataError:
            pass
    
    if ref_contains_bonds:
        assert len(ref_universe.bonds) == len(target_universe.bonds), f"Number of bonds do not match for files {ref_mol2} and {target_mol2}"
        assert np.all(ref_universe.bonds == target_universe.bonds), f"Bonds do not match for files {ref_mol2} and {target_mol2}"
    
def preprocessed_npz_equality_test(ref_npz:str, target_npz:str):
    """
    Test the equality of two npz files.

    Parameters
    ----------
    ref_npz : str
        Path to the reference npz file.
    target_npz : str
        Path to the target npz file.

    Raises
    ------
    AssertionError
        If the npz files do not match.
    """
    ref_data = np.load(ref_npz, allow_pickle=True)
    target_data = np.load(target_npz, allow_pickle=True)
    
    assert ref_data.files == target_data.files, f"Files do not match for files {ref_npz} and {target_npz}"
    
    for key in ref_data.files:
        ref_item = ref_data[key]
        target_item = target_data[key]
        if ref_item.shape == ():
            ref_item = ref_item.item()
            assert target_item.shape == ()
            target_item = target_item.item()
            
        if isinstance(ref_item, np.ndarray):
            assert_almost_equal(ref_item, target_item, err_msg=f"Data does not match for key {key} in files {ref_npz} and {target_npz}")
        elif isinstance(ref_item, csr_matrix):
            assert np.all(ref_item.data == target_item.data), f"Data does not match for key {key} in files {ref_npz} and {target_npz}"
            assert np.all(ref_item.indices == target_item.indices), f"Indices do not match for key {key} in files {ref_npz} and {target_npz}"
            assert np.all(ref_item.indptr == target_item.indptr), f"Indptr does not match for key {key} in files {ref_npz} and {target_npz}"
        elif isinstance(ref_item, dict):
            assert ref_item == target_item, f"Data does not match for key {key} in files {ref_npz} and {target_npz}"
        else:
            raise NotImplementedError(f"Data type {type(ref_data[key])} not supported for key {key} in files {ref_npz} and {target_npz}")