# Graph Attention Site Prediction (GrASP)
## Preprint
https://biorxiv.org/content/10.1101/2023.07.25.550565v1

## Colab
Fetch a PDB file and try GrASP on it in our Colab demo.

[![Open in Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiwarylab/GrASP/blob/main/GrASP.ipynb)


## Download Datasets
Coming soon!

## How to Run
Currently, only production mode on a pre-trained model is supported until datasets are online.
* Build the conda environment by running
 ```
mamba create -n grasp python==3.7.10

mamba install conda-forge::cython
mamba install conda-forge::openbabel=2.4.1
mamba install conda-forge::rdkit
mamba install conda-forge::mdtraj
mamba install conda-forge::mdanalysis

pip install networkx==2.5 ```

* Move protein structures to `./benchmark_data_dir/production/unprocessed_inputs/`. Heteroatoms do not need to be removed, they will be cleaned during parsing.
* Load `ob` and parse the structures into graphs.
 ```
 python3 parse_files.py production
 ```
* Run GrASP over the protein graphs.
 ```
 python3 infer_test_set.py
 ```
* Paint structures with GrASP predictions in the b-factor column.
 ```
 conda deactivate; conda activate ob
 python3 color_pdb.py
 ```
## Supported Formats
PDB and mol2 formats are supported and validated. Other formats supported by both MDAnalysis and OpenBabel 2.4.1 may be working but have not been tested.
