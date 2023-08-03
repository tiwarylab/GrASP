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
* Build the conda environments in `./envs/ob_env.yml` and `./envs/pytorch_env.yml`. This will add two new conda environments named `ob` and `pytorch_env` respectively.
 ```
 conda env create -f envs/ob_env.yml
 conda env create -f envs/pytorch_env.yml
 ```

* Move protein structures to `./benchmark_data_dir/production/unprocessed_inputs/`. Heteroatoms do not need to be removed, they will be cleaned during parsing.
* Load `ob` and parse the structures into graphs.
 ```
 conda activate ob
 python3 parse_files.py production
 ```
* Run GrASP over the protein graphs.
 ```
 conda deactivate; conda activate pytorch_env
 python3 infer_test_set.py
 ```
* Paint structures with GrASP predictions in the b-factor column.
 ```
 conda deactivate; conda activate ob
 python3 color_pdb.py
 ```

