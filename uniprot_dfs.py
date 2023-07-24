import requests # this is used to access json files
import pandas as pd
import argparse
import os
from tqdm.contrib import tzip

# adapted from https://gist.github.com/avrilcoghlan/e44ce43224ac601f53f1d58944ce93cf
def get_uniprot(pdb_id, chain=None):
    full_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    json_results = requests.get(full_url).json() #This calls the information back from the API using the 'requests' module, and converts it to json format

    uniprot_ids = parse_uniprots(json_results, pdb_id, chain)

    return uniprot_ids


def parse_uniprots(uniprot_json, pdb_id, pdb_chain=None):
    uniprot_dict = uniprot_json[pdb_id]['UniProt']
    uniprot_ids = list(uniprot_dict.keys())
    
    if pdb_chain is None: return uniprot_ids

    else:
        for u_id in uniprot_ids:
            chains = uniprot_dict[u_id]['mappings']
            chain_ids = [chain['chain_id'].upper() for chain in chains]
            if pdb_chain.upper() in chain_ids: return [u_id]

        return [] # when no uniprot is found


def pdb_to_uniprot_df(pdb_id_list, chain_list=None):
    if chain_list is None:
        chain_list = [None]*len(pdb_id_list)

    uniprot_id_list = []
    for pdb_id, chain in tzip(pdb_id_list, chain_list):
        uniprot_id_list.append(get_uniprot(pdb_id, chain))
    uniprot_df = pd.DataFrame({'PDB':pdb_id_list, 'UNIPROT':uniprot_id_list})
        
    return uniprot_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch UNIPROT ids for datasets.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--set", default="scpdb", choices=["scpdb", "coach420", "holo4k", "coach420_mlig", "holo4k_mlig", "misato", "chen"], help="Data set.")
    args = parser.parse_args()

    if args.set == 'scpdb':
        data_dir = './scPDB_data_dir'
        scpdb_paths = os.listdir(f'{data_dir}/unprocessed_mol2')
        scpdb_pdbs = [path[:4] for path in scpdb_paths if path[0] != '.']
        pdb_to_uniprot_df(scpdb_pdbs).to_pickle(f'{data_dir}/scPDB_uniprot.pkl')

    if args.set == 'coach420':
        data_dir = './benchmark_data_dir'
        coach_paths = pd.read_csv(f'{data_dir}/coach420.ds',names=['path'])
        coach_pdbs = [path.split('/')[-1].split('.')[0][:4] for path in coach_paths['path']]
        pdb_to_uniprot_df(coach_pdbs).to_pickle(f'{data_dir}/coach420_uniprot.pkl')

    if args.set == 'coach420_mlig':
        data_dir = './benchmark_data_dir'
        coach_mlig_paths = pd.read_csv(f'{data_dir}/coach420(mlig).ds', sep='\s+', names=['path', 'ligands'], index_col=False, skiprows=4)
        coach_mlig_paths = coach_mlig_paths[coach_mlig_paths['ligands'] != '<CONFLICTS>']
        coach_mlig_pdbs = [path.split('/')[-1].split('.')[0][:4] for path in coach_mlig_paths['path']]
        pdb_to_uniprot_df(coach_mlig_pdbs).to_pickle(f'{data_dir}/coach420(mlig)_uniprot.pkl')

    if args.set == 'holo4k':
        data_dir = './benchmark_data_dir'
        holo_paths = pd.read_csv(f'{data_dir}/holo4k.ds',names=['path'])
        holo_pdbs = [path.split('/')[-1].split('.')[0] for path in holo_paths['path']]
        pdb_to_uniprot_df(holo_pdbs).to_pickle(f'{data_dir}/holo4k_uniprot.pkl')

    if args.set == 'holo4k_mlig':
        data_dir = './benchmark_data_dir'
        holo_mlig_paths = pd.read_csv(f'{data_dir}/holo4k(mlig).ds', sep='\s+', names=['path', 'ligands'], index_col=False, skiprows=2)
        holo_mlig_paths = holo_mlig_paths[holo_mlig_paths['ligands'] != '<CONFLICTS>']
        holo_mlig_pdbs = [path.split('/')[-1].split('.')[0] for path in holo_mlig_paths['path']]
        pdb_to_uniprot_df(holo_mlig_pdbs).to_pickle(f'{data_dir}/holo4k(mlig)_uniprot.pkl')

    if args.set == 'misato':
        data_dir = './benchmark_data_dir'
        misato_pdbs = []
        with open(f'{data_dir}/misato/train_MD.txt') as file:
            misato_pdbs += file.read().lower().splitlines()
        with open(f'{data_dir}/misato/val_MD.txt') as file:
            misato_pdbs += file.read().lower().splitlines()
        with open(f'{data_dir}/misato/test_MD.txt') as file:
            misato_pdbs += file.read().lower().splitlines()
        
        pdb_to_uniprot_df(misato_pdbs).to_pickle(f'{data_dir}/misato_uniprot.pkl')

    if args.set == "chen":
        data_dir = './benchmark_data_dir'
        chen_paths = pd.read_csv(f'{data_dir}/chen11.ds', names=['path'], skiprows=5)
        chen_pdbs = [path.split('_')[-1].split('.')[0][:-1] for path in chen_paths['path']]
        chen_chains = [path.split('_')[-1].split('.')[0][-1] for path in chen_paths['path']]
        pdb_to_uniprot_df(chen_pdbs, chen_chains).to_pickle(f'{data_dir}/chen11_uniprot.pkl')

