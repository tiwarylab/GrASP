import numpy as np
import pandas as pd

def uniprot_inclusion_list(include_df, static_df): # drop entries from first df not second
    inclusion_list = []
    
    static_list = list(np.concatenate(static_df['UNIPROT']).flat)
    for index, row in include_df.iterrows():
        overlap = False
        for u_id in row['UNIPROT']:
            if u_id in static_list:
                overlap = True
                break # we can stop checking if any overlap
        if not overlap:
            inclusion_list.append(row['PDB'])
            
    return inclusion_list


def write_inclusion_list(inclusion_list, out_file):
    with open(out_file, 'w') as f:
        for pdb in inclusion_list:
            f.write(pdb + '\n')


if __name__ == '__main__':
    scpdb_uniprot = pd.read_pickle('./scPDB_data_dir/scPDB_uniprot.pkl')

    data_dir = './benchmark_data_dir'
    split_prefix = './splits/train_ids_'
    test_sets = ['coach420', 'coach420(mlig)', 'holo4k', 'holo4k(mlig)', 'misato']
    for test_set in test_sets:
            test_uniprot = pd.read_pickle(f'{data_dir}/{test_set}_uniprot.pkl')
            if test_set == 'misato':
                chen_uniprot = pd.read_pickle(f'{data_dir}/chen11_uniprot.pkl')
                inclusion_list = uniprot_inclusion_list(test_uniprot, pd.concat([scpdb_uniprot, chen_uniprot], ignore_index=True)) # here we drop from test not train
                write_inclusion_list(inclusion_list, f'./splits/test_ids_{test_set}_uniprot')
            else:
                inclusion_list = uniprot_inclusion_list(scpdb_uniprot, test_uniprot)
                write_inclusion_list(inclusion_list, f'{split_prefix}{test_set}_uniprot')

