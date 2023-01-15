import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from sklearn.utils import shuffle
from rdkit import Chem
import networkx as nx
from utils import PyDataset
import json

import warnings
warnings.filterwarnings('ignore')

# ref: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
                    
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def split_flatten_dataset(filename):
    dataset_feature = pd.read_pickle(filename)
    shuffled_dataset = shuffle(dataset_feature)
    shuffled_dataset = shuffled_dataset.reset_index(drop=True)
    train_df = shuffled_dataset[0:600] # 600 for 5-fold
    test_df = shuffled_dataset[600:753].reset_index(drop=True) # 153
    
    new_columns = ['smile', 'fingerprint', 'variantfeature', 'label']

    train_data = pd.DataFrame(columns=new_columns)
    for i in range(train_df.shape[0]): # train_df.shape[0]
        variantfeature_list = []
        smile = train_df['smile'][i]
        fingerprint = train_df['cactvs_fingerprint'][i] # array (881,)
        onehot_before = train_df['onehot_before'][i]
        for item in np.nditer(onehot_before):
            variantfeature_list.append(int(item))
        onehot_after = train_df['onehot_after'][i]
        for item in np.nditer(onehot_after):
            variantfeature_list.append(int(item))
        hhm_before = train_df['hhm_before'][i]
        for item in np.nditer(hhm_before):
            variantfeature_list.append(int(item))
        hhm_after = train_df['hhm_after'][i]
        for item in np.nditer(hhm_after):
            variantfeature_list.append(int(item))
        rasa = train_df['rasa'][i]
        for item in rasa:
            variantfeature_list.append(item)
        ss = train_df['ss'][i]
        for item in np.nditer(ss):
            variantfeature_list.append(int(item))
        variantfeature = np.array(variantfeature_list)
        label = train_df['label'][i]
        train_data = train_data.append([{'smile': smile, 'fingerprint': fingerprint, 
                                            'variantfeature': variantfeature, 'label': label}], ignore_index=True)

    test_data = pd.DataFrame(columns=new_columns)
    for i in range(test_df.shape[0]): # test_df.shape[0]
        variantfeature_list = []
        smile = test_df['smile'][i]
        fingerprint = test_df['cactvs_fingerprint'][i] # array (881,)
        onehot_before = test_df['onehot_before'][i]
        for item in np.nditer(onehot_before):
            variantfeature_list.append(int(item))
        onehot_after = test_df['onehot_after'][i]
        for item in np.nditer(onehot_after):
            variantfeature_list.append(int(item))
        hhm_before = test_df['hhm_before'][i]
        for item in np.nditer(hhm_before):
            variantfeature_list.append(int(item))
        hhm_after = test_df['hhm_after'][i]
        for item in np.nditer(hhm_after):
            variantfeature_list.append(int(item))
        rasa = test_df['rasa'][i]
        for item in rasa:
            variantfeature_list.append(item)
        ss = test_df['ss'][i]
        for item in np.nditer(ss):
            variantfeature_list.append(int(item))
        # print(len(variantfeature_list)) 5304
        variantfeature = np.array(variantfeature_list)
        label = test_df['label'][i]
        test_data = test_data.append([{'smile': smile, 'fingerprint': fingerprint, 
                                            'variantfeature': variantfeature, 'label': label}], ignore_index=True)

    print(test_data['label'].value_counts())
    return train_data, test_data

if __name__=='__main__':

    # split and flatten evidence
    evidence_data_file_train = '../datasets/middlefile/train_data_evidence.dataset'
    evidence_data_file_test = '../datasets/middlefile/test_data_evidence.dataset'
    if ((not os.path.isfile(evidence_data_file_train)) or (not os.path.isfile(evidence_data_file_test))):
        feature_dataset_path = '../datasets/dataset_featurecode.dataset'
        train_data, test_data = split_flatten_dataset(feature_dataset_path)
        train_data.to_pickle(evidence_data_file_train)
        test_data.to_pickle(evidence_data_file_test)

    # generate 5-fold dataset


    # smile graph encoding
    compound_iso_smiles = []
    options = ['train','test']
    for option in options:
        df = pd.read_pickle('../datasets/middlefile/' + option + '_data_evidence.dataset')
        compound_iso_smiles += list( df['smile'] )
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    # print(smile_graph)

    # convert to PyTorch data format
    processed_data_file_train = '../datasets/processed/train_data.pt'
    processed_data_file_test = '../datasets/processed/test_data.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df = pd.read_pickle(evidence_data_file_train)
        
        train_smile, train_fp, train_variant,  train_Y = np.asarray(list(df['smile'])),np.asarray(list(df['fingerprint'])), \
                                                        np.asarray(list(df['variantfeature'])),np.asarray(list(df['label']))
        df = pd.read_pickle(evidence_data_file_test)
        test_smile, test_fp, test_variant,  test_Y = np.asarray(list(df['smile'])),np.asarray(list(df['fingerprint'])), \
                                                        np.asarray(list(df['variantfeature'])),np.asarray(list(df['label']))

        # make data PyTorch Geometric ready
        print('preparing train_data.pt in pytorch format!')
        train_data = PyDataset(root='../datasets', dataset='train', xs=train_smile, xfp=train_fp, xv=train_variant, y=train_Y, smile_graph=smile_graph)
        print('preparing test_data.pt in pytorch format!')
        test_data = PyDataset(root='../datasets', dataset='test', xs=test_smile, xfp=test_fp, xv=test_variant, y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have already existed') 
        
        