import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from sklearn.utils import shuffle
from rdkit import Chem
import networkx as nx
from utils import PyDataset
import json
from sklearn.utils import shuffle
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

label_dict = {1: [0,1], 0: [1,0]}

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

def generate_dataset():
    evidence_path = '../datasets/DRSP/drsp_dataset_featurecode.dataset'
    processed_data_path = '../datasets/processed/DRSP_data.pt'

    df = pd.read_pickle(evidence_path)
    new_columns = ['smile', 'fingerprint', 'variantfeature', 'label']
    new_data = pd.DataFrame(columns=new_columns)
    print(df.shape[0])
    for i in range(df.shape[0]): # df.shape[0]
        variantfeature_list = []
        seqbefore_list = []
        seqafter_list = []
        smile = df['smile'][i]
        fingerprint = df['cactvs_fingerprint'][i] # array (881,)
        onehot_before = df['onehot_before'][i]
        # print(onehot_before.shape) (61, 20)
        for item in np.nditer(onehot_before):
            seqbefore_list.append(int(item))
        onehot_after = df['onehot_after'][i]
        for item in np.nditer(onehot_after):
            seqafter_list.append(int(item))
        hhm_before = df['hhm_before'][i]
        for item in np.nditer(hhm_before):
            variantfeature_list.append(int(item))
        hhm_after = df['hhm_after'][i]
        for item in np.nditer(hhm_after):
            variantfeature_list.append(int(item))
        rasa = df['rasa'][i]
        for item in rasa:
            variantfeature_list.append(item)
        ss = df['ss'][i]
        for item in np.nditer(ss):
            variantfeature_list.append(int(item))
        variantfeature = np.array(variantfeature_list) # 3904
        seqbefore = np.array(seqbefore_list) # 1220
        seqafter = np.array(seqafter_list) #1220
        #label = label_dict[df['label'][i]] # onehot
        label = df['label'][i]
        #new_data = new_data.append([{'smile': smile, 'fingerprint': fingerprint, 'seqbefore':seqbefore, 'seqafter':seqafter,
        new_data = new_data.append([{'smile': smile, 'fingerprint': fingerprint, 'seqbefore':onehot_before, 'seqafter':onehot_after,
                                            'variantfeature': variantfeature, 'label': label}], ignore_index=True)

    # smile graph encoding
    compound_iso_smiles = set(list(new_data['smile']))
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    # convert to PyTorch data format
    if (not os.path.isfile(processed_data_path)):
        train_smile, train_fp, train_sb, train_sa, train_variant,  train_Y = np.asarray(list(new_data['smile'])),np.asarray(list(new_data['fingerprint'])), \
                                                        np.asarray(new_data['seqbefore']), np.asarray(new_data['seqafter']), \
                                                        np.asarray(list(new_data['variantfeature'])),np.asarray(list(new_data['label']))
        print('preparing .pt in pytorch format!')
        train_data = PyDataset(root='../datasets', dataset='DRSP', xs=train_smile, xfp=train_fp, xsb=train_sb, xsa=train_sa, xv=train_variant, y=train_Y, smile_graph=smile_graph)         
    else:
        print('already exits!')


if __name__=='__main__':

    # generate DA test set
    generate_dataset()