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

def reverse_entries(dataset_path,evidence_path):

    new_columns = ['smile', 'fingerprint', 'variantfeature', 'label']
    new_data = pd.DataFrame(columns=new_columns)
    dataset = pd.read_pickle(dataset_path)
    #print(dataset)
    print(len(dataset['variantfeature'][0])) # 3904, 61d seq

    for i in range(dataset.shape[0]): 
        variantfeature_list = []
        smile = dataset['smile'][i]
        fingerprint = dataset['fingerprint'][i]

        # reverse label
        labelori = dataset['label'][i]
        if(labelori == 1):
            label = 0
        elif(labelori == 0):
            label = 1

        # reverse onehot sequence
        seqbefore = dataset['seqafter'][i]
        seqafter = dataset['seqbefore'][i]

        # reverse variant features
        hmm_before_list = dataset['variantfeature'][i][0:1830]
        hmm_after_list = dataset['variantfeature'][i][1830:3660]
        other_list = dataset['variantfeature'][i][3660:]
        for item in hmm_after_list:
            variantfeature_list.append(item)
        for item in hmm_before_list:
            variantfeature_list.append(item)
        for item in other_list:
            variantfeature_list.append(item)
        variantfeature = np.array(variantfeature_list) # 3904
        new_data = new_data.append([{'smile': smile, 'fingerprint': fingerprint, 'seqbefore':seqbefore, 'seqafter':seqafter,
                                            'variantfeature': variantfeature, 'label': label}], ignore_index=True)

    new_data.to_pickle(evidence_path)


def non_mutant_entries(dataset_path,evidence_path):

    new_columns = ['smile', 'fingerprint', 'variantfeature', 'label']
    new_data = pd.DataFrame(columns=new_columns)
    dataset = pd.read_pickle(dataset_path)
    #print(dataset)
    print(len(dataset['variantfeature'][0])) # 3904, 61d seq

    for i in range(dataset.shape[0]): 
        variantfeature_list = []
        smile = dataset['smile'][i]
        fingerprint = dataset['fingerprint'][i]

        # no mutation label
        label = 0

        # no mutation onehot sequence
        seqbefore = dataset['seqbefore'][i]
        seqafter = dataset['seqbefore'][i]

        # no mutation variant features
        hmm_before_list = dataset['variantfeature'][i][0:1830]
        other_list = dataset['variantfeature'][i][3660:]
        for item in hmm_before_list:
            variantfeature_list.append(item)
        for item in hmm_before_list:
            variantfeature_list.append(item)
        for item in other_list:
            variantfeature_list.append(item)
        variantfeature = np.array(variantfeature_list) # 3904
        new_data = new_data.append([{'smile': smile, 'fingerprint': fingerprint, 'seqbefore':seqbefore, 'seqafter':seqafter,
                                            'variantfeature': variantfeature, 'label': label}], ignore_index=True)

    new_data.to_pickle(evidence_path)

def generate_dataset(evidence_path, processed_data_path, root, dataset):

    new_data = pd.read_pickle(evidence_path)

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
        train_data = PyDataset(root=root, dataset=dataset, xs=train_smile, xfp=train_fp, xsb=train_sb, xsa=train_sa, xv=train_variant, y=train_Y, smile_graph=smile_graph)         
    else:
        print('already exits!')

def generate_dataset_withformer(evidence_path_rev,evidence_path_non, old_evidence_path, processed_data_path, root, dataset):

    rev_data = pd.read_pickle(evidence_path_rev)
    non_data = pd.read_pickle(evidence_path_non)
    old_data = pd.read_pickle(old_evidence_path)
    new_data = pd.concat([rev_data,non_data,old_data])
    new_data = shuffle(new_data)

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
        train_data = PyDataset(root=root, dataset=dataset, xs=train_smile, xfp=train_fp, xsb=train_sb, xsa=train_sa, xv=train_variant, y=train_Y, smile_graph=smile_graph)         
    else:
        print('already exits!')

def generate_da_train():
    train_path = '../datasets/middlefile/train_data_evidence.dataset'

    evidence_path_rev = '../datasets/middlefile/da_rev_train_data_evidence.dataset'
    reverse_entries(train_path,evidence_path_rev) 
    evidence_path_non = '../datasets/middlefile/da_non_train_data_evidence.dataset'
    non_mutant_entries(train_path,evidence_path_non)

    processed_data_path = '../datasets/processed/da_train_data.pt'
    generate_dataset_withformer(evidence_path_rev,evidence_path_non,train_path,processed_data_path,'../datasets','da_train')

def generate_da_test():
    test_path = '../datasets/middlefile/test_data_evidence.dataset'
    evidence_path = '../datasets/middlefile/da_rev_test_data_evidence.dataset'
    processed_data_path = '../datasets/processed/da_rev_test_data.pt'
    reverse_entries(test_path,evidence_path) 
    generate_dataset(evidence_path,processed_data_path,'../datasets','da_rev_test')

    evidence_path = '../datasets/middlefile/da_non_test_data_evidence.dataset'
    processed_data_path = '../datasets/processed/da_non_test_data.pt'
    non_mutant_entries(test_path,evidence_path)
    generate_dataset(evidence_path,processed_data_path,'../datasets','da_non_test')

def generate_5fold_train():
    for i in range(5):
        train_path = '../datasets/fivefold/'+str(i+1)+'fold_train_evidence.dataset'

        evidence_path_rev = '../datasets/fivefold_da/'+str(i+1)+ 'fold_rev_train.dataset'
        reverse_entries(train_path,evidence_path_rev)
        evidence_path_non = '../datasets/fivefold_da/'+str(i+1)+ 'fold_non_train.dataset'
        non_mutant_entries(train_path,evidence_path_non)

        processed_data_path = '../datasets/fivefold_da/'+str(i+1)+ 'fold_da_train.pt'
        generate_dataset_withformer(evidence_path_rev,evidence_path_non,train_path,processed_data_path,'../datasets/fivefold_da',str(i+1)+'fold_da_train')

def generate_5fold_valid():
    for i in range(5):
        valid_path = '../datasets/fivefold/'+str(i+1)+'fold_valid_evidence.dataset'

        evidence_path_rev = '../datasets/fivefold_da/'+str(i+1)+ 'fold_rev_valid.dataset'
        reverse_entries(valid_path,evidence_path_rev)
        evidence_path_non = '../datasets/fivefold_da/'+str(i+1)+ 'fold_non_valid.dataset'
        non_mutant_entries(valid_path,evidence_path_non)

        processed_data_path = '../datasets/fivefold_da/'+str(i+1)+ 'fold_da_valid.pt'
        generate_dataset_withformer(evidence_path_rev,evidence_path_non,valid_path,processed_data_path,'../datasets/fivefold_da',str(i+1)+'fold_da_valid')
        
if __name__=='__main__':

    # generate DA train set and split
    generate_da_train()

    # generate DA test set
    generate_da_test()

    '''
    # prepare 5fold-train/valid dataset (removed)
    generate_5fold_train()
    generate_5fold_valid()
    '''