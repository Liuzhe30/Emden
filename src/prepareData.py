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

def split_5fold_dataset(filename):
    dataset_feature = pd.read_pickle(filename) # have been shuffled
    fold_dfs = [dataset_feature[idx*120:(idx+1)*120] for idx in range(5)] # 5-fold datasets
    flatten_folds_valid = [df.reset_index(drop=True) for df in fold_dfs]
    print(dataset_feature[0:120],dataset_feature[240:])
    flatten_folds_train = [dataset_feature[120:],pd.concat([dataset_feature[0:120],dataset_feature[240:]]),
                            pd.concat([dataset_feature[0:240],dataset_feature[360:]]),
                            pd.concat([dataset_feature[0:360],dataset_feature[480:]]),dataset_feature[0:480]]
    flatten_folds_train = [df.reset_index(drop=True) for df in flatten_folds_train]
    return flatten_folds_train, flatten_folds_valid

def split_dataset(filename):
    dataset_feature = pd.read_pickle(filename)
    shuffled_dataset = shuffle(dataset_feature)
    shuffled_dataset = shuffled_dataset.reset_index(drop=True)
    train_df = shuffled_dataset[0:600] # 600 for 5-fold
    test_df = shuffled_dataset[600:753].reset_index(drop=True) # 153

    train_data = flatten_dataset(train_df)
    test_data = flatten_dataset(test_df)

    print(test_data['label'].value_counts())
    return train_data, test_data

def alldata(filename):
    dataset_feature = pd.read_pickle(filename)
    all_data = flatten_dataset(dataset_feature)
    return all_data

def flatten_dataset(df):
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
    return new_data

def generate_5fold_dataset():

    evidence_data_file_train = '../datasets/middlefile/train_data_evidence.dataset' # (600,16)
    train_data = pd.read_pickle(evidence_data_file_train)
    
    # smile graph encoding
    compound_iso_smiles = set(list(train_data['smile']))
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    # split train set to 5 parts
    flatten_folds_train, flatten_folds_valid = split_5fold_dataset(evidence_data_file_train)
    
    for i in range(5): # 5-fold
        flatten_folds_train[i].to_pickle('../datasets/fivefold/'+str(i+1)+'fold_train_evidence.dataset')
        smile, fp, sb, sa, variant, Y = np.asarray(list(flatten_folds_train[i]['smile'])),np.asarray(list(flatten_folds_train[i]['fingerprint'])), \
                                                        np.asarray(flatten_folds_train[i]['seqbefore']), np.asarray(flatten_folds_train[i]['seqafter']), \
                                                        np.asarray(list(flatten_folds_train[i]['variantfeature'])),np.asarray(list(flatten_folds_train[i]['label']))
        # make data PyTorch Geometric ready
        print('preparing '+str(i+1)+'fold_train.pt in pytorch format!')
        fold_data = PyDataset(root='../datasets/fivefold', dataset=str(i+1)+'fold_train', xs=smile, xfp=fp, xsb=sb, xsa=sa, xv=variant, y=Y, smile_graph=smile_graph)
        print('created')

        flatten_folds_valid[i].to_pickle('../datasets/fivefold/'+str(i+1)+'fold_valid_evidence.dataset')
        smile, fp, sb, sa, variant, Y = np.asarray(list(flatten_folds_valid[i]['smile'])),np.asarray(list(flatten_folds_valid[i]['fingerprint'])), \
                                                        np.asarray(flatten_folds_valid[i]['seqbefore']), np.asarray(flatten_folds_valid[i]['seqafter']), \
                                                        np.asarray(list(flatten_folds_valid[i]['variantfeature'])),np.asarray(list(flatten_folds_valid[i]['label']))
        # make data PyTorch Geometric ready
        print('preparing '+str(i+1)+'fold_valid.pt in pytorch format!')
        fold_data = PyDataset(root='../datasets/fivefold', dataset=str(i+1)+'fold_valid', xs=smile, xfp=fp, xsb=sb, xsa=sa, xv=variant, y=Y, smile_graph=smile_graph)
        print('created')
        

def generate_dataset():
    # split and flatten evidence
    evidence_data_file_train = '../datasets/middlefile/train_data_evidence.dataset'
    evidence_data_file_test = '../datasets/middlefile/test_data_evidence.dataset'
    if ((not os.path.isfile(evidence_data_file_train)) or (not os.path.isfile(evidence_data_file_test))):
        feature_dataset_path = '../datasets/dataset_featurecode.dataset'
        train_data, test_data = split_dataset(feature_dataset_path)
        train_data.to_pickle(evidence_data_file_train)
        test_data.to_pickle(evidence_data_file_test)

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
        
        train_smile, train_fp, train_sb, train_sa, train_variant,  train_Y = np.asarray(list(df['smile'])),np.asarray(list(df['fingerprint'])), \
                                                        np.asarray(df['seqbefore']), np.asarray(df['seqafter']), \
                                                        np.asarray(list(df['variantfeature'])),np.asarray(list(df['label']))
        df = pd.read_pickle(evidence_data_file_test)
        test_smile, test_fp, test_sb, test_sa, test_variant,  test_Y = np.asarray(list(df['smile'])),np.asarray(list(df['fingerprint'])), \
                                                        np.asarray(df['seqbefore']), np.asarray(df['seqafter']), \
                                                        np.asarray(list(df['variantfeature'])),np.asarray(list(df['label']))

        # make data PyTorch Geometric ready
        print('preparing train_data.pt in pytorch format!')
        train_data = PyDataset(root='../datasets', dataset='train', xs=train_smile, xfp=train_fp, xsb=train_sb, xsa=train_sa, xv=train_variant, y=train_Y, smile_graph=smile_graph)
        print('preparing test_data.pt in pytorch format!')
        test_data = PyDataset(root='../datasets', dataset='test', xs=test_smile, xfp=test_fp, xsb=test_sb, xsa=test_sa, xv=test_variant, y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have already existed') 

def generate_all():
    feature_dataset_path = '../datasets/dataset_featurecode.dataset'
    all_data = alldata(feature_dataset_path)
    evidence_data_file_all = '../datasets/middlefile/all_evidence.dataset'
    all_data.to_pickle(evidence_data_file_all)

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

    processed_data_file_all = '../datasets/processed/all_data.pt'
    if ((not os.path.isfile(processed_data_file_all))):
        df = pd.read_pickle(evidence_data_file_all)
        
        train_smile, train_fp, train_sb, train_sa, train_variant,  train_Y = np.asarray(list(df['smile'])),np.asarray(list(df['fingerprint'])), \
                                                        np.asarray(df['seqbefore']), np.asarray(df['seqafter']), \
                                                        np.asarray(list(df['variantfeature'])),np.asarray(list(df['label']))
        # make data PyTorch Geometric ready
        print('preparing all_data.pt in pytorch format!')
        train_data = PyDataset(root='../datasets', dataset='all', xs=train_smile, xfp=train_fp, xsb=train_sb, xsa=train_sa, xv=train_variant, y=train_Y, smile_graph=smile_graph)
        print(processed_data_file_all, ' have been created')        
    else:
        print(processed_data_file_all, ' have already existed') 
 
def data_augmentation():
    pass

def create_onehot(filepath,filename,targetpath,rootpath):
    datasetfile = pd.read_pickle(filepath+filename)

    # smile graph encoding
    compound_iso_smiles = list(datasetfile['smile'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    # print(smile_graph)

    processed_data_file = targetpath + filename.split('.')[0] + '_onehot_data.pt'
    if ((not os.path.isfile(processed_data_file))):
        new_label = []
        for i in range(datasetfile.shape[0]):
            new_label.append(label_dict[datasetfile['label'][i]])

        train_smile, train_fp, train_sb, train_sa, train_variant,  train_Y = np.asarray(list(datasetfile['smile'])),np.asarray(list(datasetfile['fingerprint'])), \
                                                        np.asarray(datasetfile['seqbefore']), np.asarray(datasetfile['seqafter']), \
                                                        np.asarray(list(datasetfile['variantfeature'])),np.asarray(new_label)
        # make data PyTorch Geometric ready
        print('preparing all_data.pt in pytorch format!')
        train_data = PyDataset(root=rootpath, dataset=filename.split('.')[0]+'_onehot', xs=train_smile, xfp=train_fp, xsb=train_sb, xsa=train_sa, xv=train_variant, y=train_Y, smile_graph=smile_graph)

if __name__=='__main__':

    generate_dataset() # prepare dataset (train/test)
    generate_5fold_dataset() # prepare dataset (5fold-train/test)
    generate_all()

    ''' # if need onehot labels
    create_onehot('../datasets/middlefile/','all_evidence.dataset','../datasets/processed/','../datasets')
    create_onehot('../datasets/middlefile/','test_data_evidence.dataset','../datasets/processed/','../datasets')
    create_onehot('../datasets/middlefile/','train_data_evidence.dataset','../datasets/processed/','../datasets')

    for i in range(5):
        create_onehot('../datasets/fivefold/',str(i+1)+'fold_train_evidence.dataset','../datasets/fivefold/processed/','../datasets/fivefold')
        create_onehot('../datasets/fivefold/',str(i+1)+'fold_valid_evidence.dataset','../datasets/fivefold/processed/','../datasets/fivefold')

    '''

    #data_augmentation() # prepare data augmentation dataset (5fold-train/test)
    