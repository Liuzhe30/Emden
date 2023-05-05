import os
import numpy as np
from math import sqrt
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch

class PyDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='train', 
                 xs=None, xfp=None, xsb=None, xsa=None, xv=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(PyDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'train'
        self.dataset = dataset
        print(self.processed_paths[0])
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xs, xfp, xsb, xsa, xv, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '_data.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-variant response prediction
    # Inputs:
    # xs - list of SMILES, xfp: list of fingerprint, xv: variants features 
    # y: list of labels
    # Return: PyTorch-Geometric format processed data
    def process(self, xs, xfp, xsb, xsa, xv, smile_graph):
        assert (len(xs) == len(xfp) and len(xs) == len(xv)), "The six lists must be the same length!"
        data_list = []
        data_len = len(xs)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xs[i]
            fingerprint = xfp[i]
            seqbefore = xsb[i]
            seqafter = xsa[i]
            variant = xv[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                )
            GCNData.fingerprint = torch.LongTensor([fingerprint])
            GCNData.seqbefore = torch.LongTensor([seqbefore])
            GCNData.seqafter = torch.LongTensor([seqafter])
            GCNData.variant = torch.LongTensor([variant])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print(data_list)
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    return sqrt(((y - f)**2).mean(axis=0))

def mse(y,f):
    return ((y - f)**2).mean(axis=0)

def cee(y,t): # cross_entropy_error
    delta = 1e-7  
    return -np.sum(t * np.log(y + delta))

def get_acc(outputs, labels):
    
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num

    return acc