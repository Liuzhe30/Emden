# -*- coding;utf-8 -*-
"""
File name : general.PY
create: 30/04/2023 21:31
Last modified: 30/04/2023 21:31
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
Version: 0.1
"""

import numpy as np

# a = np.load('/data/emden/Emden-web/model/pred_results/ori/pred_probd1a18527-02cb-47e1-9d36-ff8f60fba884.npy')


# def softmax(vec):
#     """Compute the softmax in a numerically stable way."""
#     vec = vec - np.max(vec)  # softmax(x) = softmax(x+c)
#     exp_x = np.exp(vec)
#     softmax_x = exp_x / np.sum(exp_x)
#     return softmax_x
#
#
#
# print(softmax(np.array([-17.176311, 23.558409])))
# import copy
#
# fasta_full_before = 'dgffdgdgfdgfdgdfgfdgdfg'
# mu_pos = 3
# mu_pos_after = 'k'
# fasta_full_after = copy.deepcopy(fasta_full_before)
# fasta_full_after = fasta_full_after[:mu_pos-1] + mu_pos_after + fasta_full_after[mu_pos:]
#

# import torch
#
# data, slices = torch.load('/data/emden/Emden-web/datasets/processed/testb263191a-0a19-4fde-9ce9-47453d91282e_data.pt')
#

# import pandas as pd
#
# a = pd.read_pickle('/data/emden/Emden-web/datasets/dataset_featurecode.dataset')
# a = a.loc[[97]]
# a = a.reset_index(drop=True)
# a = a.drop( columns= ['gene','uniprotac', 'variant', 'drug', 'label', 'source'])
# a.to_pickle('/data/emden/Emden-web/datasets/dataset_featurecode1.dataset')
# b = pd.read_pickle('/data/emden/Emden-web/datasets/dataset_featurecode123.dataset')



a = np.array([1.])
b = float(a)


