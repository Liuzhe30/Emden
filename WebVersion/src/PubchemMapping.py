# -*- coding;utf-8 -*-
"""
File name : PubchemMapping.PY
create: 30/04/2023 16:31
Last modified: 30/04/2023 16:31
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
Version: 0.1
"""
import warnings
warnings.filterwarnings('ignore')
import pubchempy as pcp
import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('timestamp', type=str)
parser.add_argument('drugname', type=str)
args = parser.parse_args()

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

# drugname and request no
request_no = args.timestamp
drugname = args.drugname
drugname = drugname.strip()

new_drug_table = pd.DataFrame(columns=['drugname', 'smile', 'molecular_weight', 'molecular_formula', 'atom', 'fingerprint', 'cactvs_fingerprint'])


compound = pcp.get_compounds(drugname,'name')[0]
try:
    smile = compound.isomeric_smiles
except AttributeError:
    smile = np.nan
try:
    molecular_weight = compound.molecular_weight
except AttributeError:
    molecular_weight = np.nan
try:
    molecular_formula = compound.molecular_formula
except AttributeError:
    molecular_formula = np.nan
try:
    atom = compound.atoms
except AttributeError:
    atom = np.nan
try:
    fingerprint = compound.fingerprint
except AttributeError:
    fingerprint = np.nan
try:
    cactvs_fingerprint = compound.cactvs_fingerprint
except AttributeError:
    cactvs_fingerprint = np.nan
new_drug_table = new_drug_table.append([{'drugname':drugname, 'smile':smile, 'molecular_weight':molecular_weight, 'molecular_formula':molecular_formula,
                                'atom':atom, 'fingerprint':fingerprint, 'cactvs_fingerprint':cactvs_fingerprint}], ignore_index=True)

new_drug_table['smile_array'] = 0
new_drug_table['smile_array'] = new_drug_table['smile_array'].astype(object)
for i in range(new_drug_table.shape[0]):
    drugname = new_drug_table['drugname'][i]
    smile = new_drug_table['smile'][i]
    smile_num_list = []
    for strr in smile:
        smile_num_list.append(CHARISOSMISET[strr])
    #smile_num = np.array(smile_num_list)
    #new_drug_table['smile_array'][i] = new_drug_table['smile_array'][i].apply(lambda x: smile_num_list)
    new_drug_table.loc[:,'smile_array'].loc[i] = np.array(smile_num_list)
new_drug_table.to_pickle('../datasets/middlefile/civic_drug_table_fpfixed_smilenum' + str(request_no) + '.pkl')
