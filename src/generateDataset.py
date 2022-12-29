import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

eyes = np.eye(20)
protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
    'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
    'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
    'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}
fingerprint_dict = {'1':1,'0':0}

# feature coding & dataset generate
window_len = 30

merged_dataset = pd.read_csv('../datasets/merged_evidence.csv')
civic_drug_table = pd.read_csv('../datasets/middlefile/civic_drug_table_fpfixed_smilenum.csv')
pharmgkb_drug_table = pd.read_csv('../datasets/middlefile/pharmgkb_drug_table_fpfixed_smilenum.csv')
dataset_feature = pd.DataFrame(columns=['gene', 'uniprotac', 'variant', 'drug', 
                                        'smile', 'smile_num' ,'cactvs_fingerprint', 'molecular_weight', # drug features
                                        'fasta_before', 'fasta_after', 'onehot_before', 'onehot_after', 'hhm_before', 'hhm_after', 'ss', 'rasa', # sequence features
                                        'label', 'source'])
for i in tqdm(range(merged_dataset.shape[0])):
    gene = merged_dataset['gene'][i]
    uniprotac = merged_dataset['uniprotac'][i]
    variant = merged_dataset['variant'][i]
    drug = merged_dataset['drug'][i]
    smile = merged_dataset['smile'][i]
    label = merged_dataset['label'][i]
    source = merged_dataset['source'][i]
    # drug features from file
    if(source == 'civic'):
        smile_num = civic_drug_table[civic_drug_table['drugname'] == drug]['smile_array'].values[0]
        cactvs_fingerprint_str = civic_drug_table[civic_drug_table['drugname'] == drug]['cactvs_fingerprint'].values[0]
        molecular_weight = civic_drug_table[civic_drug_table['drugname'] == drug]['molecular_weight'].values[0]
    else:
        smile_num = pharmgkb_drug_table[pharmgkb_drug_table['drugname'] == drug]['smile_array'].values[0]
        cactvs_fingerprint_str = pharmgkb_drug_table[pharmgkb_drug_table['drugname'] == drug]['cactvs_fingerprint'].values[0]
        molecular_weight = pharmgkb_drug_table[pharmgkb_drug_table['drugname'] == drug]['molecular_weight'].values[0]
    cactvs_fingerprint = []
    for strr in cactvs_fingerprint_str:
        cactvs_fingerprint.append(fingerprint_dict[strr])
    # sequence features from file
    pos = int(variant[1:-1])
    pos_after = variant[-1]
    # fasta before
    with open('../datasets/middlefile/fasta/' + uniprotac + '.fasta') as file:
        fasta_file = file.readlines()
        fasta_full = fasta_file[1]
        fasta_before = fasta_full[pos-window_len-1:pos+window_len]
        onehot_before = []
        for strr in fasta_before:
            onehot_before.append(protein_dict[strr])
    # fasta after
    with open('../datasets/middlefile/fasta/' + gene + '_' + variant + '.fasta') as file:
        fasta_file = file.readlines()
        fasta_full = fasta_file[1]
        fasta_after = fasta_full[pos-window_len-1:pos+window_len]
        onehot_after = []
        for strr in fasta_after:
            onehot_after.append(protein_dict[strr])
    fasta_len = len(fasta_full)
    # hhm before
    with open('../datasets/middlefile/hhm/' + uniprotac + '.hhm') as hhm_file:     
        hhm_matrix = np.zeros([fasta_len, 30], float)
        hhm_line = hhm_file.readline()
        idxx = 0
        while(hhm_line[0] != '#'):
            hhm_line = hhm_file.readline()
        for i in range(0,5):
            hhm_line = hhm_file.readline()
        while hhm_line:
            if(len(hhm_line.split()) == 23):
                idxx += 1
                if(idxx == fasta_len + 1):
                    break
                each_item = hhm_line.split()[2:22]
                for idx, s in enumerate(each_item):
                    if(s == '*'):
                        each_item[idx] = '99999'                            
                for j in range(0, 20):
                    try:
                        hhm_matrix[idxx - 1, j] = int(each_item[j])
                        #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))                                              
                    except IndexError:
                        pass
            elif(len(hhm_line.split()) == 10):
                each_item = hhm_line.split()[0:10]
                for idx, s in enumerate(each_item):
                    if(s == '*'):
                        each_item[idx] = '99999'                             
                for j in range(20, 30):
                    try:
                        hhm_matrix[idxx - 1, j] = int(each_item[j - 20]) 
                        #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j - 20])/2000))                                               
                    except IndexError:
                        pass                            
            hhm_line = hhm_file.readline()
        hhm_before_array = hhm_matrix[pos-window_len-1:pos+window_len, :]
        hhm_before = hhm_before_array.tolist()
    # hhm after
    with open('../datasets/middlefile/hhm/' + gene + '_' + variant + '.hhm') as hhm_file:     
        hhm_matrix = np.zeros([fasta_len, 30], float)
        hhm_line = hhm_file.readline()
        idxx = 0
        while(hhm_line[0] != '#'):
            hhm_line = hhm_file.readline()
        for i in range(0,5):
            hhm_line = hhm_file.readline()
        while hhm_line:
            if(len(hhm_line.split()) == 23):
                idxx += 1
                if(idxx == fasta_len + 1):
                    break
                each_item = hhm_line.split()[2:22]
                for idx, s in enumerate(each_item):
                    if(s == '*'):
                        each_item[idx] = '99999'                            
                for j in range(0, 20):
                    try:
                        hhm_matrix[idxx - 1, j] = int(each_item[j])
                        #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))                                              
                    except IndexError:
                        pass
            elif(len(hhm_line.split()) == 10):
                each_item = hhm_line.split()[0:10]
                for idx, s in enumerate(each_item):
                    if(s == '*'):
                        each_item[idx] = '99999'                             
                for j in range(20, 30):
                    try:
                        hhm_matrix[idxx - 1, j] = int(each_item[j - 20]) 
                        #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j - 20])/2000))                                               
                    except IndexError:
                        pass                            
            hhm_line = hhm_file.readline()
        hhm_after_array = hhm_matrix[pos-window_len-1:pos+window_len, :]
        hhm_after = hhm_after_array.tolist()
    # rasa
    rasa_array = np.loadtxt('../datasets/middlefile/rASA/' + uniprotac + '.rasa')
    rasa_array = rasa_array[pos-window_len-1:pos+window_len]
    rasa = rasa_array.tolist()
    # ss
    ss_array = np.loadtxt('../datasets/middlefile/SS/' + uniprotac + '.ss')
    ss_array = ss_array[pos-window_len-1:pos+window_len, :]
    ss = ss_array.tolist()
    dataset_feature = dataset_feature.append([{'gene': gene, 'uniprotac': uniprotac, 'variant': variant, 'drug': drug, 
                                        'smile': smile, 'smile_num':smile_num, 'cactvs_fingerprint': cactvs_fingerprint, 'molecular_weight': molecular_weight, 
                                        'fasta_before':fasta_before, 'fasta_after':fasta_after, 'onehot_before':onehot_before, 'onehot_after':onehot_after, 
                                        'hhm_before': hhm_before, 'hhm_after': hhm_after, 'ss': ss, 'rasa': rasa, 
                                        'label': label, 'source':source}], ignore_index=True)
dataset_feature.to_csv('../datasets/dataset_featurecode.csv', index=None)