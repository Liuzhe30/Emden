import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('fasta_full_before', type=str)
parser.add_argument('mu_pos', type=int)
parser.add_argument('mu_pos_after', type=str)
parser.add_argument('timestamp', type=str)

args = parser.parse_args()


warnings.filterwarnings('ignore')

eyes = np.eye(20)
protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
    'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
    'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
    'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}
fingerprint_dict = {'1':1,'0':0}
request_no = args.timestamp

fasta_full_before = args.fasta_full_before
mu_pos = args.mu_pos
mu_pos_after = args.mu_pos_after
fasta_full_after = copy.deepcopy(fasta_full_before)
fasta_full_after = fasta_full_after[:mu_pos-1] + mu_pos_after + fasta_full_after[mu_pos:]

outputfile = open('../datasets/middlefile/seq_' + str(request_no) + '_before.fasta', 'w')
w_content = '>seq_' + str(request_no) + '\n' + fasta_full_before
outputfile.write(w_content)
outputfile.close()
outputfile = open('../datasets/middlefile/seq_' + str(request_no) + '_after.fasta', 'w')
w_content = '>seq_' + str(request_no) + '\n' + fasta_full_after
outputfile.write(w_content)
outputfile.close()

os.system('hhblits -i ../datasets/middlefile/seq_' + str(request_no) + '_before.fasta -d /data/emden/scop70_1.75/scop70_1.75 -ohhm ../datasets/middlefile/hhm/' + str(request_no) + '_before.hhm')
os.system('hhblits -i ../datasets/middlefile/seq_' + str(request_no) + '_after.fasta -d /data/emden/scop70_1.75/scop70_1.75 -ohhm ../datasets/middlefile/hhm/' + str(request_no) + '_after.hhm')


# feature coding & dataset generate
window_len = 30

# update (mannuly remove conflicts 2023.3.14)
civic_drug_table = pd.read_pickle('../datasets/middlefile/civic_drug_table_fpfixed_smilenum' + str(request_no) + '.pkl')
dataset_feature = pd.DataFrame(columns=['smile', 'smile_num' ,'cactvs_fingerprint', 'molecular_weight', # drug features

                                        'onehot_before', 'onehot_after', 'hhm_before', 'hhm_after'# sequence features
                                        ])

smile = civic_drug_table['smile']
smile_array = np.zeros(shape=(85,))
# drug features from file
smile_num = civic_drug_table.loc[0]['smile_array']
for j in range(min(85, smile_num.shape[0])):
    smile_array[j] = smile_num[j]
cactvs_fingerprint_str = civic_drug_table.loc[0]['cactvs_fingerprint']
molecular_weight = civic_drug_table.loc[0]['molecular_weight']
cactvs_fingerprint = []
for strr in cactvs_fingerprint_str:
    cactvs_fingerprint.append(fingerprint_dict[strr])
cactvs_fingerprint = np.array(cactvs_fingerprint)
# sequence features from file
pos = mu_pos
pos_after = mu_pos_after
# fasta before
seq_len = len(fasta_full_before)
onehot_before = []
if(pos <= window_len - 1):
    for j in range(window_len-pos+1): # padding head
        onehot_before.append(np.zeros(20))
    for strr in fasta_full_before[0:pos+window_len]:
        onehot_before.append(protein_dict[strr])
elif(seq_len - pos < window_len):
    for strr in fasta_full_before[pos-window_len-1:]:
        onehot_before.append(protein_dict[strr])
    for j in range(window_len-seq_len+pos): # padding end
        onehot_before.append(np.zeros(20))
else:
    for strr in fasta_full_before[pos-window_len-1:pos+window_len]:
        onehot_before.append(protein_dict[strr])
onehot_before = np.array(onehot_before)
# fasta after
#fasta_after = fasta_full[pos-window_len-1:pos+window_len]
onehot_after = []
if(pos <= window_len - 1):
    for j in range(window_len-pos+1): # padding head
        onehot_after.append(np.zeros(20))
    for strr in fasta_full_after[0:pos+window_len]:
        onehot_after.append(protein_dict[strr])
elif(seq_len - pos < window_len):
    for strr in fasta_full_after[pos-window_len-1:]:
        onehot_after.append(protein_dict[strr])
    for j in range(window_len-seq_len+pos): # padding end
        onehot_after.append(np.zeros(20))
else:
    for strr in fasta_full_after[pos-window_len-1:pos+window_len]:
        onehot_after.append(protein_dict[strr])
fasta_len = len(fasta_full_after)
onehot_after = np.array(onehot_after)
# hhm before
with open('../datasets/middlefile/hhm/' + str(request_no) + '_before.hhm') as hhm_file:
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
                hhm_matrix[idxx - 1, j] = int(each_item[j])
                #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))
        elif(len(hhm_line.split()) == 10):
            each_item = hhm_line.split()[0:10]
            for idx, s in enumerate(each_item):
                if(s == '*'):
                    each_item[idx] = '99999'
            for j in range(20, 30):
                hhm_matrix[idxx - 1, j] = int(each_item[j - 20])
                #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j - 20])/2000))
        hhm_line = hhm_file.readline()
    if(pos <= window_len - 1):
        padding = np.zeros(shape=[window_len-pos+1,30])
        hhm_before_array = np.vstack((padding,hhm_matrix[0:pos+window_len, :]))
    elif(seq_len - pos < window_len):
        padding = np.zeros(shape=[window_len-seq_len+pos,30])
        hhm_before_array = np.vstack((hhm_matrix[pos-window_len-1:, :],padding))
    else:
        hhm_before_array = hhm_matrix[pos-window_len-1:pos+window_len, :]
    #hhm_before = hhm_before_array.tolist()
# hhm after
with open('../datasets/middlefile/hhm/' + str(request_no) + '_after.hhm') as hhm_file:
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
                hhm_matrix[idxx - 1, j] = int(each_item[j])
                #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))
        elif(len(hhm_line.split()) == 10):
            each_item = hhm_line.split()[0:10]
            for idx, s in enumerate(each_item):
                if(s == '*'):
                    each_item[idx] = '99999'
            for j in range(20, 30):
                hhm_matrix[idxx - 1, j] = int(each_item[j - 20])
                #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j - 20])/2000))
        hhm_line = hhm_file.readline()
    if(pos <= window_len - 1):
        padding = np.zeros(shape=[window_len-pos+1,30])
        hhm_after_array = np.vstack((padding,hhm_matrix[0:pos+window_len, :]))
    elif(seq_len - pos < window_len):
        padding = np.zeros(shape=[window_len-seq_len+pos,30])
        hhm_after_array = np.vstack((hhm_matrix[pos-window_len-1:, :],padding))
    else:
        hhm_after_array = hhm_matrix[pos-window_len-1:pos+window_len, :]
    #hhm_after = hhm_after_array.tolist()

dataset_feature = dataset_feature.append([{
                                    'smile': smile, 'smile_num':smile_array, 'cactvs_fingerprint': cactvs_fingerprint, 'molecular_weight': molecular_weight,
                                    'onehot_before':onehot_before, 'onehot_after':onehot_after,
                                    'hhm_before': hhm_before_array, 'hhm_after': hhm_after_array,
                                    }], ignore_index=True)

filtered_dataset_feature = dataset_feature.copy()
#filtered_dataset_feature = dataset_feature[~(dataset_feature['smile'] == "[Pt]")]
filtered_dataset_feature['smile'] = filtered_dataset_feature['smile'].apply(lambda x: x.replace('[Pt]', '[Pt]=[Pt]'))
filtered_dataset_feature = filtered_dataset_feature.reset_index(drop=True)
#dataset_feature.to_csv('../datasets/dataset_featurecode.csv', index=None)
filtered_dataset_feature.to_pickle('../datasets/dataset_featurecode' + str(request_no) + '.dataset')