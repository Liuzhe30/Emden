# -*- coding;utf-8 -*-
"""
File name : visualization.PY
create: 22/10/2023 12:13
Last modified: 22/10/2023 12:13
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
Version: 0.1
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
parser = argparse.ArgumentParser()
parser.add_argument('timestamp', type=str)
args = parser.parse_args()


request_no = args.timestamp


middle_output_trans_f = np.load('/data/emden/Emden-web/model/middle_output/trans_f_output' + str(request_no) + '.npy')
trans_fsum_array = np.sum(middle_output_trans_f, axis=0)
trans_fsum_array = np.sum(trans_fsum_array, axis=2)

plt.figure(figsize=(30, 7))
sns.heatmap(trans_fsum_array,cmap=sns.light_palette((210, 90, 60), input="husl"))
plt.yticks(ticks=[0],labels=[''])
plt.xlabel('Pubchem Fingerprint Index')
plt.ylabel('Sum of attention output')
plt.savefig('/data/emden/Emden/static/img/fingerprint-attention-vis' + str(request_no) + '.png')
