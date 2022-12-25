# env: space_hhblits
import os
pdb_path = '../datasets/middlefile/AF2pdb/'
dssp_path = '../datasets/middlefile/dssp/'
path_files = os.listdir(pdb_path)  
name_list = []
for fi in path_files: 
    name = fi.split('.')[0]
    name_list.append(name)
print(len(name_list))

for i in name_list:
    os.system("mkdssp -i " + pdb_path + i + '.pdb -o ' + dssp_path + i + '.dssp')