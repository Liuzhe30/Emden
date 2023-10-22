# -*- coding;utf-8 -*-
"""
File name : general_test.PY
create: 19/09/2023 14:51
Last modified: 19/09/2023 14:51
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
Version: 0.1
"""



from Bio import ExPASy
from Bio import SwissProt

uniprot_id = "P12345"
with ExPASy.get_sprot_raw(uniprot_id) as handle:
    record = SwissProt.read(handle)
print("UniProt ID:", record.entry_name)
print("蛋白质序列:")
print(record.sequence)

