import os
import sys
import subprocess as sp
import numpy as np

import my_utils as my
import data_preparation.extract_and_triangulate_lib as ext_and_trg

args = sys.argv[1:]
argc = len(args)

# ====== find the best d_cut for filtering the false-positive ground-truth =========
#if(not argc in [0]):
#    print('usage:\n' + sys.argv[0])
#d_arr = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0, 3.5, 4.0]
#for d in d_arr:
#    sp.run(['./run_full_u.sh 1Z0K-uR-1000_A 1Z0K-C-36457_A ' + str(d)], shell=True)


# ======================= compute AUC(step) =====================
step = 50
N_steps = 50100
chain_names = {0:'A', 1:'B', 2:'C', 3:'D'}

if(not argc in [2, 3]):
    print('usage:\n' + sys.argv[0] + '   PDBid   uR/uL   [step (N_steps + 1)]')
    exit(1)
pdb_id = args[0]
md_id = args[1]
step =  (N_steps * 2 + 1) if(argc < 3) else int(args[2])

chain_id_in_C = my.chain_ids_table[pdb_id][md_id]
model_name = pdb_id + '-' + md_id
ROCAUC_filename = model_name + '.dat'
ROCAUC_filepath = os.path.join(ext_and_trg.pdbs_dir, ROCAUC_filename)
if(os.path.isfile(ROCAUC_filepath)):
    os.remove(ROCAUC_filepath)

for i in range(48400, N_steps + 1, step):
    i_str = str(i)
    u_name = model_name + '-' + i_str + '_A'
    C_name = model_name + '-C-' + i_str + '_' + chain_names[chain_id_in_C]
    my.run_it('./run_full_u.sh ' + u_name + ' ' + C_name + ' ' + ROCAUC_filename)
