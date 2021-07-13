import os
import sys
import subprocess as sp
import numpy as np

import my_utils as my

# ================= cycle params ==================
pdbs = list(my.chain_ids_table.keys())
step = 50
mds = ['uR', 'uL']

# ============== input parse ==============
args = sys.argv[1:]
argc = len(args)
pdbs_to_do = range(len(pdbs)) if(args[0] == 'all') else [int(i) for i in args]

# ================ cycle =====================
#for pdb_id in pdbs:
for pdb_i in pdbs_to_do:
    for md_id in mds:
        sp.run(['python', 'run_cycle.py', pdbs[pdb_i], md_id, str(step)])
            