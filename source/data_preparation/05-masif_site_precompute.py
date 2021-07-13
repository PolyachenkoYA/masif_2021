import sys
import time
import os
import numpy as np
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)

# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts
from data_preparation.extract_and_triangulate_lib import parse_names
np.random.seed(0)

# Load training data (From many files)
from masif_modules.read_data_from_surface_my import read_data_from_surface, compute_shape_complementarity

# =============== input parse ==================
args = sys.argv[1:]
argc = len(args)
if(not argc in [1]):
    print("Usage:\n", sys.argv[0], '   PDBID_chain')
    sys.exit(1)

ppi_pair_id = args[0]
[u_pdb_name, u_chain_name] = ppi_pair_id.split('_')

# =========== paths handle ============
params = masif_opts['site']
params['ply_chain_dir'] = masif_opts['ply_chain_dir']
my_precomp_dir = params['masif_precomputation_dir']+ppi_pair_id+'/'
if not os.path.exists(my_precomp_dir):
    os.makedirs(my_precomp_dir)
    
# Read directly from the ply file.
fields = ppi_pair_id.split('_')
ply_filepath = masif_opts['ply_file_template'].format(u_pdb_name, u_chain_name)
    
# ================= computation =============
# Compute shape complementarity between the two proteins. 
input_feat, rho, theta, mask, neigh_indices, iface_labels, verts  = \
    read_data_from_surface(ply_filepath, params)

# Save data only if everything went well. 
np.save(my_precomp_dir+'p1_rho_wrt_center', rho)
np.save(my_precomp_dir+'p1_theta_wrt_center', theta)
np.save(my_precomp_dir+'p1_input_feat', input_feat)
np.save(my_precomp_dir+'p1_mask', mask)
np.save(my_precomp_dir+'p1_list_indices', neigh_indices)
np.save(my_precomp_dir+'p1_iface_labels', iface_labels)
# Save x, y, z
np.save(my_precomp_dir+'p1_X.npy', verts[:,0])
np.save(my_precomp_dir+'p1_Y.npy', verts[:,1])
np.save(my_precomp_dir+'p1_Z.npy', verts[:,2])
print('all written to the "', my_precomp_dir, '"')
