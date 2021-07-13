### python -W ignore $masif_source/data_preparation/02-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1 -p f
### python -W ignore 02-pdb_extract_and_triangulate.py $PDB1_ID $CHAIN1 $PDB2_ID $CHAIN2
# 1,2 - uX, 3,4 - C

#!/usr/bin/python
import sys
import os
import numpy as np
import glob
import subprocess
import shutil
import pathlib
from default_config.masif_opts import masif_opts
import data_preparation.extract_and_triangulate_lib as ext_and_trg
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply

import my_utils as my_u
import mylib as my

args = sys.argv[1:]
argc = len(args)
if(not argc in [1, 2]):
	print('Usage:\n' + sys.argv[0] + '   pdbid1   [to_recompute ((1)/0)]')
	sys.exit(1)

pdb_filebase = args[0]
to_recompute = ((args[1] == '1') if (argc > 1) else True)
print(to_recompute)

pdb_filename = pdb_filebase + '.pdb'
u_chain_filepath_base = os.path.join(my.git_root_path(), 'data', 'linear_avg', pdb_filebase)
if(os.path.isdir(u_chain_filepath_base)):
	if(to_recompute):
		shutil.rmtree(u_chain_filepath_base)
	else:
		print('PDB was already processed. Set to_recompure=1 and rerun to owerwrite')
		sys.exit(1)
os.makedirs(u_chain_filepath_base)
u_chain_filepath = os.path.join(u_chain_filepath_base, pdb_filebase + '.pdb')
shutil.move(u_chain_filepath_base + '.pdb', u_chain_filepath)
u_chain_filepath_base = os.path.join(u_chain_filepath_base, pdb_filebase)
#ply_filepath = os.path.join(masif_opts['ply_chain_dir'], u_chain_filepath_base + '.ply')
ply_filepath = u_chain_filepath_base + '.ply'

# construct unbound the mesh.
u_regular_mesh, u_vertex_normals, u_vertices, u_names = \
    ext_and_trg.msms_wrap(u_chain_filepath, to_recompute=to_recompute)
u_vertex_hbond, u_vertex_hphobicity, u_vertex_charges = \
    ext_and_trg.compute_features(u_chain_filepath_base, u_vertices, u_names, u_regular_mesh, to_recompute=to_recompute)

# save results
save_ply(ply_filepath, u_regular_mesh.vertices,\
         u_regular_mesh.faces, normals=u_vertex_normals, charges=u_vertex_charges,\
         normalize_charges=True, hbond=u_vertex_hbond, hphob=u_vertex_hphobicity)
