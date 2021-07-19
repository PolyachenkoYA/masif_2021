### python -W ignore $masif_source/data_preparation/02-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1 -p f
### python -W ignore 02-pdb_extract_and_triangulate.py $PDB1_ID $CHAIN1 $PDB2_ID $CHAIN2
# 1,2 - uX, 3,4 - C

#!/usr/bin/python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
#import glob
#import subprocess
import shutil
#import pathlib
import pymesh
#from default_config.masif_opts import masif_opts
import data_preparation.extract_and_triangulate_lib as ext_and_trg
#from input_output.extractPDB import extractPDB
#from input_output.save_ply import save_ply
from  triangulation.make_surface import make_mesh
import scipy
import scipy.stats

import mdtraj as md

#import my_utils as my_u
import mylib as my

center_coords_names = ['CA centers', 'center-of-mass positions']
features_names = ['charges', 'hbond', 'hphobicity']
main_mode = 1
N_center_types = len(center_coords_names)
N_feat = len(features_names)

### =============================== parse input ==============================
yes_flags = ['y', 'yes', '1']
no_flags = ['n', 'no', '0']
yn_flgs = yes_flags + no_flags
[pdb_filebase, Ravg, to_recompute, to_save_ply, to_draw_centers, to_draw_atoms, to_draw_vertices, to_plot_features], _ = \
	my.parse_args(sys.argv[1:], ['-file', '-R', '-recompute', '-save_ply_mesh', '-draw_centers', '-draw_atoms', '-draw_vertices', '-plot_features'], \
				   possible_values=[None, None, yn_flgs, yn_flgs, yn_flgs, yn_flgs, yn_flgs, yn_flgs], \
				   possible_arg_numbers=[[1], [1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], \
				   default_values=[None, None, [yes_flags[0]], [no_flags[0]], [yes_flags[0]], [yes_flags[0]], [yes_flags[0]], [yes_flags[0]]])
Ravg = float(Ravg)
to_recompute = (to_recompute[0] in yes_flags)
to_save_ply = (to_save_ply[0] in yes_flags)
to_draw_centers = (to_draw_centers[0] in yes_flags)
to_draw_atoms = (to_draw_atoms[0] in yes_flags)
to_draw_vertices = (to_draw_vertices[0] in yes_flags)

### =================== make the dirs ======================
pdb_filename = pdb_filebase + '.pdb'
pdb_filepath_base = os.path.join(my.git_root_path(), 'data', 'linear_avg', pdb_filebase)
pdb_filepath = os.path.join(pdb_filepath_base, pdb_filebase + '.pdb')
if(os.path.isdir(pdb_filepath_base)):
	if(to_recompute):
		shutil.rmtree(pdb_filepath_base)
	else:
		print('PDB was already processed. Searching for "' + pdb_filepath + '"')
if(to_recompute or (not os.path.isdir(pdb_filepath_base))):
	os.makedirs(pdb_filepath_base)
	shutil.copy(pdb_filepath_base + '.pdb', pdb_filepath)
pdb_filepath_base = os.path.join(pdb_filepath_base, pdb_filebase)
ply_filepath = pdb_filepath_base + '.ply'

### ======================== load molecule ============================
molecule = md.load_pdb(pdb_filepath)
N_resd = molecule.topology.n_residues
center_coords = np.zeros((N_center_types, N_resd, 3))
center_coords[0, :, :] = molecule.xyz[0, molecule.topology.select('name CA'), :]   # C_A coords
for res_i in range(N_resd):
	center_coords[1, res_i, :] = np.mean(molecule.xyz[0, molecule.topology.select('resid == ' + str(res_i)), :], axis=0)   # center of mass coords

### ==================== construct the mesh.===================
regular_mesh, vertex_normals, vertices, names = \
    ext_and_trg.msms_wrap(pdb_filepath, to_recompute=to_recompute)
vertex_hbond, vertex_hphobicity, vertex_charges = \
    ext_and_trg.compute_features(pdb_filepath_base, vertices, names, regular_mesh, to_recompute=to_recompute)
vertices = regular_mesh.vertices / 10   # looks like MaSIF pipeline works in (A) but not in (nm)

### ================== draw protein ======================
if(to_draw_centers or to_draw_atoms or to_draw_vertices):
	fig_protein, ax_protein = my.get_fig('x (nm)', 'y (nm)', title='centers', projection='3d', zlbl='z (nm)')
	for center_type_i in range(center_coords.shape[0]):
		if(to_draw_centers):
			ax_protein.plot3D(center_coords[center_type_i, :, 0], center_coords[center_type_i, :, 1], center_coords[center_type_i, :, 2], \
					 label='C:' + center_coords_names[center_type_i])
		if(to_draw_atoms):
			ax_protein.scatter3D(molecule.xyz[0, :, 0], molecule.xyz[0, :, 1], molecule.xyz[0, :, 2], \
			      label='A:' + center_coords_names[center_type_i], s=1)
		if(to_draw_vertices):
	#		ax_protein.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], \
#			      label='M:' + center_coords_names[center_type_i], s=1)
			#ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=regular_mesh.faces, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.0, shade=False)
			ax_protein.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=regular_mesh.faces, alpha=0.1)

### ================== comp sequences ======================
def get_R_cor(vertices, center_coords, vertices_features, Ravg, main_mode):
	N_center_types = center_coords.shape[0]
	N_resd = center_coords.shape[1]
	N_feat = vertices_features.shape[0]
	lin_feauture_sq = np.zeros((N_center_types, N_resd, N_feat))
	lin_feauture_sq_u = np.zeros((N_center_types, N_resd, N_feat))
	for center_type_i in range(N_center_types):
		for res_i in range(N_resd):
			crds = vertices - center_coords[center_type_i, res_i, :]
			dists = np.linalg.norm(crds, axis=1)
			w = np.exp( - np.power(dists, 2) / (2 * Ravg**2))
			#w = np.heaviside(Ravg - dists, 0)  # Ravg = 0.43 is optimal here
			for f_i in range(N_feat):
				lin_feauture_sq[center_type_i, res_i, f_i] = np.sum(vertices_features[f_i, :] * w)
			# =========== uniform sphere ===========
# 			close_atoms_inds = (np.linalg.norm(crds, axis=1) < Ravg)
# 			if(np.any(close_atoms_inds)):
# 				for f_i in range(N_feat):
# 					lin_feauture_sq[center_type_i, res_i, f_i] = np.mean(vertices_features[f_i, close_atoms_inds])
# 			else:
# 				print('ERROR: no atoms inside R = ' + str(Ravg) + \
# 				      ' for the AA (' + str(res_i) + ',' + str(center_type_i) + \
# 				      ') at ' + str(center_coords[center_type_i, res_i]))
# 				return None, None, None, [None, None, None]
# 				#sys.exit(1)
		for f_i in range(N_feat):
			lin_feauture_sq_u[center_type_i, :, f_i] = my.unitize(lin_feauture_sq[center_type_i, :, f_i])

	R_cor = np.zeros((N_feat, N_feat))
	for f1_i in range(N_feat):
		R_cor[f1_i, f1_i] = 1
		for f2_i in range(f1_i + 1, N_feat):
			#print(scipy.stats.pearsonr(lin_feauture_sq_u[main_mode, :, f1_i], lin_feauture_sq_u[main_mode, :, f2_i]))
			R_cor[f1_i, f2_i], _ = scipy.stats.pearsonr(lin_feauture_sq_u[main_mode, :, f1_i], lin_feauture_sq_u[main_mode, :, f2_i])
			R_cor[f2_i, f1_i] = R_cor[f1_i, f2_i]

	return lin_feauture_sq_u, R_cor

vertices_features = np.concatenate((vertex_charges[np.newaxis, :], vertex_hbond[np.newaxis, :], vertex_hphobicity[np.newaxis, :]))
lin_feauture_sq_u, _ = \
	get_R_cor(vertices, center_coords, vertices_features, Ravg, main_mode)

### =============================== get R ===================================
N_R = 100
Rmin = 0.1
Rmax = 4
R_log_base = 1.1
R_cor = np.zeros((N_R, N_feat, N_feat))
R_arr = np.power(R_log_base, np.linspace(0, np.log(Rmax / Rmin) / np.log(R_log_base), N_R)) * Rmin
for R_i in range(N_R):
	_, R_cor[R_i, :, :] = get_R_cor(vertices, center_coords, vertices_features, R_arr[R_i], main_mode)
	#print('R cor done: ' + str((R_i + 1) / N_R * 100) + ' %')

fig_R_avg_cor, ax_R_avg_cor = my.get_fig('$\sigma_{avg}$ (nm)', 'cor', title='$cor(\sigma_{avg})$', xscl='log')
for f1_i in range(N_feat):
	for f2_i in range(f1_i + 1, N_feat):
		ax_R_avg_cor.plot(R_arr, R_cor[:, f1_i, f2_i], label = features_names[f1_i] + ':' + features_names[f2_i])
ax_R_avg_cor.plot([Ravg] * 2, [np.min(R_cor.flatten()), np.max(R_cor.flatten())], '--', label='$R_{main}$')
fig_R_avg_cor.legend()

if(to_draw_centers or to_draw_atoms or to_draw_vertices):
	fig_protein.legend()

if(to_plot_features):
	fig_feat, ax_feat = my.get_fig('sq #', 'feature', title='features along the sequence; $\sigma_{avg} = ' + str(Ravg) + '$')
	sq_nums = range(N_resd)
	for f_i in range(N_feat):
		ax_feat.plot(sq_nums, lin_feauture_sq_u[main_mode, :, f_i], '--o', label=features_names[f_i])
	fig_feat.legend()

### ================= comp continous features ===============
#backbone_crds = molecule.xyz[0, molecule.topology.select('backbone'), :]   # backbone_coords
#molecule.compute_gyration_tensor

if(to_save_ply):
	# ======================== save results ===========================
	mesh = make_mesh(regular_mesh.vertices,\
					  regular_mesh.faces, normals=vertex_normals, charges=vertex_charges,\
					  normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
	pymesh.save_mesh(ply_filepath, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)

if(to_draw_centers):
	plt.show()