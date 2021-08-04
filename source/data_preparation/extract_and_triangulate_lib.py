#!/usr/bin/python
import numpy as np
import os
import sys
import shutil
import matplotlib.pyplot as plt
import pymesh
import trimesh
import scipy.stats
import bisect
import shapely.geometry

#import networkx as nx
#from IPython.core.debugger import set_trace
#from sklearn.neighbors import KDTree
#import Bio
#from Bio.PDB import *

import mdtraj as md

# Local includes
from default_config.masif_opts import masif_opts
#from input_output.extractPDB import extractPDB
#from input_output.save_ply import save_ply
#from input_output.read_ply import read_ply
#from input_output.protonate import protonate
#from geometry.vertices_graph import vertices_graph
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
import triangulation.meshcut as meshcut
from  triangulation.make_surface import make_mesh


# my includes
#import my_utils as my
import mylib as my

pdbs_dir = os.path.join(my.user_home_path, 'ppi_traj', 'PDBS')

def copy_tmp2dst(src_file, dst_dir, verbose=True):
    dst_file = os.path.join(dst_dir, os.path.basename(src_file))
    if(os.path.abspath(src_file) != os.path.abspath(dst_file)):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(src_file, dst_file)
        if(verbose):
            print(dst_file + ' written')
    else:
        if(verbose):
            print(dst_file + ' already exist')

def parse_names(arg, tmp_dir=None, git_root=my.git_root_path(), verbose=True):
    chain_filename_base = arg
    [pdb_name, chain_name] = chain_filename_base.split('_')
    pdb_filename = pdb_name + '.pdb'
    if(verbose):
        print('parsing names for "', pdb_filename, '", chain "', chain_name, '"')
    #raw_pdb_path = os.path.join(git_root, 'data', 'masif_site', masif_opts['raw_pdb_dir'])
    #raw_pdb_path = os.path.join(pdbs_dir, pdb_name[:4])
    raw_pdb_path = os.path.join(pdbs_dir, pdb_name)
    pdb_filepath = os.path.join(raw_pdb_path, pdb_filename)
    chain_filepath_base = os.path.join(masif_opts["tmp_dir"] if tmp_dir==None else tmp_dir, chain_filename_base)
    chain_filepath = chain_filepath_base + '.pdb'
    return pdb_name, chain_name, pdb_filepath, chain_filepath_base, chain_filepath

def msms_wrap(chain_filepath, verbose=True):
    if(verbose):
        print('performing MSMS on', chain_filepath)
    vertices, faces, normals, names, areas = computeMSMS(chain_filepath, have_xyzrn=True)
    mesh = pymesh.form_mesh(vertices, faces)
    regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
    vertex_normals = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    return regular_mesh, vertex_normals, vertices, names

def compute_surface_features(chain_filepath_base, vertices, names, regular_mesh, verbose=True):
    if(verbose):
        print('computing features for "', chain_filepath_base, '"')
    vertex_hbond = computeCharges(chain_filepath_base, vertices, names) if masif_opts['use_hbond'] else None
    vertex_hphobicity = computeHydrophobicity(names) if masif_opts['use_hphob'] else None
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices, vertex_hbond, masif_opts) if masif_opts['use_hbond'] else None
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices, vertex_hphobicity, masif_opts) if masif_opts['use_hphob'] else None
    vertex_charges = computeAPBS(regular_mesh.vertices, chain_filepath_base + '.pdb', chain_filepath_base) if masif_opts['use_apbs'] else None
    return vertex_hbond, vertex_hphobicity, vertex_charges

def interpolate_3Dpoints(crds, N):
	# put N equally spaced points into a polygonal chain
	polygonal_vectors = crds[1:, :] - crds[:-1, :]
	stick_dists = np.linalg.norm(polygonal_vectors, axis=1)
	N_resd = crds.shape[0]
	stick_cumul_dists = np.zeros(N_resd)
	stick_cumul_dists[0] = 0
	for res_i in range(0, N_resd- 1):
		stick_cumul_dists[res_i + 1] = stick_cumul_dists[res_i] + stick_dists[res_i]
	new_centers_dists = np.arange(N) * (stick_cumul_dists[-1] / N)
	interp_coords = np.zeros((N, 3))
	insert_indices = np.zeros(N, int)
	for k in range(N):
		insert_index = bisect.bisect(stick_cumul_dists, new_centers_dists[k])
		insert_indices[k] = insert_index
		interp_coords[k, :] = (crds[insert_index, :] * (new_centers_dists[k] - stick_cumul_dists[insert_index - 1]) + \
						       crds[insert_index - 1, :] * (stick_cumul_dists[insert_index] - new_centers_dists[k])) / \
			                   (stick_cumul_dists[insert_index] - stick_cumul_dists[insert_index - 1])   # m is never 0 because all dists are > 0 and Lk[0] = 0
	return interp_coords, insert_indices, polygonal_vectors

def get_lin_features_from_vertices(vertices, center_coords, vertices_features, Ravg):
	# compute features averaged from vertices around the points in the center_coords
	N_resd = center_coords.shape[0]
	N_feat = vertices_features.shape[0]
	lin_features = np.zeros((N_resd, N_feat))
	for res_i in range(N_resd):
		crds = vertices - center_coords[res_i, :]
		dists = np.linalg.norm(crds, axis=1)
		w = np.exp( - np.power(dists, 2) / (2 * Ravg**2))   # Ravg=0.23 is optimal here
		#w = np.heaviside(Ravg - dists, 0)  # Ravg = 0.43 is optimal here
		for f_i in range(N_feat):
			lin_features[res_i, f_i] = np.sum(vertices_features[f_i, :] * w) / np.sum(w)

	return lin_features

def comp_R_cor(features, to_normalize=True):
	# compute pairwise R-corellation
	N_feat = features.shape[1]
	R_cor = np.zeros((N_feat, N_feat))
	for f1_i in range(N_feat):
		R_cor[f1_i, f1_i] = 1
		for f2_i in range(f1_i + 1, N_feat):
			f1 = features[:, f1_i]
			f2 = features[:, f2_i]
			if(to_normalize):
				f1 = my.unitize(f1)
				f2 = my.unitize(f2)
			R_cor[f1_i, f2_i], _ = scipy.stats.pearsonr(f1, f2)
			R_cor[f2_i, f1_i] = R_cor[f1_i, f2_i]

	return R_cor

def get_R_cor(vertices, center_coords, vertices_features, section_features, N_R=20, Rmax=1, Rmin=0.1, R_log_base=1.1):
	N_feat = vertices_features.shape[0] + section_features.shape[1]
	R_cor = np.zeros((N_R, N_feat, N_feat))
	R_arr = np.power(R_log_base, np.linspace(0, np.log(Rmax / Rmin) / np.log(R_log_base), N_R)) * Rmin
	for R_i in range(N_R):
		R_cor[R_i, :, :] = comp_R_cor(np.concatenate((get_lin_features_from_vertices(vertices, center_coords, vertices_features, R_arr[R_i]), \
												        section_features), axis=1))
			#print('R cor done: ' + str((R_i + 1) / N_R * 100) + ' %')

	return R_cor, R_arr

def get_lin_features(pdb_filepath, Ravg, to_comp_cor=False, main_mode=0, N_centers=200, chain_vec_avg=1, \
					 to_recompute=False, to_save_ply=False, to_draw_centers=False, \
					 to_draw_atoms=False, to_draw_vertices=False,  to_plot_features=False, \
					 id_sections_to_draw=None, to_draw_2D_sections=False, to_plot_normalized_features=False, \
					 to_plot_Rcor=False):
	### ===================== hardcode params =========================
	center_coords_names = ['CA centers', 'center-of-mass positions', 'backbone']
	#features_names = ['charge', '$H_{bond}$', '$H_{phob}$', '$S$ ($nm^2$)', '$S_{conv}$ ($nm^2$)', '$S_{conv} / S$']
	features_names = ['charge', '$H_{bond}$', '$H_{phob}$']
	N_feat = len(features_names)
	chain_vec_avg = 1   # average 2n+1 heighbour chain elements to get a normal for sections

	### ======================== paths ==================================
	pdb_filepath_base = os.path.splitext(pdb_filepath)[0]
	ply_filepath = pdb_filepath_base + '.ply'
	features_filepath = pdb_filepath_base + '_'.join(['_features', \
													 'R' + str(Ravg), \
													 'Ninterp' + str(N_centers), \
													 'chainAvg' + str(chain_vec_avg), \
													 'mode' + str(main_mode)]) + '.npy'

	molecule = md.load_pdb(pdb_filepath)

	### =================== construct the chain =======================
	N_resd = molecule.topology.n_residues
	if(main_mode == 0):
			center_coords_residues = molecule.xyz[0, molecule.topology.select('name CA'), :]   # C_A coords
	elif(main_mode == 1):
			center_coords_residues = np.zeros((N_resd, 3))   # center of mass coords
			for res_i in range(N_resd):   # center of mass coords
				center_coords_residues[res_i, :] = \
					np.mean(molecule.xyz[0, molecule.topology.select('resid == ' + str(res_i)), :], axis=0)   # center of mass coords
	elif(main_mode == 2):
			center_coords_residues = molecule.xyz[0, molecule.topology.select('backbone'), :]   # backbone_coords
	center_coords, center_resd_indices, chain_vectors = \
		interpolate_3Dpoints(center_coords_residues, N_centers)

	### ==================== construct the mesh.===================
	regular_mesh, vertex_normals, vertices, names = msms_wrap(pdb_filepath)

	### ================== comp surface features ======================
	to_draw_sections = (id_sections_to_draw is not None)
	to_draw_anything = to_draw_centers or to_draw_atoms or to_draw_vertices or to_draw_sections \
	                   or to_plot_Rcor or to_plot_normalized_features or to_plot_features

	if(to_recompute or (not os.path.isfile(features_filepath)) or to_draw_anything):
		vertex_hbond, vertex_hphobicity, vertex_charges = \
				compute_surface_features(pdb_filepath_base, vertices, names, regular_mesh)
		vertices = regular_mesh.vertices / 10   # looks like MaSIF pipeline works in (A) but not in (nm)

		vertices_features = np.concatenate((vertex_charges[np.newaxis, :], \
										    vertex_hbond[np.newaxis, :], \
											vertex_hphobicity[np.newaxis, :]))

		lin_vertices_feautures = get_lin_features_from_vertices(vertices, center_coords, vertices_features, Ravg)

		### ======================== save surface ===========================
		if(to_save_ply):
			mesh = make_mesh(regular_mesh.vertices,\
							  regular_mesh.faces, normals=vertex_normals, charges=vertex_charges,\
							  normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
			pymesh.save_mesh(ply_filepath, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)

		### ================== comp section features ======================
		#main_sections_areas = np.zeros(N_centers)
		#main_sections_convex_areas = np.zeros(N_centers)
		#sections_3D = []
		#sections_2D = []
		#main_subsections_ids = - np.ones(N_centers, dtype=int)
		center_crds_2D = np.zeros((2, N_centers))
		sections_to_3D = np.zeros((N_centers, 4, 4))
		regular_trimesh = trimesh.Trimesh(vertices=vertices, faces=regular_mesh.faces)
		if(not regular_trimesh.is_watertight):
			print('ERROR: not watertight surface mesh')
		cut_mesh = meshcut.TriangleMesh(vertices, regular_mesh.faces)
		#print('pymesh: ', regular_mesh.faces.shape)
		#print('trimesh: ', regular_trimesh.faces.shape)
		#print('cutmesh: ', cut_mesh.tris.shape)
		section_normals = np.zeros((3, N_centers))
		for cent_i in range(N_centers):
			chain_vectors_to_average = chain_vectors[max(center_resd_indices[cent_i] - 1 - chain_vec_avg, 0) : \
								                     min(center_resd_indices[cent_i] - 1 + chain_vec_avg, chain_vectors.shape[0] - 1), :]
			#section_normals[:, cent_i] = np.main(chain_vectors_to_average / np.linalg.norm(chain_vectors_to_average, axis=), axis=0)
			section_normals[:, cent_i] = np.sum(chain_vectors_to_average, axis=0)
# 			section = meshcut.cross_section_mesh(cut_mesh, meshcut.Plane(center_coords[cent_i, :], \
# 												                         section_normals[:, cent_i]))   # trimesh does not separate closed subsections, this does, tho it works much slover
			_, sections_to_3D[cent_i, :, :] = \
				regular_trimesh.section(plane_origin=center_coords[cent_i, :], \
								        plane_normal=section_normals[:, cent_i]).to_planar(check=False)

			center_crds_2D[:, cent_i] = np.linalg.solve(sections_to_3D[cent_i, :, :], \
												         np.append(center_coords[cent_i, :], 1).T)[0:2]
# 			sections_2D.append([])
# 			for subsection_id in range(len(section)):
# 				section[subsection_id] = section[subsection_id].T
# 				subsection_coord_augmented = np.concatenate((section[subsection_id], \
# 												             np.ones((1, section[subsection_id].shape[1]))), axis=0)
# 				subsection_2Dcoord = np.linalg.solve(sections_to_3D[cent_i, :, :], \
# 										             subsection_coord_augmented)[0:2, :]
# 				sections_2D[-1].append(subsection_2Dcoord)

# 				subsection_polygon = shapely.geometry.asPolygon(subsection_2Dcoord.T)
# 				if(subsection_polygon.contains(shapely.geometry.asPoint(center_crds_2D[:, cent_i]))):
# 					if(main_subsections_ids[cent_i] < 0):
# 						main_subsections_ids[cent_i] = subsection_id
# 						main_sections_areas[cent_i] = subsection_polygon.area
# 						main_sections_convex_areas[cent_i] = subsection_polygon.convex_hull.area
# 					else:
# 						print('ERROR: 2 polygons containing the center point were found. Aborting.')
# 						sys.exit(1)
# 			sections_3D.append(section)

			print('sections comp: ' + my.f2s((cent_i + 1) / N_centers * 100) + ' %               \r', end='')

# 		section_features = np.concatenate((main_sections_areas[:, np.newaxis], \
# 										   main_sections_convex_areas[:, np.newaxis], \
# 										   (main_sections_convex_areas / main_sections_areas)[:, np.newaxis]), axis=1)

		### ================== join all features ======================
		#lin_features = np.concatenate((lin_vertices_feautures, section_features), axis=1)
		lin_features = lin_vertices_feautures

		print('saving features to "' + features_filepath + '"')
		with open(features_filepath, 'wb') as f:
			np.save(f, lin_features)
	else:
		print('loading features from "' + features_filepath + '"')
		with open(features_filepath, 'rb') as f:
			lin_features = np.load(f)

	### =================== get correlations ===========================
	if(to_comp_cor):
		pass
		#R_cor, R_arr = get_R_cor(vertices, center_coords, vertices_features, section_features)
	else:
		R_cor = None
		R_arr = None

	### ================== draw protein ======================
	if(to_draw_centers or to_draw_atoms or to_draw_vertices or to_draw_sections):
		fig_protein, ax_protein = my.get_fig('x (nm)', 'y (nm)', title='3D polymer', projection='3d', zlbl='z (nm)')
		if(to_draw_centers):
	 		ax_protein.plot3D(center_coords[:, 0], center_coords[:, 1], center_coords[:, 2], \
		                      label='C:' + center_coords_names[main_mode])
		if(to_draw_atoms):
			ax_protein.scatter3D(molecule.xyz[0, :, 0], molecule.xyz[0, :, 1], molecule.xyz[0, :, 2], \
			      label='A:' + center_coords_names[main_mode], s=1)
		if(to_draw_vertices):
			ax_protein.plot_trisurf(regular_trimesh.vertices[:,0], regular_trimesh.vertices[:,1], regular_trimesh.vertices[:,2], \
						            triangles=regular_trimesh.faces, alpha=0.1)

			#surfaces_to_mark = [1483, 1484, 1578, 1579]
			surfaces_to_mark = [1485, 1486, 1580, 1581]
			colors_m = ['red', 'green', 'blue', 'black']
# 			print(regular_trimesh.faces[surfaces_to_mark].shape)
# 			print(regular_trimesh.faces[np.newaxis, surfaces_to_mark[0]].shape)
# 			surf = ax_protein.plot_trisurf(regular_trimesh.vertices[:, 0], regular_trimesh.vertices[:, 1], regular_trimesh.vertices[:, 2], \
# 							        triangles=regular_trimesh.faces[surfaces_to_mark], alpha=1.0, color='blue')

			for i_s, s in enumerate(surfaces_to_mark):
				surf = ax_protein.plot_trisurf(regular_trimesh.vertices[:, 0], regular_trimesh.vertices[:, 1], regular_trimesh.vertices[:, 2], \
								        triangles=regular_trimesh.faces[np.newaxis, s], color=colors_m[i_s], label=str(regular_trimesh.faces[s]))
				surf._facecolors2d = surf._facecolor3d
				surf._edgecolors2d = surf._edgecolor3d

		if(to_draw_sections):
			for section_id_i in range(len(id_sections_to_draw)):
				section_id = id_sections_to_draw[section_id_i]
				section_title = 'section ' + str(section_id)
				ax_protein.plot(center_coords[section_id, 0], center_coords[section_id, 1], center_coords[section_id, 2], \
					               '.', markersize=10, color=my.colors[section_id_i], label=None)

				if(to_draw_2D_sections):
					fig_section, ax_section = my.get_fig("x'", "y'", title=section_title)
					ax_section.scatter(center_crds_2D[0, section_id], center_crds_2D[1, section_id], marker='+', \
						               color='red', label='chain')

# 				for subsection_id in range(len(sections_3D[section_id])):
# 					subsection = sections_3D[section_id][subsection_id]
# 					ax_protein.plot(subsection[0, :], subsection[1, :], subsection[2, :], \
# 					               '+', markersize=4, label=(section_title if subsection_id == 0 else None), color=my.colors[section_id_i])

# 					if(to_draw_2D_sections):
# 						ax_section.plot(sections_2D[section_id][subsection_id][0, :], sections_2D[section_id][subsection_id][1, :], \
# 					                    '.-', markersize=2, label='subsec ' + str(subsection_id) + \
# 											(('; $S_{conv}$ = ' + my.f2s(main_sections_convex_areas[section_id]) + ' $nm^2$') \
# 								             if subsection_id == main_subsections_ids[section_id] else ''))

				if(to_draw_2D_sections):
					fig_section.legend()

	if(to_draw_centers or to_draw_atoms or to_draw_vertices or to_draw_sections):
		fig_protein.legend()

	### ================== plot results ======================
	if(to_plot_Rcor):
		fig_R_avg_cor, ax_R_avg_cor = \
			my.get_fig('$\sigma_{avg}$ (nm)', '$R_{cor}$', title='$cor(\sigma_{avg})$; avg along the "' + center_coords_names[main_mode] + '"', xscl='log')
		for f1_i in range(N_feat):
			for f2_i in range(f1_i + 1, N_feat):
				ax_R_avg_cor.plot(R_arr, R_cor[:, f1_i, f2_i], label = features_names[f1_i] + ':' + features_names[f2_i])
		ax_R_avg_cor.plot([Ravg] * 2, [np.min(R_cor.flatten()), np.max(R_cor.flatten())], '--', label='$R_{main}$')
		fig_R_avg_cor.legend()

	if(to_plot_normalized_features):
		fig_feats, ax_feats = my.get_fig('residue #', 'feature', \
								   title='features along the "' + center_coords_names[main_mode] + '"; $\sigma_{avg} = ' + str(Ravg) + '$')
		residue_index_x = (np.arange(N_centers) + 1) * (N_resd / N_centers)
		for f_i in range(N_feat):
			ax_feats.plot(residue_index_x, my.unitize(lin_features[:, f_i]), '-.', label=features_names[f_i])
		ax_feats.plot([0, N_resd], [0, 0], label=None)
		fig_feats.legend()

	if(to_plot_features):
		residue_index_x = (np.arange(N_centers) + 1) * (N_resd / N_centers)
		residue_index_x = np.arange(N_centers)
		for f_i in range(N_feat):
			fig_feat, ax_feat = my.get_fig('center # ~ (residue #)x' + my.f2s(N_centers / N_resd), features_names[f_i], title=features_names[f_i])
			ax_feat.plot(residue_index_x, lin_features[:, f_i], '-o', markersize=2)
			#fig_feat.legend()

	if(to_draw_anything):
		plt.show()

	return lin_features, [R_cor, R_arr], features_names

### ======================== masif-specific stuff ==========================

# def filter_noise(iface_v, mesh, noise_patch_size=-50, verbose=True):
#     """
#     Filter small false-positive ground truth iface vertices which were marked due to fluctuating sidechains

#     noise_patch_size = N:
#     N > 0: delete all isolated components of the initial ground-truth which have len <= N
#     N == 0: do nothing, i.e. leave iface as it was received from the vertices(d >= d0) selection
#     N == -1: delete everything except the biggest component. If there are > 1 biggest components with the same size, then delete everything except them and say a Warning.
#     N == -2: delete everything except components with len > max(components_sizes) // 5
#     N < -2: delete everything except components with len > max(max(components_sizes) // 5, -N)
#     """
#     if(verbose):
#         print('filtering false-positives ground-truth with the noise_patch_size = ', noise_patch_size)
#     iface = np.zeros(len(mesh.vertices))

#     if(noise_patch_size == 0):
#         true_iface_v = iface_v
#     else:
#         G = vertices_graph(mesh, weighted=False)

#         N_verts = len(mesh.vertices)
#         not_iface_v = []
#         for v in range(N_verts):
#             if(not v in iface_v):
#                 not_iface_v.append(v)
#         not_iface_v = np.array(not_iface_v)

#         G.remove_nodes_from(not_iface_v)

#         iface_components = [np.array(list(c)) for c in nx.connected_components(G)]
#         iface_components_N = len(iface_components)
#         components_sizes = [len(c) for c in iface_components]
#         true_iface_components = []
#         if(noise_patch_size == -1):
#             max_component_ind = np.argmax(components_sizes)
#             max_size = components_sizes[max_component_ind]
#             true_iface_components.append(iface_components[max_component_ind])
#             for i in range(max_component_ind + 1, iface_components_N):
#                 if(components_sizes[i] == max_size):
#                     true_iface_components.append(iface_components[i])
#             N_big_components = len(true_iface_components)

#             if(N_big_components > 1):
#                 print('Warning:\niface components ' + str(true_iface_components) + ' (' + str(N_big_components) + ' items) have the same size = ' + str(max_size) + ' and they all are the biggest ones. Using their union as a ground truth.')

#         else:
#             if(noise_patch_size == -2):
#                 noise_patch_size = max(components_sizes) // 5
#             elif(noise_patch_size < -2):
#                 noise_patch_size = max(max(components_sizes) // 5, -noise_patch_size)

#             for i, G_c in enumerate(iface_components):
#                 if(components_sizes[i] > noise_patch_size):
#                     true_iface_components.append(G_c)

#         true_iface_v = np.concatenate(true_iface_components)

#     iface[true_iface_v] = 1.0
#     return iface

# def find_iface(C_mesh, u_mesh, ground_truth_cut_dist, verbose=True):
#     if(verbose):
#         print('determining iface for the iface_d_cut = ', ground_truth_cut_dist)
#     # Find the vertices that are in the iface.
#     # Find the distance between every vertex in u_regular_mesh.vertices and those in the full complex.
#     kdt = KDTree(C_mesh.vertices)
#     d, r = kdt.query(u_mesh.vertices)
#     d = np.square(d) # Square d, because this is how it was in the pyflann version.
#     assert(len(d) == len(u_mesh.vertices))
#     iface_v = np.where(d >= ground_truth_cut_dist)[0]

#     iface = filter_noise(iface_v, u_mesh, noise_patch_size=-1)

#     return iface
