#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pymesh
import trimesh
import scipy
import scipy.stats
import bisect
import shapely.geometry

#from default_config.masif_opts import masif_opts
import data_preparation.extract_and_triangulate_lib as ext_and_trg
#from input_output.extractPDB import extractPDB
#from input_output.save_ply import save_ply
import triangulation
import triangulation.meshcut as meshcut
from  triangulation.make_surface import make_mesh

import mdtraj as md

#import my_utils as my_u
import mylib as my

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
	lin_feauture_sq = np.zeros((N_resd, N_feat))
	lin_feauture_sq_u = np.zeros((N_resd, N_feat))
	for res_i in range(N_resd):
		crds = vertices - center_coords[res_i, :]
		dists = np.linalg.norm(crds, axis=1)
		w = np.exp( - np.power(dists, 2) / (2 * Ravg**2))   # Ravg=0.23 is optimal here
		#w = np.heaviside(Ravg - dists, 0)  # Ravg = 0.43 is optimal here
		for f_i in range(N_feat):
			lin_feauture_sq[res_i, f_i] = np.sum(vertices_features[f_i, :] * w)

	for f_i in range(N_feat):
		lin_feauture_sq_u[:, f_i] = my.unitize(lin_feauture_sq[:, f_i])

	return lin_feauture_sq_u

def get_R_cor_sq(features):
	# compute pairwise R-corellation
	N_feat = features.shape[1]
	R_cor = np.zeros((N_feat, N_feat))
	for f1_i in range(N_feat):
		R_cor[f1_i, f1_i] = 1
		for f2_i in range(f1_i + 1, N_feat):
			#print(scipy.stats.pearsonr(lin_feauture_sq_u[main_mode, :, f1_i], lin_feauture_sq_u[main_mode, :, f2_i]))
			R_cor[f1_i, f2_i], _ = scipy.stats.pearsonr(features[:, f1_i], features[:, f2_i])
			R_cor[f2_i, f1_i] = R_cor[f1_i, f2_i]

	return R_cor

def main():
	center_coords_names = ['CA centers', 'center-of-mass positions', 'backbone']
	main_mode = 0
	#N_center_types = len(center_coords_names)

	N_interp_scale = 5
	N_R = 20  # log space
	R_log_base = 1.1
	Rmin = 0.1  # nm
	Rmax = 1  # nm

	features_names = ['charge', '$H_{bond}$', '$H_{phobicity}$', '$S$', '$S_{conv}$']
	N_feat = len(features_names)
	chain_vec_avg = 1

	### =============================== parse input ==============================
	yes_flags = ['y', 'yes', '1']
	no_flags = ['n', 'no', '0']
	yn_flgs = yes_flags + no_flags
	[pdb_filebase, Ravg, to_recompute, to_save_ply, to_draw_centers, to_draw_atoms, to_draw_vertices, to_plot_features, id_sections_to_draw, to_draw_2D_sections], _ = \
		my.parse_args(sys.argv[1:], ['-file', '-R', '-recompute', '-save_ply_mesh', '-draw_centers', '-draw_atoms', '-draw_vertices', '-plot_features', '-section_to_draw', '-draw_2D_sections'], \
					   possible_values=[None, None, yn_flgs, yn_flgs, yn_flgs, yn_flgs, yn_flgs, yn_flgs, None, yn_flgs], \
					   possible_arg_numbers=[[1], [1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], None, [0, 1]], \
					   default_values=[None, None, [yes_flags[0]], [no_flags[0]], [yes_flags[0]], [yes_flags[0]], [yes_flags[0]], [yes_flags[0]], None, [yes_flags[0]]])
	Ravg = float(Ravg)
	to_recompute = (to_recompute[0] in yes_flags)
	to_save_ply = (to_save_ply[0] in yes_flags)
	to_draw_centers = (to_draw_centers[0] in yes_flags)
	to_draw_atoms = (to_draw_atoms[0] in yes_flags)
	to_draw_vertices = (to_draw_vertices[0] in yes_flags)
	if(id_sections_to_draw is not None):
		id_sections_to_draw = [int(s) for s in id_sections_to_draw]
	to_draw_2D_sections = (to_draw_2D_sections[0] in yes_flags)
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
	if(main_mode == 0):
			center_coords_residues = molecule.xyz[0, molecule.topology.select('name CA'), :]   # C_A coords
	elif(main_mode == 1):
			center_coords_residues = np.zeros((N_resd, 3))   # center of mass coords
			for res_i in range(N_resd):   # center of mass coords
				center_coords_residues[res_i, :] = np.mean(molecule.xyz[0, molecule.topology.select('resid == ' + str(res_i)), :], axis=0)   # center of mass coords
	elif(main_mode == 2):
			center_coords_residues = molecule.xyz[0, molecule.topology.select('backbone'), :]   # backbone_coords
	### =================== construct paths =======================
	N_centers = N_resd * N_interp_scale
	center_coords, center_resd_indices, chain_vectors = \
		interpolate_3Dpoints(center_coords_residues, N_centers)

	### ==================== construct the mesh and compute features.===================
	regular_mesh, vertex_normals, vertices, names = \
	    ext_and_trg.msms_wrap(pdb_filepath, to_recompute=to_recompute)
	vertex_hbond, vertex_hphobicity, vertex_charges = \
	    ext_and_trg.compute_features(pdb_filepath_base, vertices, names, regular_mesh, to_recompute=to_recompute)
	vertices = regular_mesh.vertices / 10   # looks like MaSIF pipeline works in (A) but not in (nm)

	### ================== comp features ======================
	vertices_features = np.concatenate((vertex_charges[np.newaxis, :], vertex_hbond[np.newaxis, :], vertex_hphobicity[np.newaxis, :]))
	lin_vertices_feautures = get_lin_features_from_vertices(vertices, center_coords, vertices_features, Ravg)

	main_sections_areas = np.zeros(N_centers)
	main_sections_convex_areas = np.zeros(N_centers)
	sections_3D = []
	sections_2D = []
	main_subsections_ids = - np.ones(N_centers, dtype=int)
	center_crds_2D = np.zeros((2, N_centers))
	sections_to_3D = np.zeros((N_centers, 4, 4))
	section_normals = np.zeros((3, N_centers))
	regular_trimesh = trimesh.Trimesh(vertices=vertices, faces=regular_mesh.faces)
	cut_mesh = meshcut.TriangleMesh(vertices, regular_mesh.faces)
	for cent_i in range(N_centers):
		chain_vectors_to_average = chain_vectors[max(center_resd_indices[cent_i] - 1 - chain_vec_avg, 0) : \
							                     min(center_resd_indices[cent_i] - 1 + chain_vec_avg, chain_vectors.shape[0] - 1), :]
		#section_normals[:, cent_i] = np.main(chain_vectors_to_average / np.linalg.norm(chain_vectors_to_average, axis=), axis=0)
		section_normals[:, cent_i] = np.sum(chain_vectors_to_average, axis=0)
		section = meshcut.cross_section_mesh(cut_mesh, meshcut.Plane(center_coords[cent_i, :], \
											                         section_normals[:, cent_i]))   # trimesh does not separate closed subsections, this does, tho it works much slover
		_, sections_to_3D[cent_i, :, :] = \
			regular_trimesh.section(plane_origin=center_coords[cent_i, :], \
							        plane_normal=section_normals[:, cent_i]).to_planar(check=False)

		center_crds_2D[:, cent_i] = np.linalg.solve(sections_to_3D[cent_i, :, :], \
											         np.append(center_coords[cent_i, :], 1).T)[0:2]
		sections_2D.append([])
		for subsection_id in range(len(section)):
			section[subsection_id] = section[subsection_id].T
			subsection_coord_augmented = np.concatenate((section[subsection_id], \
											             np.ones((1, section[subsection_id].shape[1]))), axis=0)
			subsection_2Dcoord = np.linalg.solve(sections_to_3D[cent_i, :, :], \
									             subsection_coord_augmented)[0:2, :]
			sections_2D[-1].append(subsection_2Dcoord)

			subsection_polygon = shapely.geometry.asPolygon(subsection_2Dcoord.T)
			if(subsection_polygon.contains(shapely.geometry.asPoint(center_crds_2D[:, cent_i]))):
				if(main_subsections_ids[cent_i] < 0):
					main_subsections_ids[cent_i] = subsection_id
					main_sections_areas[cent_i] = subsection_polygon.area
					main_sections_convex_areas[cent_i] = subsection_polygon.convex_hull.area
				else:
					print('ERROR: 2 polygons containing the center point were found. Aborting.')
					sys.exit(1)
		sections_3D.append(section)

	section_features = np.concatenate((my.unitize(main_sections_areas)[:, np.newaxis], my.unitize(main_sections_convex_areas)[:, np.newaxis]), axis=1)

	lin_feauture_sq_u = np.concatenate((lin_vertices_feautures, section_features), axis=1)

	### =============================== get R ===================================
	R_cor = np.zeros((N_R, N_feat, N_feat))
	R_arr = np.power(R_log_base, np.linspace(0, np.log(Rmax / Rmin) / np.log(R_log_base), N_R)) * Rmin
	for R_i in range(N_R):
		lin_feautures = get_lin_features_from_vertices(vertices, center_coords, vertices_features, R_arr[R_i])
		R_cor[R_i, :, :] = get_R_cor_sq(np.concatenate((lin_feautures, section_features), axis=1))
		#print('R cor done: ' + str((R_i + 1) / N_R * 100) + ' %')

	### ================== draw protein ======================
	if(to_draw_centers or to_draw_atoms or to_draw_vertices):
		fig_protein, ax_protein = my.get_fig('x (nm)', 'y (nm)', title='3D polymer', projection='3d', zlbl='z (nm)')
		if(to_draw_centers):
	 		ax_protein.plot3D(center_coords[:, 0], center_coords[:, 1], center_coords[:, 2], \
		                      label='C:' + center_coords_names[main_mode])
		if(to_draw_atoms):
			ax_protein.scatter3D(molecule.xyz[0, :, 0], molecule.xyz[0, :, 1], molecule.xyz[0, :, 2], \
			      label='A:' + center_coords_names[main_mode], s=1)
		if(to_draw_vertices):
			ax_protein.plot_trisurf(regular_trimesh.vertices[:,0], regular_trimesh.vertices[:,1], regular_trimesh.vertices[:,2], \
						            triangles=regular_trimesh.faces, alpha=0.5)
		if(id_sections_to_draw is not None):
			for section_id_i in range(len(id_sections_to_draw)):
				section_id = id_sections_to_draw[section_id_i]
				section_title = 'section ' + str(section_id)
				ax_protein.plot(center_coords[section_id, 0], center_coords[section_id, 1], center_coords[section_id, 2], \
					               '.', markersize=10, color=my.colors[section_id_i], label=None)

				if(to_draw_2D_sections):
					fig_section, ax_section = my.get_fig("x'", "y'", title=section_title)
					ax_section.scatter(center_crds_2D[0, section_id], center_crds_2D[1, section_id], marker='+', \
						               color='red', label=('chain' if subsection_id == 0 else None))

				for subsection_id in range(len(sections_3D[section_id])):
					subsection = sections_3D[section_id][subsection_id]
					ax_protein.plot(subsection[0, :], subsection[1, :], subsection[2, :], \
					               '+', markersize=4, label=(section_title if subsection_id == 0 else None), color=my.colors[section_id_i])

					if(to_draw_2D_sections):
						ax_section.plot(sections_2D[section_id][subsection_id][0, :], sections_2D[section_id][subsection_id][1, :], \
					                    '.-', markersize=2, label='subsec ' + str(subsection_id) + (('; S = ' + my.f2s(main_sections_areas[section_id]) + ' $nm^2$') if subsection_id == main_subsections_ids[section_id] else ''))

				if(to_draw_2D_sections):
					fig_section.legend()

	### ================== plot results ======================
	fig_R_avg_cor, ax_R_avg_cor = my.get_fig('$\sigma_{avg}$ (nm)', '$R_{cor}$', title='$cor(\sigma_{avg})$; avg along the "' + center_coords_names[main_mode] + '"', xscl='log')
	for f1_i in range(N_feat):
		for f2_i in range(f1_i + 1, N_feat):
			ax_R_avg_cor.plot(R_arr, R_cor[:, f1_i, f2_i], label = features_names[f1_i] + ':' + features_names[f2_i])
	ax_R_avg_cor.plot([Ravg] * 2, [np.min(R_cor.flatten()), np.max(R_cor.flatten())], '--', label='$R_{main}$')
	fig_R_avg_cor.legend()

	if(to_draw_centers or to_draw_atoms or to_draw_vertices):
		fig_protein.legend()

	if(to_plot_features):
		fig_feats, ax_feats = my.get_fig('residue #', 'feature', title='features along the "' + center_coords_names[main_mode] + '"; $\sigma_{avg} = ' + str(Ravg) + '$')
		for f_i in range(N_feat):
			ax_feats.scatter((np.arange(N_centers) + 1) * (N_resd / N_centers), lin_feauture_sq_u[:, f_i], s=2, label=features_names[f_i])
		ax_feats.plot([0, N_resd], [0, 0], label=None)
		fig_feats.legend()

# 		for f_i in range(N_feat):
# 			fig_feat, ax_feat = my.get_fig('residue #', features_names[f_i], title='features along the "' + center_coords_names[main_mode] + '"; $\sigma_{avg} = ' + str(Ravg) + '$')
# 			fig_feat.legend()

	### ======================== save results ===========================
	if(to_save_ply):
		mesh = make_mesh(regular_mesh.vertices,\
						  regular_mesh.faces, normals=vertex_normals, charges=vertex_charges,\
						  normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
		pymesh.save_mesh(ply_filepath, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)

	if(to_draw_centers):
		plt.show()

if __name__ == "__main__":
    main()
