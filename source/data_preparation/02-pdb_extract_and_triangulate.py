#!/usr/bin/python

import numpy as np
import sys
import os
import shutil

import data_preparation.extract_and_triangulate_lib as ext_and_trg

import mylib as my

def main():
	### =============================== parse input ==============================
	to_draw_default = (my.yes_flags[0] if 1 else my.no_flags[0])
	[pdb_filebase, Ravg, to_recompute, to_save_ply, to_draw_centers, to_draw_atoms, to_draw_vertices, to_plot_features, id_sections_to_draw, to_draw_2D_sections, to_plot_normalized_features, to_plot_Rcor, to_comp_corelations], _ = \
		my.parse_args(sys.argv[1:], ['-file', '-R', '-recompute_features', '-save_ply_mesh', '-draw_centers', '-draw_atoms', '-draw_vertices', '-plot_features', '-sections_to_draw', '-draw_2D_sections', '-plot_normalized_features', '-plot_correlations', '-comp_corelations'], \
					   possible_values=[None, None, None, my.yn_flgs, my.yn_flgs, my.yn_flgs, my.yn_flgs, my.yn_flgs, None, my.yn_flgs, my.yn_flgs, my.yn_flgs, my.yn_flgs], \
					   possible_arg_numbers=[[1], [1], range(ext_and_trg.N_feat + 1), [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], None, [0, 1], [0, 1], [0, 1], [0, 1]], \
					   default_values=[None, None, [], [to_draw_default], [to_draw_default], [to_draw_default], [to_draw_default], [to_draw_default], None, [to_draw_default], [to_draw_default], [to_draw_default], [my.yes_flags[0]]])
	Ravg = float(Ravg)
	to_recompute = [int(cf) for cf in to_recompute]
	to_save_ply = (to_save_ply[0] in my.yes_flags)
	to_draw_centers = (to_draw_centers[0] in my.yes_flags)
	to_draw_atoms = (to_draw_atoms[0] in my.yes_flags)
	to_draw_vertices = (to_draw_vertices[0] in my.yes_flags)
	to_plot_features = (to_plot_features[0] in my.yes_flags)
	if(id_sections_to_draw is not None):
		id_sections_to_draw = [int(s) for s in id_sections_to_draw]
	to_draw_2D_sections = (to_draw_2D_sections[0] in my.yes_flags)
	to_plot_normalized_features = (to_plot_normalized_features[0] in my.yes_flags)
	to_plot_Rcor = (to_plot_Rcor[0] in my.yes_flags)
	to_comp_corelations = (to_comp_corelations[0] in my.yes_flags)

	to_recomp_mesh = (-1 in to_recompute)
	to_recompute = np.array([(i in to_recompute) for i in range(ext_and_trg.N_feat + 1)])
	to_recompute[-1] = to_recomp_mesh

	if(not to_comp_corelations):
		to_plot_Rcor = False
		print('WARNING: you asked to plot correlations between features but said to not compute them; Rcor will not be plotted')

	### =================== make the dirs ======================
	pdb_filepath_base = os.path.join(my.git_root_path(), 'data', 'linear_avg', pdb_filebase)
	pdb_filepath = os.path.join(pdb_filepath_base, pdb_filebase + '.pdb')
	if(os.path.isdir(pdb_filepath_base)):
		if(to_recompute.any()):
			shutil.rmtree(pdb_filepath_base)
		else:
			print('PDB was already processed. Searching for "' + pdb_filepath + '"')
	if(to_recompute.any() or (not os.path.isdir(pdb_filepath_base))):
		os.makedirs(pdb_filepath_base)
		shutil.copy(pdb_filepath_base + '.pdb', pdb_filepath)

	### =================== compute & plot ======================
	ext_and_trg.get_lin_features(pdb_filepath, Ravg, to_comp_cor=to_comp_corelations, \
					 to_recompute=to_recompute, to_save_ply=to_save_ply, \
					 to_draw_centers=to_draw_centers,  to_draw_atoms=to_draw_atoms, \
					 to_draw_vertices=to_draw_vertices, to_plot_features=to_plot_features, \
					 id_sections_to_draw=id_sections_to_draw, to_draw_2D_sections=to_draw_2D_sections, \
					 to_plot_normalized_features=to_plot_normalized_features, \
					 to_plot_Rcor=to_plot_Rcor)

if __name__ == "__main__":
    main()
