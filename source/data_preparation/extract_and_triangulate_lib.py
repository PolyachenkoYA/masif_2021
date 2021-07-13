#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
import subprocess
import networkx as nx
from IPython.core.debugger import set_trace

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree
from geometry.vertices_graph import vertices_graph

import my_utils as my

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

def msms_wrap(chain_filepath, verbose=True, to_recompute=False):
    if(verbose):
        print('performing MSMS on', chain_filepath)
    vertices, faces, normals, names, areas = computeMSMS(chain_filepath, protonate=True, to_recompute=to_recompute)
    mesh = pymesh.form_mesh(vertices, faces)
    regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
    vertex_normals = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    return regular_mesh, vertex_normals, vertices, names

def compute_features(chain_filepath_base, vertices, names, regular_mesh, verbose=True, to_recompute=False):
    if(verbose):
        print('computing features for "', chain_filepath_base, '"')
    vertex_hbond = computeCharges(chain_filepath_base, vertices, names) if masif_opts['use_hbond'] else None
    vertex_hphobicity = computeHydrophobicity(names) if masif_opts['use_hphob'] else None
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices, vertex_hbond, masif_opts) if masif_opts['use_hbond'] else None
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices, vertex_hphobicity, masif_opts) if masif_opts['use_hphob'] else None
    vertex_charges = computeAPBS(regular_mesh.vertices, chain_filepath_base + '.pdb', chain_filepath_base) if masif_opts['use_apbs'] else None
    return vertex_hbond, vertex_hphobicity, vertex_charges

def filter_noise(iface_v, mesh, noise_patch_size=-50, verbose=True):
    """
    Filter small false-positive ground truth iface vertices which were marked due to fluctuating sidechains
    
    noise_patch_size = N:
    N > 0: delete all isolated components of the initial ground-truth which have len <= N 
    N == 0: do nothing, i.e. leave iface as it was received from the vertices(d >= d0) selection
    N == -1: delete everything except the biggest component. If there are > 1 biggest components with the same size, then delete everything except them and say a Warning.
    N == -2: delete everything except components with len > max(components_sizes) // 5
    N < -2: delete everything except components with len > max(max(components_sizes) // 5, -N)
    """
    if(verbose):
        print('filtering false-positives ground-truth with the noise_patch_size = ', noise_patch_size)
    iface = np.zeros(len(mesh.vertices))
    
    if(noise_patch_size == 0):
        true_iface_v = iface_v
    else:                      
        G = vertices_graph(mesh, weighted=False)
        
        N_verts = len(mesh.vertices)
        not_iface_v = []
        for v in range(N_verts):
            if(not v in iface_v):
                not_iface_v.append(v)
        not_iface_v = np.array(not_iface_v)

        G.remove_nodes_from(not_iface_v)
        
        iface_components = [np.array(list(c)) for c in nx.connected_components(G)]
        iface_components_N = len(iface_components)
        components_sizes = [len(c) for c in iface_components]
        true_iface_components = []        
        if(noise_patch_size == -1):
            max_component_ind = np.argmax(components_sizes)
            max_size = components_sizes[max_component_ind]
            true_iface_components.append(iface_components[max_component_ind])
            for i in range(max_component_ind + 1, iface_components_N):
                if(components_sizes[i] == max_size):
                    true_iface_components.append(iface_components[i])
            N_big_components = len(true_iface_components)
            
            if(N_big_components > 1):
                print('Warning:\niface components ' + str(true_iface_components) + ' (' + str(N_big_components) + ' items) have the same size = ' + str(max_size) + ' and they all are the biggest ones. Using their union as a ground truth.')

        else:
            if(noise_patch_size == -2):
                noise_patch_size = max(components_sizes) // 5
            elif(noise_patch_size < -2):
                noise_patch_size = max(max(components_sizes) // 5, -noise_patch_size)
                
            for i, G_c in enumerate(iface_components):
                if(components_sizes[i] > noise_patch_size):
                    true_iface_components.append(G_c)
                    
        true_iface_v = np.concatenate(true_iface_components)

    iface[true_iface_v] = 1.0
    return iface
            
def find_iface(C_mesh, u_mesh, ground_truth_cut_dist, verbose=True):
    if(verbose):
        print('determining iface for the iface_d_cut = ', ground_truth_cut_dist)
    # Find the vertices that are in the iface.
    # Find the distance between every vertex in u_regular_mesh.vertices and those in the full complex.
    kdt = KDTree(C_mesh.vertices)
    d, r = kdt.query(u_mesh.vertices)
    d = np.square(d) # Square d, because this is how it was in the pyflann version.
    assert(len(d) == len(u_mesh.vertices))
    iface_v = np.where(d >= ground_truth_cut_dist)[0]
        
    iface = filter_noise(iface_v, u_mesh, noise_patch_size=-1)
    
    return iface
