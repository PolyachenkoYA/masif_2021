import pymesh
import os
import sys
import shutil
from IPython.core.debugger import set_trace
from sklearn.metrics import roc_auc_score
import importlib
import numpy as np
from default_config.masif_opts import masif_opts

import data_preparation.extract_and_triangulate_lib as ext_and_trg

"""
masif_site_label_surface.py: Color a protein ply surface file by the MaSIF-site interface score.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

args = sys.argv[1:]
argc = len(args)
if(not argc in [3, 4]):
    print('usage:\n' + sys.argv[0] + '   params_model   PPI_id   ROCAUC_filename   [to_save_scores(1/(0))]')
to_save_scores = False if(argc < 4) else (args[3] == '1')
custom_params_file = args[0]
ppi_pair_id = args[1]
ROCAUC_filename = args[2]

params = masif_opts["site"]
custom_params = importlib.import_module(custom_params_file, package=None)
custom_params = custom_params.custom_params
ROCAUC_filepath = os.path.join(ext_and_trg.pdbs_dir, ROCAUC_filename)

for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]

# Shape precomputation dir.
#parent_in_dir = params["masif_precomputation_dir"]

[pdb_id, chain_name] = ppi_pair_id.split("_")
frame_id = pdb_id.split('-')[-1]
scrores_filepath = params["out_pred_dir"] + "/pred_" + ppi_pair_id + ".npy"
groundtruth_ply_filepath = masif_opts["ply_file_template"].format(pdb_id, chain_name)
    
try:
    mymesh = pymesh.load_mesh(groundtruth_ply_filepath)
except:
    print("File does not exist: {}".format(groundtruth_ply_filepath))
    exit(1)

try:
    scores = np.load(scrores_filepath)
except:
    print("File does not exist: {}".format(scrores_filepath))
    exit(1)

ground_truth = mymesh.get_attribute('vertex_iface')
main_patch_size = np.sum(ground_truth)
# Compute ROC AUC for this protein. 
try:
    roc_auc = roc_auc_score(ground_truth, scores[0])
    print("ROC AUC score for protein {} : {:.2f} ".format(ppi_pair_id, roc_auc))
    
    with open(ROCAUC_filepath, 'a') as rocauc_file:
        print(frame_id, roc_auc, main_patch_size, file=rocauc_file)
        print('ROCAUC = ' + str(roc_auc))    
except: 
    print("No ROC AUC computed for protein (possibly, no ground truth defined in input)") 
   
to_save_scores = True
if(to_save_scores):
    mymesh.remove_attribute("vertex_iface")
    mymesh.add_attribute("iface")
    mymesh.set_attribute("iface", scores[0])
    mymesh.remove_attribute("vertex_x")
    mymesh.remove_attribute("vertex_y")
    mymesh.remove_attribute("vertex_z")
    mymesh.remove_attribute("face_vertex_indices")

    if not os.path.exists(params["out_surf_dir"]):
        os.makedirs(params["out_surf_dir"])

    scores_ply_filepath = params["out_surf_dir"] + pdb_id + "-pred_" + chain_name + ".ply"
    pymesh.save_mesh(
            scores_ply_filepath,
            mymesh,
            *mymesh.get_attribute_names(),
            use_float=True,
            ascii=True
    )
    print("Successfully saved file " + scores_ply_filepath)

precomp_dir = os.path.join(params["masif_precomputation_dir"], ppi_pair_id)
#os.remove(groundtruth_ply_filepath)
#print(groundtruth_ply_filepath, 'deleted')
shutil.rmtree(precomp_dir)
print(precomp_dir, 'deleted')