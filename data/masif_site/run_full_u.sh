#!/bin/bash
set -e

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab

# ================= arg parse ===================
argc=$#
if [ $argc -ne 3 ] && [ $argc -ne 4 ]
then
        #printf "usage:\n$0   UpdbID_Uchain   CpdbID_Cchain   [g_truth_d_cut (2.0)]\n"
        printf "usage:\n$0   UpdbID_Uchain   CpdbID_Cchain   ROCAUC_filename   [save_ply_scores (1/(0)]\n"
        exit 1
fi
if [ $argc -gt 3 ]
then
    to_save_ply_scores=$4
else
    to_save_ply_scores=0
fi

u_pdb_name=$(echo $1| cut -d"_" -f1)
u_chain_name=$(echo $1| cut -d"_" -f2)
C_pdb_name=$(echo $2| cut -d"_" -f1)
C_chain_name=$(echo $2| cut -d"_" -f2)
#u_ppi_dc_name=$u_pdb_name\-dc$ground_truth_d_cut\_$u_chain_name
u_ppi_dc_name=$1
ROCAUC_filename=$3

# ======================= run ====================

python -W ignore $masif_source/data_preparation/02-pdb_extract_and_triangulate.py    $u_pdb_name\_$u_chain_name   $C_pdb_name\_$C_chain_name  "2.0"
    
python $masif_source/data_preparation/05-masif_site_precompute.py    $u_ppi_dc_name

python -W ignore $masif_source/masif_site/masif_site_predict.py nn_models.all_feat_3l.custom_params   $u_ppi_dc_name

python -W ignore $masif_source/masif_site/masif_site_comp_ROCAUC.py nn_models.all_feat_3l.custom_params   $u_ppi_dc_name   $ROCAUC_filename   $to_save_ply_scores