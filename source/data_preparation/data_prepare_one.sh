#!/bin/bash
masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
masif_matlab=$masif_root/source/matlab_libs
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab
PDB_ID=$1
python -W ignore $masif_source/data_preparation/02-pdb_extract_and_triangulate.py $PDB_ID
