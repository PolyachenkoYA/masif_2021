#!/bin/bash

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source
#masif_matlab=$masif_root/source/matlab_libs
export PYTHONPATH=$PYTHONPATH:$masif_source
#export masif_matlab

#python -W ignore $masif_source/data_preparation/02-pdb_extract_and_triangulate.py "$@"
python $masif_source/data_preparation/02-pdb_extract_and_triangulate.py "$@"
