#!/bin/bash
set -e

masif_root=$(git rev-parse --show-toplevel)
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab

# ======================= run ====================

python -W ignore $masif_root/data/masif_site/run_cycle.py  $1  $2  $3
    
