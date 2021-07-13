import os
import sys
import numpy as np
import pandas as pd
import mdtraj as md
import parmed as pmd
import matplotlib.pyplot as plt
import subprocess as sp

import mdtraj_utils as mdu
import my_rmsd

import data_preparation.extract_and_triangulate_lib as ext_and_trg

# =========== glocal constants ========
md_C = 'C'
md_uR = 'uR'
md_uL = 'uL'
traj_type_ref = 'traj_ref'
traj_type_full = 'traj'
frame_step = 50

md_runs = [md_C, md_uR, md_uL]  # this is waiting for the pull-request to be approved because of a indexing issue in the compiled part of the package


# =================================== defines =========================================

def print_usage_and_exit(exe_name=sys.argv[0]):
    print('usage:\n' + exe_name + '   [PDBID]')
    sys.exit(1)

def chain_atom_indices(traj, chain_id=0):
    return np.array([a.index for a in traj.topology.chain(chain_id).atoms])

def chain_atom_names(traj, chain_id=0):
    return np.array([a.name for a in traj.topology.chain(chain_id).atoms])

def compare_chains_in_trajs(traj1, traj2, chain_id1=0, chain_id2=0, traj_type=traj_type_full):
    chainA_1_names = chain_atom_names(traj1, chain_id=chain_id1)
    chainA_2_names = chain_atom_names(traj2, chain_id=chain_id2)
    N_atoms = len(chainA_1_names)

    res = None
    if(len(chainA_2_names) == N_atoms):
        for i in range(N_atoms):
            if(chainA_1_names[i] != chainA_2_names[i]):
                res = i
                break
    else:
        res = -1
                
    return  res

def exctact_pdbs(pdb_id, u_name, C_name, dc, frame_inds=100, u_chain_id=0, C_chain_id=0, traj_type=traj_type_full):
    u_traj = dc[pdb_id][u_name][traj_type]
    C_traj = dc[pdb_id][C_name][traj_type]
    u_info_str = dc[pdb_id][u_name]['info']['pdbid'] + '_' + dc[pdb_id][u_name]['info']['mdid']
    C_info_str = dc[pdb_id][C_name]['info']['pdbid'] + '_' + dc[pdb_id][C_name]['info']['mdid']
    not_matching_atom = \
        compare_chains_in_trajs(u_traj, C_traj, chain_id1=u_chain_id, chain_id2=C_chain_id)
    if(not_matching_atom is None):        
        N_frames = len(u_traj)
        if(isinstance(frame_inds, int)):
            frame_inds = np.arange(0, N_frames, frame_inds)
        for i, frame_i in enumerate(frame_inds):
            if(frame_i >= N_frames):
                print('ERROR:\nframe_inds[' + str(i) + '] = ' + str(frame_i) + ' exceeds the u_trajectory:', dc[pdb_id][u_name], file=sys.stderr)
                return 1
        
        pdb_path = os.path.join('PDBS', pdb_id)
        os.makedirs(pdb_path, exist_ok=True)
            
        u_chain_idices = chain_atom_indices(u_traj[0], chain_id=u_chain_id)
        C_chain_idices = chain_atom_indices(C_traj[0], chain_id=C_chain_id)
        #print(u_chain_idices)
        #print(C_chain_idices)
        N_frames_to_print = len(frame_inds)
        for n_done, frame_i in enumerate(frame_inds):
            C_traj.superpose(u_traj, frame=frame_i, \
                                atom_indices=C_chain_idices, ref_atom_indices=u_chain_idices)
            #rmsd_u_C = md.rmsd(C_traj, u_traj, frame=frame_i, \
            #                    atom_indices=C_chain_idices, ref_atom_indices=u_chain_idices)
            rmsd_u_C = my_rmsd.rmsd(C_traj, u_traj, frame=frame_i, \
                                atom_indices=C_chain_idices, ref_atom_indices=u_chain_idices)

            closest_conf_ind = np.argmin(rmsd_u_C)
            
            u_traj[frame_i].save(os.path.join(pdb_path, '-'.join([pdb_id, u_name, str(frame_i)]) + '.pdb'))
            C_traj[closest_conf_ind].save(os.path.join(pdb_path, '-'.join([pdb_id, u_name, C_name, str(frame_i)]) + '.pdb'))
            print((n_done + 1) / N_frames_to_print * 100, '% done             \r', end='')
    else:
        if(not_matching_atom > 0):
            print(u_info_str + '.atom[' + str(not_matching_atom) + ']:', u_traj.topology.atom(not_matching_atom), file=sys.stderr)
            print(C_info_str + '.atom[' + str(not_matching_atom) + ']:', C_traj.topology.atom(not_matching_atom), file=sys.stderr)
        else:
            print('len(' + u_info_str + '_chain' + str(u_chain_id) + ') = ' + str(len(chain_atom_names(u_traj, chain_id=u_chain_id))))
            print('len(' + C_info_str + '_chain' + str(C_chain_id) + ') = ' + str(len(chain_atom_names(C_traj, chain_id=C_chain_id))))
    
    return 0
    
def process_pdb_traj(pdb_id, md_runs, frame_inds=100, verbose=True, traj_type=traj_type_full):
    if(verbose):
        print('============ processing ' + pdb_id + ' ============')

    dc = mdu.data.DataConnector("database", safe=True)
    if(not traj_type in [traj_type_full, traj_type_ref]):
        print('"' + str(traj_type) + '" trajectory type is invalid', file=sys.stderr)
        return 1
    
    md_C = md_runs[0]
    dc.load_reference(pdb_id, md_C)
    dc.load_info(pdb_id, md_C)
    if(traj_type == traj_type_full):
        dc.load_trajectory(pdb_id, md_C)
    elif(traj_type == traj_type_ref):
        frame_inds = [0]        
    
    for md_id in md_runs[1:]:        
        if(verbose):
            print('processing "' + md_id + '"')
            
        dc.load_reference(pdb_id, md_id)
        dc.load_info(pdb_id, md_id)
        if(traj_type == traj_type_full):
            dc.load_trajectory(pdb_id, md_id)
        
        exctact_pdbs(pdb_id, md_id, md_C, dc, frame_inds=frame_inds, u_chain_id=0, C_chain_id=ext_and_trg.chain_ids_table[pdb_id][md_id], traj_type=traj_type)
        
    return 0

# ============== parse input ============
args = sys.argv[1:]
argc = len(args)

pdbs_list = args

# ================ process ===============
if(len(pdbs_list) == 0):
    for dir_name in os.listdir('database'):
        if((len(dir_name) == 4) and (dir_name.upper() == dir_name) and dir_name[0].isdigit()):
            pdbs_list.append(dir_name)

for pdb_id in pdbs_list:
    #process_pdb_traj(pdb_id, md_runs, frame_inds=frame_step, traj_type=traj_type_ref)
    process_pdb_traj(pdb_id, md_runs, frame_inds=frame_step, traj_type=traj_type_full)