import os
import sys
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import scipy.stats as stats

import mdtraj_utils as mdu
import mdtraj_utils.trajectory_utils as mdu_traj
import my_utils as my

def get_rmsd(pdb_id, u_md_id, b_md_id, database_path):
    dc = mdu.data.DataConnector(database_path, safe=True)
    
    dc.load_reference(pdb_id, b_md_id)
    dc.load_info(pdb_id, b_md_id)
        
    dc.load_trajectory(pdb_id, u_md_id)
    dc.load_reference(pdb_id, u_md_id)
    dc.load_info(pdb_id, u_md_id)
        
    u_traj = dc[pdb_id][u_md_id]['traj']
    u_ref = dc[pdb_id][u_md_id]['traj_ref']
    b_ref = dc[pdb_id][b_md_id]['traj_ref']
    main_chain_atom_ids = mdu_traj.chain_atom_indices(u_traj, 0)
    rmsd_self = md.rmsd(u_traj, u_ref, frame=0, atom_indices=main_chain_atom_ids)
    rmsd_ub = md.rmsd(u_traj, b_ref, frame=0, atom_indices=main_chain_atom_ids)
        
    return rmsd_self, rmsd_ub, u_traj.n_frames
        
def proc_scatter(x, y, x_lbl, y_lbl, model_id, x_title=None, save_ext='jpg', to_show=0, n_comps=2, file_dat=None, is_1st=False, is_last=False):
    if(x_title is None):
        x_title = x_lbl
    fig_title = model_id + '; ' + y_lbl + '(' + x_title + ')'
    fig, ax = my.get_fig(x_lbl, y_lbl, title=fig_title)
    
    R = stats.pearsonr(x, y)[0]
    x_mean, y_mean = my.gauss_classif(fig, ax, x, y, n_comps=n_comps)
    
    if(save_ext):
        fig.savefig(model_id + '_' + x_title + '_' + str(n_comps) + '.' + save_ext)
    if(to_show):
        fig.canvas.draw()
    plt.close(fig)
    
    if(not file_dat is None):
        datfile_end = '\n' if(is_last) else ' '
        if(is_1st):
            #print(model_id, x_mean, y_mean, R, file=file_dat, end=datfile_end)
            print(model_id, x_mean, R, file=file_dat, end=datfile_end)
        else:
            #print(x_mean, y_mean, R, file=file_dat, end=datfile_end)
            print(x_mean, R, file=file_dat, end=datfile_end)
    
    return (x_mean, y_mean, R), (fig, ax)
    
def timeseries_proc(time, y, y_axis_label, y_title=None, model_id=None, filter_margin=3, k_base=6, save_ext='jpg', to_show=0, file_dat=None, file_humanread=None, is_1st=False, is_last=False):
    if(y_title is None):
        y_title = y_axis_label
    if(model_id is None):
        print(my.error)
    
    N_points = len(time)
    fit_coefs, fit_err = np.polyfit(time, y, 1, cov=True)
    fit_err = np.sqrt(np.diagonal(fit_err))
    mean_y = np.mean(y)
    full_std = np.std(y)
    anomaly_ids = (np.abs(y - mean_y) > full_std * filter_margin)
    y_filtered = y[~anomaly_ids]
    time_filtered = time[~anomaly_ids]
    N_points_filt = len(time_filtered)
    fit_filt_coefs, fit_filt_err = np.polyfit(time_filtered, y_filtered, 1, cov=True)
    fit_filt_err = np.sqrt(np.diagonal(fit_filt_err))
    mean_y_filtered = np.mean(y_filtered)    
    
    line_label = '; ' + ('k = (' + my.f2str(fit_coefs[0] * 10**k_base) + ' \pm ' + my.f2str(fit_err[0] * 10**k_base) + ') e' + str(-k_base) if(abs(fit_coefs[0]) > fit_err[0]) else '')
    filt_line_label = '; ' + ('k = (' + my.f2str(fit_filt_coefs[0] * 10**k_base) + ' \pm ' + my.f2str(fit_filt_err[0] * 10**k_base) + ') e' + str(-k_base) if(abs(fit_filt_coefs[0]) > fit_filt_err[0]) else '')
    plot_title = model_id + '; ' + y_title + '(t); mean = ' + my.f2str(mean_y)
    fig, ax = my.get_fig('time (ns)', y_axis_label, title=plot_title)
    ax.plot(time, y)
    ax.plot(time, np.polyval(fit_coefs, time), label=r'full $' + line_label + '$')
    if(np.any(anomaly_ids)):
        ax.plot(time, np.polyval(fit_filt_coefs, time), label='$' + str(filter_margin) + ' \sigma; ' + filt_line_label + '$')
    ax.legend()   
    
    if(save_ext):
        fig.savefig(model_id + '_' + y_title + '_time.' + save_ext)
    if(to_show):
        fig.canvas.draw()
    plt.close(fig)    
    
    # ====== inter-protein stats ======
    file_end = '\n' if(is_last) else ' '
    if(not file_humanread is None):
        save_line = (model_id + ': ') if(is_1st) else ''
        save_line += y_axis_label + ' mean = ' + my.f2str(mean_y) + ', slope = ' + my.f2str(fit_coefs[0])
        save_line += ((' +- ' + my.f2str(fit_err[0])) if(abs(fit_coefs[0]) > fit_err[0]) else '')
        print(save_line, file=file_humanread, end=file_end)
    if(not file_dat is None):        
        save_line = []
        #if(is_1st):
        #    save_line.append(model_id)
        save_line = [mean_y, fit_coefs[0], fit_err[0]]
        save_line = ' '.join([str(a) for a in save_line])
        print(save_line, file=file_dat, end=file_end)            
            
    return anomaly_ids
    
# ================= cycle params ==================
pdbs = list(my.chain_ids_table.keys())

step = 50
dt = 0.02   # ns
pdb_dir = 'PDBS'
N_worst_cases = 5
verbose = True
database_path = 'database'
save_ext_flag = '-save_ext'
verbose_flag = '-v'
to_show_flag = '-show'
pdbs_ids_flag = '-pdb_ids'
mds_flag = '-md_labels'
flags = [pdbs_ids_flag, mds_flag, save_ext_flag, verbose_flag, to_show_flag]
all_pdbs_const = 'all'

# ============== input parse ==============
possible_pdbs_ids = [str(i) for i in range(len(pdbs))] + [all_pdbs_const]
possible_pdbs_ids_numbers = range(len(pdbs) + 1)
[pdbs_ids, md_labels, save_ext, verbose, to_show], correct_input = \
    my.parse_args(sys.argv[1:], flags, \
                  possible_values=[possible_pdbs_ids, ['R', 'L'], ['eps', 'jpg'], ['0', '1'], ['0', '1']], \
                  possible_arg_numbers=[possible_pdbs_ids_numbers, [1, 2], [0, 1], [0, 1], [0, 1]], \
                  default_values=[all_pdbs_const, None, '', '0', '0'])
if(save_ext):
    save_ext = save_ext[0]
pdbs_ids = range(len(pdbs)) if(pdbs_ids[0] == all_pdbs_const) else [int(i) for i in pdbs_ids]
verbose = (verbose[0] == '1')
to_show = (to_show[0] == '1')

# ================ cycle =====================
res_file_humanread = open('rocauc.txt', 'w')
res_file_dat = open('rocauc.dat', 'w')
print(r'# ' + ', '.join([pdbs[i] for i in pdbs_ids]) + ' | ' + ', '.join(md_labels), file=res_file_dat)
print(r'# rocauc_mean   linear_rocauc(t)_k   linear_rocauc(t)_k_err   GTsize_mean   linear_GTsize(t)_k   linear_GTsize(t)_k_err   GTsize_mean   GTsize_rocauc_R   RMSDub_mean  RMSDub_rocauc_R   RMSDself_mean RMSDself_rocauc_R', file=res_file_dat)
for pdb_i in pdbs_ids:
    pdb_id = pdbs[pdb_i]
    for part_id in md_labels:
        # ===== names ====
        b_md_id = 'b' + part_id
        u_md_id = 'u' + part_id
        model_id = '-'.join([pdb_id, u_md_id])
        print('working on ' + model_id)
        
        # ===== RMSD ====
        rmsd_self, rmsd_ub, N_frames = get_rmsd(pdb_id, u_md_id, b_md_id, database_path)
        
        # ===== ROCAUC ====
        data = np.loadtxt(os.path.join(pdb_dir, model_id + '.dat'))
        frames_ids = np.intc(data[:, 0])
        rocauc = data[:, 1]
        groundtruth_size = data[:, 2]
        time = frames_ids * dt     
        sorted_frames_ids = sorted(enumerate(frames_ids), key=lambda f_i: rocauc[f_i[0]])
                        
        if(verbose):            
            missing_frames = []
            for f_i in range(0, N_frames, step):
                if(not f_i in frames_ids):
                    missing_frames.append(f_i)
            if(missing_frames):
                print('these frames were not found:\n', missing_frames)
            print('\n' + str(N_worst_cases) + ' worst cases:')
            for i in range(N_worst_cases):
                print('frame', sorted_frames_ids[i][1], ': ROCAUC = ', rocauc[sorted_frames_ids[i][0]])
                
            print('\n' + str(N_worst_cases) + ' best cases:')
            for i in range(N_worst_cases):
                print('frame', sorted_frames_ids[-1-i][1], ': ROCAUC = ', rocauc[sorted_frames_ids[-1-i][0]])
                
            rocauc_mean = np.mean(rocauc)
            sorted_middle_frames_ids = sorted(enumerate(frames_ids), key=lambda f_i: abs(rocauc[f_i[0]] - rocauc_mean))
            print('\n' + str(N_worst_cases) + ' typical (close to the average) cases:')
            for i in range(N_worst_cases):
                print('frame', sorted_middle_frames_ids[i][1], ': ROCAUC = ', rocauc[sorted_middle_frames_ids[i][0]])

        rmsd_ub_picked = rmsd_ub[frames_ids]
        rmsd_self_picked = rmsd_self[frames_ids]
               
        # ===== proc & plot =====
        timeseries_proc(time, rocauc, y_axis_label='ROCAUC', model_id=model_id, filter_margin=3, k_base=6, save_ext=save_ext, to_show=0, file_dat=res_file_dat, file_humanread=res_file_humanread, is_1st=True)
        timeseries_proc(time, groundtruth_size, y_axis_label='main GT patch size', y_title='GT', model_id=model_id, filter_margin=3, k_base=3, save_ext=save_ext, to_show=0, file_dat=res_file_dat, file_humanread=res_file_humanread)
        timeseries_proc(np.arange(N_frames) * dt, rmsd_ub, y_axis_label='RMSD_ub', y_title='RMSDub', model_id=model_id, filter_margin=3, k_base=6, save_ext=save_ext, to_show=0, file_dat=res_file_dat, file_humanread=res_file_humanread)
        timeseries_proc(np.arange(N_frames) * dt, rmsd_self, y_axis_label='RMSD_self', y_title='RMSDself', model_id=model_id, filter_margin=3, k_base=6, save_ext=save_ext, to_show=0, file_dat=res_file_dat, file_humanread=res_file_humanread)        
        
        for n_comps in range(1, 4):
            dat_file_link = (res_file_dat if(n_comps==1) else None)
            proc_scatter(groundtruth_size, rocauc, 'ground-truth patch size (vertices)', 'ROCAUC', model_id, x_title='GT', save_ext=save_ext, to_show=0, n_comps=n_comps, file_dat=dat_file_link)
            proc_scatter(rmsd_ub_picked, rocauc, '$rmsd - bR0 (nm)$', 'ROCAUC', model_id, x_title='RMSD_ub', save_ext=save_ext, to_show=0, n_comps=n_comps, file_dat=dat_file_link)
            proc_scatter(rmsd_self_picked, rocauc, '$rmsd - uR0 (nm)$', 'ROCAUC', model_id, x_title='RMSD_self', save_ext=save_ext, to_show=0, n_comps=n_comps, is_last=True, file_dat=dat_file_link)
        
res_file_dat.close()
res_file_humanread.close()