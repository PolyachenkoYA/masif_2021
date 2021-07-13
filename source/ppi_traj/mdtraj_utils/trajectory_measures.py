import numpy as np
import torch as pt
import mdtraj as md
from tqdm import tqdm

from .trajectory_utils import superpose, align


def compute_distance_matrix(traj, ids_a, ids_b):
    # compute distance matrix
    xyz = traj.xyz
    D = np.sqrt(np.sum(np.square(np.expand_dims(xyz[:, ids_a], 2) - np.expand_dims(xyz[:, ids_b], 1)), -1))*1e1

    return D


def compute_rmsd(traj_ref, *trajs, selection="name CA"):
    # superpose trajectory to reference
    superpose_output = superpose(traj_ref, *trajs, selection=selection)
    trajs_sup = superpose_output[:-1]
    ids_sim_sel = superpose_output[-1]

    # compute rmsd for each trajectory
    rmsd_l = []
    for k in range(len(trajs)):
        # get atom position
        xyz = trajs_sup[k].xyz[:,ids_sim_sel[:,k+1],:]
        xyz_ref = traj_ref.xyz[:,ids_sim_sel[:,0],:].copy()
        #xyz_ref = (xyz_ref - np.expand_dims(np.mean(xyz_ref,axis=1),1))

        # compute rmsd
        rmsd_l.append(np.sqrt(np.mean(np.sum(np.square(xyz - xyz_ref), axis=2), axis=1))*1e1)

    return tuple(rmsd_l)


def compute_irmsd(traj_ref, traj_R, traj_L, *trajs, r_thr=10.0):
    # determine interface of reference and corresponding other units based on subunits
    ids_ira, ids_irb = interface_residues_within(traj_R, traj_L, r_thr, traj_ref, *trajs, selection="not type H")

    # get full interface
    ids_int = np.concatenate([ids_ira, ids_irb], axis=0)
    traj_ref_int = traj_ref.atom_slice(np.sort(ids_int[:,0]))
    trajs_int = [trajs[k].atom_slice(np.sort(ids_int[:,k+1])) for k in range(len(trajs))]

    # compute rmsd from reference interface for each trajectory
    return compute_rmsd(traj_ref_int, *trajs_int, selection="name CA")


def compute_fnat(traj_ref, traj_R, traj_L, *trajs, r_thr=5.0):
    # get residues within r_thr of the interface
    ids_ira, ids_irb = interface_residues_within(traj_R, traj_L, r_thr, traj_ref, *trajs, selection="not type H")

    # add reference trajectory at the top of the list
    all_trajs = [traj_ref]+list(trajs)

    # compute fnat for each trajectory
    Rc_map_ref = None
    fnat_l = []
    for k in range(len(trajs)+1):
        # extract current trajectory and interface atoms indices
        traj = all_trajs[k]
        ids_a = ids_ira[:,k]
        ids_b = ids_irb[:,k]

        # get all resids
        resids = np.array([a.residue.index for a in traj.topology.atoms])

        # setup mapping between resids and atom id
        mr_a = np.isclose(resids[ids_a].reshape(-1,1), np.unique(resids[ids_a]).reshape(1,-1))
        mr_b = np.isclose(resids[ids_b].reshape(-1,1), np.unique(resids[ids_b]).reshape(1,-1))

        # number of frames, residues on subunit A and residues on subunit B
        N = traj.xyz.shape[0]
        Nr_a = mr_a.shape[1]
        Nr_b = mr_b.shape[1]

        # find residue-residue contacts for each frames, one pair of residues at a time
        Rc_map = np.zeros((N, Nr_a, Nr_b), dtype=bool)
        for i in range(Nr_a):
            for j in range(Nr_b):
                # get atoms indices for residue i of A and residue j of B
                ids_ri_a = ids_a[np.where(mr_a[:,i])[0]]
                ids_rj_b = ids_b[np.where(mr_b[:,j])[0]]
                # compute distance matrix for all frames between residues i of A and residue j of B
                D_ij = compute_distance_matrix(traj, ids_ri_a, ids_rj_b)
                # update residue-residue contacts map
                Rc_map[:,i,j] = np.any(np.any((D_ij < r_thr), axis=2), axis=1)

        # set reference contacts map or compute fnat
        if Rc_map_ref is None:
            Rc_map_ref = Rc_map.copy()
        else:
            # compute fraction of native contacts based on
            fnat = np.sum(np.sum((Rc_map & Rc_map_ref), axis=2), axis=1) / (np.sum(Rc_map_ref))
            # store result
            fnat_l.append(fnat)

    return tuple(fnat_l)


def compute_contacts(sub_a, sub_b, traj, r_thr=5.0, selection="not type H", device_name="cuda"):
    # get indices for atoms for subunit A and B in the reference trajectory
    ids_sim_a = align(sub_a, traj, selection=selection)
    ids_sim_b = align(sub_b, traj, selection=selection)

    # get atom positions from subunit A and B
    xyz_a = traj.xyz[:,ids_sim_a[:,1],:]
    xyz_b = traj.xyz[:,ids_sim_b[:,1],:]

    # define device
    device = pt.device(device_name)
    # send data to device
    xyz_a = pt.from_numpy(xyz_a.astype(np.float32)).to(device)
    xyz_b = pt.from_numpy(xyz_b.astype(np.float32)).to(device)

    # for each frame
    contacts_l = []
    for k in tqdm(range(traj.xyz.shape[0])):
        # compute distance matrix
        D = pt.sqrt(pt.sum(pt.pow(xyz_a[k].unsqueeze(1) - xyz_b[k].unsqueeze(0), 2), axis=2))*1e1

        # get indices of atoms bellow threshold distance
        ids_ia, ids_ib = pt.where(D < r_thr)

        # send data back to cpu and to numpy array
        d = D[ids_ia, ids_ib].cpu().numpy().astype(np.float32)
        ica = ids_sim_a[ids_ia.cpu().numpy(),1].astype(np.int32)
        icb = ids_sim_b[ids_ib.cpu().numpy(),1].astype(np.int32)

        # store data
        contacts_l.append([d, np.stack([ica, icb], axis=1)])

    return contacts_l


def compute_sasa(traj):
    # define number of frames and atoms
    N = traj.xyz.shape[0]
    M = traj.xyz.shape[1]

    # compute solvent accessible surface area for each frame for each atom
    sasa = np.zeros((N,M), dtype=np.float32)
    for k in tqdm(range(traj.xyz.shape[0])):
        sasa[k] = md.shrake_rupley(traj[k]).ravel()

    return sasa


def interface_residues_within(sub_a, sub_b, r_thr, *trajs, selection="not type H"):
    # get indices for atoms for subunit A and B in the reference trajectory
    ids_sim_a = align(sub_a, *trajs, selection=selection)
    ids_sim_b = align(sub_b, *trajs, selection=selection)

    # compute reference distance matrix
    D = compute_distance_matrix(trajs[0][0], ids_sim_a[:,1], ids_sim_b[:,1])[0]

    # get indices for distances within r_thr
    ids_ia, ids_ib = np.where(D <= r_thr)

    # for each trajectory, find atoms from residues at interface
    ids_ira_l = []
    ids_irb_l = []
    for k in range(len(trajs)):
        # get indices of atoms for interface of trajectory k
        ids_ia_k = ids_sim_a[ids_ia,k+1]
        ids_ib_k = ids_sim_b[ids_ib,k+1]

        # get all residue indices
        resids = np.array([a.residue.index for a in trajs[k].topology.atoms])

        # get atoms from residues at interface by adding all atoms of a residue with at least one atom within r_thr
        ids_ira = np.where(np.isclose(resids.reshape(-1,1), np.unique(resids[ids_ia_k]).reshape(1,-1)))[0]
        ids_irb = np.where(np.isclose(resids.reshape(-1,1), np.unique(resids[ids_ib_k]).reshape(1,-1)))[0]

        # store indices
        ids_ira_l.append(ids_ira)
        ids_irb_l.append(ids_irb)

    return np.stack(ids_ira_l, axis=-1), np.stack(ids_irb_l, axis=-1)
