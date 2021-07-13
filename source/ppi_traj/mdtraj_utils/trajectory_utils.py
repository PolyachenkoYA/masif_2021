import numpy as np
import mdtraj as md


def join_trajectories(traj_list, selection="all"):
    # align trajectories
    ids_sim = align(traj_list[0][0], *traj_list[1:], selection=selection)

    # topology checks
    df_topo_ref = traj_list[0].topology.to_dataframe()[0].iloc[ids_sim[:,0]][['name', 'resName']]
    for k in range(1,len(traj_list)):
        assert np.all(df_topo_ref.values == traj_list[k].topology.to_dataframe()[0].iloc[ids_sim[:,k]][['name', 'resName']].values)

    # create new trajectory
    xyz = np.concatenate([traj_list[k].xyz[:,ids_sim[:,k],:] for k in range(len(traj_list))], axis=0)
    topology = traj_list[0].atom_slice(ids_sim[:,0]).topology

    return md.Trajectory(xyz, topology=topology)


# alignment
def get_atoms_per_chain(traj, selection='all'):
    # define filter for atom type
    return [np.array([a.index for a in chain.atoms]) for chain in traj.topology.chains]

def chain_atom_indices(traj, chain_id):
    return np.array([a.index for a in traj.topology.chain(chain_id).atoms])

def chain_atom_names(traj, chain_id):
    return np.array([a.name for a in traj.topology.chain(chain_id).atoms])

def compare_chains_in_trajs(traj1, traj2, chain_id1=0, chain_id2=0, traj_type='traj'):
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

def unwrap_pbc(traj):
    # setup meshgrid for PBC repetitions
    dgrid = np.array([0.0, 1.0, -1.0])
    dX, dY, dZ = np.meshgrid(dgrid, dgrid, dgrid)
    dV = np.stack([dX.ravel(), dY.ravel(), dZ.ravel()], -1)

    # get indices of atoms for each molecules
    ids_mol_l = get_atoms_per_chain(traj)

    # compute center of mass of each molecule and its images
    pcm_rep_mol = np.zeros((len(ids_mol_l), 27, traj.xyz.shape[0], 3))
    for i in range(len(ids_mol_l)):
        # compute center of mass
        pcm = md.geometry.distance.compute_center_of_mass(traj.atom_slice(ids_mol_l[i]))

        # compute CM for all nearest periodic images
        for k in range(dV.shape[0]):
            pcm_rep_mol[i][k] = (pcm + traj.unitcell_lengths * dV[k].reshape(1,-1))

    # choose reference molecule with CM in reference cell
    pcm_ref = pcm_rep_mol[0][0]

    # make copy of trajectory
    traj_fix = traj[:]

    # for each other molecule
    for i in range(1,pcm_rep_mol.shape[0]):
        # compute distance of all images with reference molecule
        dcm_rep = np.sqrt(np.sum(np.square(pcm_rep_mol[i] - np.expand_dims(pcm_ref,0)), axis=2))

        # find molecule image closest to reference molecule
        ids_img = np.argmin(dcm_rep, axis=0)

        # update position of molecule
        traj_fix.xyz[:,ids_mol_l[i],:] += np.expand_dims(traj.unitcell_lengths * dV[ids_img],1)

    return traj_fix


def identify(top_a, top_b):
    # identify similar and mutated atoms pairs
    ids_sim_l = []
    ids_mut_l = []
    chain_a_used = set()
    chain_b_used = set()
    for chain_a in top_a.chains:
        # get number of residue of chain from molecule a
        n_res_a = len(chain_a._residues)
        for chain_b in top_b.chains:
            # get number of residue of chain from molecule b
            n_res_b = len(chain_b._residues)

            # length check
            if (n_res_a == n_res_b) and (chain_a.index not in chain_a_used) and (chain_b.index not in chain_b_used):
                # single residue chains (molecules, ions)
                if n_res_a == 1:
                    if list(chain_a.residues)[0].name.lower() != list(chain_b.residues)[0].name.lower():
                        continue

                # sequence check
                for res_a, res_b in zip(chain_a.residues, chain_b.residues):
                    # mutation warning
                    if res_a.name.lower() != res_b.name.lower():
                        print("WARNING: [{}]{} != [{}]{}".format(chain_a.index, res_a, chain_b.index, res_b))

                # get indices of matching residues
                for ra, rb in zip(chain_a.residues, chain_b.residues):
                    if ra.name.lower() == rb.name.lower():
                        # get all atoms of corresponding residues
                        ra_atoms = [a for a in ra.atoms]
                        rb_atoms = [b for b in rb.atoms]

                        # check that the two residues have the same number of atoms
                        if (len(ra_atoms) != len(rb_atoms)):
                            # if not same number of atoms -> nothing to do
                            print("ERROR: different number of atoms for {}({}) : {}({})".format(ra, len(ra_atoms), rb, len(rb_atoms)))
                        else:
                            # try to find unique ordering of atoms
                            a_names = [a.name for a in ra.atoms]
                            b_names = [b.name for b in rb.atoms]

                            # if not unique -> nothing to do
                            if ((len(a_names) != len(np.unique(a_names))) or (len(b_names) != len(np.unique(b_names)))):
                                print("ERROR: non-unique atoms mismatch for {} : {}".format(ra, rb))

                            elif np.all([a_name==b_name for a_name,b_name in zip(a_names,b_names)]):
                                for a, b in zip(ra.atoms, rb.atoms):
                                    ids_sim_l.append([a.index, b.index])

                            else:
                                print("INFO: reordering atoms mismatch for {} : {}".format(ra, rb))

                                # find unique ordering
                                ids_reo_a = np.argsort(a_names)
                                ids_reo_b = np.argsort(b_names)

                                # get corresponding reordered atom indices
                                a_ids = np.array([a.index for a in ra.atoms])[ids_reo_a]
                                b_ids = np.array([b.index for b in rb.atoms])[ids_reo_b]

                                for ia, ib in zip(a_ids, b_ids):
                                    ids_sim_l.append([ia, ib])

                    else:
                        ids_mut_l.append(([a.index for a in ra.atoms], [b.index for b in rb.atoms]))

                # history chain used
                chain_a_used.add(chain_a.index)
                chain_b_used.add(chain_b.index)

    return np.array(ids_sim_l), ids_mut_l


def align(traj_ref, *trajs, selection="all"):
    # reference trajectory with selection
    ids_sel_ref = traj_ref.topology.select(selection)
    traj_sel_ref = traj_ref[0].atom_slice(ids_sel_ref)

    # for each input trajectory
    ids_sim_l = []
    for traj in trajs:
        # selected trajectory
        ids_sel = traj.topology.select(selection)
        traj_sel = traj[0].atom_slice(ids_sel)

        # identify to reference trajectory
        ids_sim_sel, _ = identify(traj_sel_ref.topology, traj_sel.topology)
        # get indices for input and not selected subset
        ids_sim_l.append(np.stack([ids_sel_ref[ids_sim_sel[:,0]], ids_sel[ids_sim_sel[:,1]]], axis=-1))

    # find common atoms between all trajectories
    ids_sim = ids_sim_l[0].copy()
    for k in range(1, len(ids_sim_l)):
        # intersection masks
        m0 = np.in1d(ids_sim[:,0], ids_sim_l[k][:,0])
        m1 = np.in1d(ids_sim_l[k][:,0], ids_sim[:,0])

        # filter previous indices and insert new indices
        ids_sim = np.concatenate([ids_sim[m0], ids_sim_l[k][m1,1].reshape(-1,1)], axis=1)

    return ids_sim


def center(traj):
    traj_c = traj[:]
    traj_c.xyz = (traj_c.xyz - np.expand_dims(np.mean(traj_c.xyz,axis=1),1))
    return traj_c


def superpose_transform(xyz_ref, xyz):
    # copy data
    p = xyz.copy()
    p_ref = xyz_ref.copy()

    # centering
    t = np.expand_dims(np.mean(p,axis=1),1)
    t_ref = np.expand_dims(np.mean(p_ref,axis=1),1)

    # SVD decomposition
    U, S, Vt = np.linalg.svd(np.matmul(np.swapaxes(p_ref-t_ref,1,2), p-t))

    # reflection matrix
    Z = np.zeros(U.shape) + np.expand_dims(np.eye(U.shape[1], U.shape[2]),0)
    Z[:,-1,-1] = np.linalg.det(U) * np.linalg.det(Vt)

    R = np.matmul(np.swapaxes(Vt,1,2), np.matmul(Z, np.swapaxes(U,1,2)))
    return t, R, t_ref  # np.matmul(xyz - t, R) + t_ref


def superpose(traj_ref, *trajs, selection='name CA'):
    # identify same chains
    ids_sim = align(traj_ref, *trajs, selection=selection)

    # get reference positions
    xyz_ref = traj_ref.xyz[:,ids_sim[:,0],:]

    # align each input trajectory to reference
    traj_sup_l = []
    for k in range(len(trajs)):
        # get positions
        xyz = trajs[k].xyz[:,ids_sim[:,k+1],:]

        # compute the alignment transformation
        t, R, t_ref = superpose_transform(xyz_ref, xyz)

        # superpose trajectory to reference
        traj_sup_l.append(trajs[k][:])
        traj_sup_l[-1].xyz = np.matmul(traj_sup_l[-1].xyz-t, R) + t_ref

    return tuple(traj_sup_l + [ids_sim])


def atoms_to_residue_contacts(topology, ic_l, dc_l):
    # get all resids
    resids = np.array([a.residue.index for a in topology.atoms])

    # setup mapping between resids and atom id
    mr = np.isclose(resids.reshape(-1,1), np.unique(resids).reshape(1,-1))

    # find residue-residue contacts
    resids_int_l = []
    dmin_rr_l = []
    for k in range(len(ic_l)):
        if len(ic_l[k]) > 0:
            # get residue to atom at interface A and B
            resids_ia = np.where(mr[ic_l[k][:,0]])[1]
            resids_ib = np.where(mr[ic_l[k][:,1]])[1]

            # get unique residue-residue contacts
            resids_int, ids_inv = np.unique(np.stack([resids_ia, resids_ib], axis=1), return_inverse=True, axis=0)

            # find minimum distances for each residue-residue contact
            dmin_rr = np.zeros(resids_int.shape[0], dtype=np.float32)
            for i in np.unique(ids_inv):
                dmin_rr[i] = np.min(dc_l[k][np.where(ids_inv == i)[0]])
        else:
            resids_int = np.array([])
            dmin_rr = np.array([])

        # store data
        resids_int_l.append(resids_int)
        dmin_rr_l.append(dmin_rr)

    return resids_int_l, dmin_rr_l
