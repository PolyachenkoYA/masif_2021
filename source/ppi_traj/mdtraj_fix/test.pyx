import numpy as np
import mdtraj as md

import my_rmsd

def test_gh_1571():
    n_frames = 1
    n_atoms_1 = 1
    n_atoms_2 = 2
    
    top_1 = md.Topology()
    top_1.add_chain()
    top_1.add_residue('RS2', top_1.chain(0))
    top_1.add_atom('A2', 'H', top_1.residue(0))
    
    top_2 = md.Topology()
    top_2.add_chain()
    top_2.add_residue('RS1', top_2.chain(0))
    top_2.add_atom('A1', 'H', top_2.residue(0))
    top_2.add_chain()
    top_2.add_residue('RS2', top_2.chain(1))
    top_2.add_atom('A2', 'H', top_2.residue(1))
    # here the 2nd chain in the top_2 is rmsd-compatible to the one in the top_1 so we should be able to compute rsmd between them.
    
    trj_1 = md.Trajectory(np.random.RandomState(0).randn(n_frames, n_atoms_1, 3), top_1)
    trj_2 = md.Trajectory(np.random.RandomState(0).randn(n_frames, n_atoms_2, 3), top_2)
    
    my_rmsd.rmsd(trj_1, trj_2, atom_indices=[0], ref_atom_indices=[1])
    my_rmsd.rmsd(trj_2, trj_1, atom_indices=[1], ref_atom_indices=[0])
    print('my_rmsd done')
    
    md.rmsd(trj_1, trj_2, atom_indices=[0], ref_atom_indices=[1])
    md.rmsd(trj_2, trj_1, atom_indices=[1], ref_atom_indices=[0])
    print('md.rmsd done')
    
#test_gh_1571()
#print('DONE')