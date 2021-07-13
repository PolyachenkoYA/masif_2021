import numpy as np
import torch as pt
from tqdm import tqdm


def contacts_distribution(xyz0_, xyz1_, bins, device=pt.device("cuda")):
    # setup data
    xyz0 = pt.from_numpy(xyz0_).to(device)
    xyz1 = pt.from_numpy(xyz1_).to(device)

    # define bins
    r_inf = pt.from_numpy(np.array(bins[:-1])).reshape(1,1,-1).to(device)
    r_sup = pt.from_numpy(np.array(bins[1:])).reshape(1,1,-1).to(device)

    # empty probability matrix
    P = pt.zeros(xyz0.shape[1], xyz1.shape[1], len(bins)-1).to(device)

    # compute contacts
    for k in tqdm(range(xyz0.shape[0])):
        # compute distances matrix
        D = pt.sqrt(pt.sum(pt.pow(xyz0[k].unsqueeze(1) - xyz1[k].unsqueeze(0), 2), dim=2))

        # compute contacts probability
        P += ((D.unsqueeze(2) < r_sup) & (D.unsqueeze(2) >= r_inf)).float()

    # normalize
    P = P / (pt.sum(P, dim=2) + 1e-6).unsqueeze(2)

    return P.cpu().numpy()


class StatisticalContactsModel:
    def __init__(self, xmin, xmax, num_bins, device_name="cuda"):
        # set bins
        self.bins = np.linspace(xmin, xmax, num_bins)
        # define device
        self.device = pt.device(device_name)

    def fit(self, traj, other_traj=None):
        # if single trajectory provided
        if other_traj is None:
            self.P = contacts_distribution(traj.xyz, traj.xyz, self.bins, device=self.device)
        else:
            self.P = contacts_distribution(traj.xyz, other_traj.xyz, self.bins, device=self.device)

    def loglikelihood(self, traj, other_traj=None):
        # if single trajectory provided
        if other_traj is None:
            xyz0 = pt.from_numpy(traj.xyz).to(self.device)
            xyz1 = pt.from_numpy(traj.xyz).to(self.device)
        else:
            xyz0 = pt.from_numpy(traj.xyz).to(self.device)
            xyz1 = pt.from_numpy(other_traj.xyz).to(self.device)

        # setup model
        P = pt.from_numpy(self.P).to(self.device)

        # define bins
        r_inf = pt.from_numpy(np.array(self.bins[:-1])).reshape(1,1,-1).to(self.device)
        r_sup = pt.from_numpy(np.array(self.bins[1:])).reshape(1,1,-1).to(self.device)

        # empty probability matrix
        lll = pt.zeros(xyz0.shape[0]).to(self.device)

        # compute contacts
        for k in tqdm(range(xyz0.shape[0])):
            # compute distances matrix
            D = pt.sqrt(pt.sum(pt.pow(xyz0[k].unsqueeze(1) - xyz1[k].unsqueeze(0), 2), dim=2))

            # compute contacts probability
            Q = ((D.unsqueeze(2) < r_sup) & (D.unsqueeze(2) >= r_inf)).float()
            lll[k] = -pt.mean(pt.log((1.0 - (P * Q) + pt.floor(P * Q))))

        return lll.cpu().numpy()
