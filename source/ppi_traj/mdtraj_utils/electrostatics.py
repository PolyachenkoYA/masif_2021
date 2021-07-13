import numpy as np

# constants
eps0 = 8.8541878128e-22  # C^2 J^-1 Å^-1
eps = 80.0 * eps0  # C^2 J^-1 Å^-1
e = -1.602176634e-19  # C
N = 6.02214179e23  # Avogadro

# scaling factors
cV = 1e-3 * N * (e*e) / (4.0 * np.pi * eps)  # kJ mol^-1 Å
cF = 1e3 * 1e9 * 1e10 / N


def compute_dipole_moment(q, m, X):
    r0 = np.sum(X*m.reshape(1,-1,1), axis=1) / np.sum(m)
    p = np.sum(q.reshape(1,-1,1) * (X-r0.reshape(-1,1,3)), axis=1)
    return p, r0


def compute_charge_charge_interaction_potential(q1, q2, r):
    d = np.sqrt(np.sum(np.square(r), axis=1))
    return cV * q1 * q2 / d


def compute_charge_dipole_interaction_potential(q, p, r):
    d = np.sqrt(np.sum(np.square(r), axis=1))
    r_hat = r / d.reshape(-1,1)
    return cV * q * np.sum(r_hat * p, axis=1) / (d*d)


def compute_dipole_dipole_interaction_potential(p1, p2, r):
    d = np.sqrt(np.sum(np.square(r), axis=1))
    r_hat = r / d.reshape(-1,1)
    return cV * (np.sum(p1 * p2, axis=1) - 3.0 * np.sum(p1 * r_hat, axis=1) * np.sum(p2 * r_hat, axis=1)) / (d*d*d)


def compute_charge_charge_force_amplitude(q1, q2, r, h=1e-1):
    # direction vector
    d = np.sqrt(np.sum(np.square(r), axis=1))
    r_hat = r / d.reshape(-1,1)
    # compute force amplitude along r_hat
    F_rph = compute_charge_charge_interaction_potential(q1, q2, r + 0.5 * h * r_hat)
    F_rmh = compute_charge_charge_interaction_potential(q1, q2, r - 0.5 * h * r_hat)
    return cF * -(F_rph - F_rmh) / h


def compute_charge_dipole_force_amplitude(q, p, r, h=1e-1):
    # direction vector
    d = np.sqrt(np.sum(np.square(r), axis=1))
    r_hat = r / d.reshape(-1,1)
    # compute force
    F_rph = compute_charge_dipole_interaction_potential(q, p, r + 0.5 * h * r_hat)
    F_rmh = compute_charge_dipole_interaction_potential(q, p, r - 0.5 * h * r_hat)
    return cF * -(F_rph - F_rmh) / h


def compute_dipole_dipole_force_amplitude(p1, p2, r, h=1e-1):
    # direction vector
    d = np.sqrt(np.sum(np.square(r), axis=1))
    r_hat = r / d.reshape(-1,1)
    # compute force amplitude along r_hat
    F_rph = compute_dipole_dipole_interaction_potential(p1, p2, r + 0.5 * h * r_hat)
    F_rmh = compute_dipole_dipole_interaction_potential(p1, p2, r - 0.5 * h * r_hat)
    return cF * -(F_rph - F_rmh) / h
