#!/usr/bin/env python3
"""Does any single run of the 667 L800 runs fit two bands at once? Follow-up to npe_lf_3param_l800.ipynb, no network.

Compares the old pair (Cole+01 K, Norberg+02 bJ) with the new pair (Driver+12 K and r), for two values of sigma_disc.
"""
import h5py
import numpy as np

CAL = "/cosma/apps/durham/dc-hick2/galform_clustering_calibration/data"
OBS = "/cosma/apps/durham/dc-lock1/galform/Obs_Data2"
f = h5py.File(f"{CAL}/lf_runs.h5", "r")
theta = h5py.File(f"{CAL}/dataset_l800.h5", "r")["thetas"][:]
edges = f["lf_edges"][:]
mid = 0.5 * (edges[1:] + edges[:-1])
V = 542.16**3 / 1024
LF = {b: np.log(np.maximum(f[f"l800/{b}/counts"][:].sum(1), 0.5) / (64 * V * np.diff(edges))) for b in ("k", "bj", "r")}
RANGE = {"k": (-23.75, -18.5), "bj": (-22.8, -16.0), "r": (-23.4, -17.75)}


def dataset(band, mag, lnphi, sigma):
    use = (mag > RANGE[band][0]) & (mag < RANGE[band][1])
    return band, mag[use], lnphi[use], sigma[use]


def driver(band, file_band):
    m, p, e = np.loadtxt(f"{OBS}/lf{file_band}_z0_driver12.data", usecols=(0, 1, 2), unpack=True)
    ok = p > 0
    return dataset(band, m[ok], np.log(p[ok] / 0.5), e[ok] / p[ok])


m, p, e = np.loadtxt(f"{OBS}/lfJK_Cole2001.data", usecols=(0, 3, 4), unpack=True)
ok = p > 0
cole = dataset("k", m[ok] + 1.870, np.log(p[ok]), e[ok] / p[ok])
p, m, e = np.loadtxt(f"{OBS}/lfBJ_norberg02.data", usecols=(0, 1, 2), unpack=True)
ok = p > 0
norberg = dataset("bj", m[ok] - 0.089, np.log(p[ok]), e[ok] / p[ok])
DATA = {"Cole K": cole, "Norberg bJ": norberg, "GAMA K": driver("k", "k"), "GAMA r": driver("r", "r")}


def chi2(d, sigma_disc):
    band, mag, lnphi, sigma = d
    x = np.array([np.interp(mag, mid, run) for run in LF[band]])
    return ((x - lnphi)**2 / (sigma**2 + sigma_disc**2)).sum(1), len(mag)


for sigma_disc in (0.2, 0.1):
    c = {n: chi2(d, sigma_disc) for n, d in DATA.items()}
    print(f"\nsigma_disc = {sigma_disc}; best chi2/bin per dataset:", {n: round(v[0].min() / v[1], 2) for n, v in c.items()})
    for label, (a, b) in (("old", ("Cole K", "Norberg bJ")), ("new", ("GAMA K", "GAMA r"))):
        total = c[a][0] + c[b][0]
        i = np.argmin(total)
        print(f"  {label} {a} + {b}: best joint chi2/bin {total[i] / (c[a][1] + c[b][1]):.2f} at {theta[i].round(2)} "
              f"(parts {c[a][0][i] / c[a][1]:.2f}, {c[b][0][i] / c[b][1]:.2f})")
