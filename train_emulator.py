#!/usr/bin/env python3
"""Train a Gaussian process emulator of the z=0 SDSS-r luminosity function.

Inputs are the three SN feedback parameters (alphahot, vhot, alpha_reheat) of the
50 COLIBRE-L100m6 emu_ runs. The output is log10(phi) in each magnitude bin.
Runs 0-39 train the emulator, runs 40-49 and the fb_ runs test it.

    emulator_test.png         emulator vs GALFORM for the test runs
    emulator_sensitivity.png  how the LF changes as each parameter is varied

module load python/3.12.4
python3 train_emulator.py
"""

import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

ROOT = "/cosma5/data/durham/dc-lock1"
SIM, IZ = "COLIBRE-L100m6", "iz127"
N_TRAIN = 40

# parameter ranges of the design (alpha_reheat was sampled in log)
NAMES = ["alphahot", "vhot", "alpha_reheat"]
LOW = np.array([2.0, 200.0, np.log10(0.3)])
HIGH = np.array([4.5, 450.0, np.log10(1.5)])
FID = np.array([3.2, 320.0, 0.64])

# the fb_ runs, with their parameter values
FB_RUNS = {
    "fid": FID,
    "alphahot2.2": [2.2, 320, 0.64], "alphahot4.2": [4.2, 320, 0.64],
    "vhot250": [3.2, 250, 0.64], "vhot400": [3.2, 400, 0.64],
    "alpha_reheat0.32": [3.2, 320, 0.32], "alpha_reheat1.28": [3.2, 320, 1.28],
}

# gal.lf: column 0 is magnitude, then 4 columns per band (no dust, err, dust, err).
# Band 14 is SDSS-r (rest), so its dust-attenuated LF is column 1 + 4*13 + 2.
COL = 1 + 4 * 13 + 2
MAG_RANGE = (-23.5, -16)  # fit only these magnitudes (outside, the LF is noisy or zero)


def read_lf(folder):
    """Mean LF over the subvolumes (each is already normalised by its own volume)."""
    files = glob.glob(f"{ROOT}/{folder}/{SIM}/lc16/{IZ}/ivol*/gal.lf")
    data = [np.loadtxt(f) for f in files]
    return data[0][:, 0], np.mean([d[:, COL] for d in data], axis=0)


def scale(params):
    """Map parameters onto the unit cube, which is what the GP sees."""
    p = np.array(params, dtype=float).reshape(-1, 3).copy()
    p[:, 2] = np.log10(p[:, 2])
    return (p - LOW) / (HIGH - LOW)


# ---- read the runs ---------------------------------------------------------
design = np.loadtxt(f"{ROOT}/emu_design.csv", delimiter=",", skiprows=1)
params = design[:, 1:]
mag, _ = read_lf("emu_000")
lfs = np.array([read_lf(f"emu_{int(i):03d}")[1] for i in design[:, 0]])
fb_lfs = np.array([read_lf(f"fb_{name}")[1] for name in FB_RUNS])

use = (mag >= MAG_RANGE[0]) & (mag <= MAG_RANGE[1])
use &= (lfs > 0).all(axis=0) & (fb_lfs > 0).all(axis=0)
mag = mag[use]
y = np.log10(lfs[:, use])
y_fb = np.log10(fb_lfs[:, use])
print(f"{len(mag)} magnitude bins from {mag.min():.2f} to {mag.max():.2f}")

# ---- train -----------------------------------------------------------------
# one length scale per parameter, plus a small noise term for the run-to-run scatter
kernel = ConstantKernel(1.0) * RBF([0.5, 0.5, 0.5], (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-6, 1e-1))
gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
gp.fit(scale(params[:N_TRAIN]), y[:N_TRAIN])
print("fitted kernel:", gp.kernel_)

# ---- test ------------------------------------------------------------------
pred_test = gp.predict(scale(params[N_TRAIN:]))
pred_fb = gp.predict(scale(list(FB_RUNS.values())))
err_test = pred_test - y[N_TRAIN:]
err_fb = pred_fb - y_fb
print(f"test runs 40-49: rms error in log10(phi) = {np.sqrt(np.mean(err_test**2)):.3f} dex,"
      f" worst = {np.abs(err_test).max():.3f} dex")
print(f"fb_ runs:        rms error in log10(phi) = {np.sqrt(np.mean(err_fb**2)):.3f} dex,"
      f" worst = {np.abs(err_fb).max():.3f} dex")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for k in range(len(pred_test)):
    axes[0].plot(mag, y[N_TRAIN + k], color=f"C{k}")
    axes[0].plot(mag, pred_test[k], color=f"C{k}", ls="--")
    axes[1].plot(mag, err_test[k], color=f"C{k}")
for k in range(len(pred_fb)):
    axes[1].plot(mag, err_fb[k], color="grey", lw=0.8)
axes[0].plot([], [], "k-", label="GALFORM")
axes[0].plot([], [], "k--", label="emulator")
axes[0].set_ylabel(r"$\log_{10}(dn/dM\ /\ h^{3}\,{\rm Mpc}^{-3}\,{\rm mag}^{-1})$")
axes[0].set_title("Test runs 40-49")
axes[0].legend()
axes[1].plot([], [], "C0-", label="test runs 40-49")
axes[1].plot([], [], color="grey", lw=0.8, label="fb_ runs")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("emulator - GALFORM  [dex]")
axes[1].set_title("Emulator error")
axes[1].legend()
for ax in axes:
    ax.set_xlabel(r"$M_{\rm AB} - 5\log_{10} h$  (SDSS r)")
    ax.set_xlim(mag.max() + 0.2, mag.min() - 0.2)  # fainter on the left
    ax.grid(alpha=0.4)
fig.tight_layout()
fig.savefig("emulator_test.png", dpi=150)

# ---- sensitivity -----------------------------------------------------------
# vary one parameter across its whole range, keeping the other two at lc16 values
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
y_fid = gp.predict(scale(FID))[0]
for j, ax in enumerate(axes):
    values = np.linspace(LOW[j], HIGH[j], 6)
    if j == 2:
        values = 10**values
    colours = plt.cm.viridis(np.linspace(0, 1, len(values)))
    for v, c in zip(values, colours):
        p = FID.copy()
        p[j] = v
        mean, std = gp.predict(scale(p), return_std=True)
        ax.plot(mag, mean[0] - y_fid, color=c, label=f"{NAMES[j]} = {v:.3g}")
        ax.fill_between(mag, mean[0] - y_fid - std[0], mean[0] - y_fid + std[0], color=c, alpha=0.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$M_{\rm AB} - 5\log_{10} h$  (SDSS r)")
    ax.set_xlim(mag.max() + 0.2, mag.min() - 0.2)
    ax.set_title(f"varying {NAMES[j]}  (others at lc16)")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)
axes[0].set_ylabel(r"$\log_{10}(\phi / \phi_{\rm fid})$  [dex]")
fig.tight_layout()
fig.savefig("emulator_sensitivity.png", dpi=150)
