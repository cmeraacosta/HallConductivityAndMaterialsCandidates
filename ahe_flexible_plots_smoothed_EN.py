#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AHE plots (orbital- and spin-resolved) for the corrected 4×4 Hamiltonian,
with *noise-reduction* options:
  - Fermi-Dirac smearing (kT) to smooth the μ-dependence
  - Denominator regularization: |b|^2 → |b|^2 + eta_b^2
  - Optional (ky→-ky, τ→-τ) symmetrization to enforce mirror+τ symmetry

USAGE
-----
python ahe_flexible_plots_smoothed.py --N 41 --K 1.4 --mu-min -0.2 --mu-max 1.0 --n-mu 101 \
  --Qxy 0.3 --Qyz 0.25 --Qxz 0.4 --kT 0.02 --eta-b 1e-3 --symmetrize-y --outdir figs_smooth

Set --kT 0.0 to recover sharp Θ(μ−E). Increase N (and possibly K) for convergence.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl


mpl.rcParams.update({
    "font.size": 48,          # base size
    "axes.titlesize": 20,     # titles of each subplot
    "axes.labelsize": 20,     # axis labels (xlabel/ylabel)
    "xtick.labelsize": 48,    # numbers on the X axis
    "ytick.labelsize": 48,    # numbers on the Y axis
    "legend.fontsize": 48,    # legend
})

linestyle_p = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 5))),
     ('densely dotted',        (0, (1, 1))),

     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

# --------------------------- Argparse ---------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="AHE (orbital+spin resolved) with smearing & regularization")
    # Model parameters
    ap.add_argument("--alpha-x", type=float, default=1.0, help="Rashba α_x")  # Rashba symmeric
    ap.add_argument("--alpha-z", type=float, default=1.0, help="Rashba α_z")
    ap.add_argument("--mz",      type=float, default=1.0, help="m_z prefactor of σ_z terms")
    ap.add_argument("--J",       type=float, default=1.0, help="Exchange J")
    ap.add_argument("--p-x",     type=float, default=1.0, help="dipole p_x")  # px role: 
    ap.add_argument("--p-y",     type=float, default=0.2, help="dipole p_y")
    ap.add_argument("--p-z",     type=float, default=1.0, help="dipole p_z")
    ap.add_argument("--Qxx",     type=float, default=0.0, help="quadrupole Q_xx")
    ap.add_argument("--Qyy",     type=float, default=0.0, help="quadrupole Q_yy")
    ap.add_argument("--Qxy",     type=float, default=1.0, help="quadrupole Q_xy")
    ap.add_argument("--Qyz",     type=float, default=0.0, help="quadrupole Q_yz")
    ap.add_argument("--Qxz",     type=float, default=1.0, help="quadrupole Q_xz")
    # Integration grid and μ-range
    ap.add_argument("--K",       type=float, default=1.57, help="k-space cutoff; integrate over [-K,K]^3")
    ap.add_argument("--N",       type=int,   default=51,  help="grid points per axis (odd recommended)")
    ap.add_argument("--mu-min",  type=float, default=-1.0, help="min chemical potential")
    ap.add_argument("--mu-max",  type=float, default=1.0,  help="max chemical potential")
    ap.add_argument("--n-mu",    type=int,   default=101,   help="number of μ samples")
    # Noise-control options
    ap.add_argument("--kT",      type=float, default=0.021, help="Fermi-Dirac smearing kT (0 = Heaviside)")
    ap.add_argument("--eta-b",   type=float, default=5e-3, help="regularization for |b|: denom ← (|b|^2 + eta_b^2)^(3/2)")
    ap.add_argument("--symmetrize-y", action="store_true",
                    help="enforce (ky→-ky, τ→-τ) pairwise averaging for numerical cancellation")
    # Output
    ap.add_argument("--outdir",  type=str,   default="figs", help="directory to save PNG figures")
    ap.add_argument("--no-show", action="store_true", help="do not display figures; only save PNGs")
    return ap

# --------------------------- Model core ---------------------------
class Model:
    def __init__(self, par):
        self.alpha_x = par.alpha_x
        self.alpha_z = par.alpha_z
        self.mz = par.mz
        self.Jc = par.J
        self.px, self.py, self.pz = par.p_x, par.p_y, par.p_z
        self.Qxx, self.Qyy, self.Qxy, self.Qyz, self.Qxz = par.Qxx, par.Qyy, par.Qxy, par.Qyz, par.Qxz
        self.eta_b = par.eta_b

    def fields_grads(self, kx, ky, kz, tau):
        ax, az = self.alpha_x, self.alpha_z
        J, mz = self.Jc, self.mz

        # b-vector
        bx = az * ky
        by = -az * kx + ax * kz

        F  = (-self.Qxx * kx**2 + self.Qyy * ky**2) - (self.Qxx - self.Qyy) * kz**2
        M0 = J * self.py * ky + 2*J * (self.Qxy * kx - self.Qyz * kz) * ky
        Mz = J * self.px * kx + J * self.pz * kz + 2*J * ( F + self.Qxz * kx * kz )
        bz = -ax * ky + mz * ( M0 + tau * Mz )

        b = np.array([bx, by, bz], dtype=float)
        b2 = float(np.dot(b,b))
        bnorm = np.sqrt(b2)

        # derivatives for Ω
        dM0_dx = 2*J * self.Qxy * ky
        dM0_dy = J * self.py + 2*J * (self.Qxy * kx - self.Qyz * kz)
        dM0_dz = -2*J * self.Qyz * ky

        dF_dx = -2*self.Qxx * kx
        dF_dy =  2*self.Qyy * ky
        dF_dz = -2*(self.Qxx - self.Qyy) * kz

        dMz_dx = J * self.px + 2*J * ( dF_dx + self.Qxz * kz )
        dMz_dy = 2*J * ( dF_dy + 0.0 )
        dMz_dz = J * self.pz + 2*J * ( dF_dz + self.Qxz * kx )

        beta_x = mz * ( dM0_dx + tau * dMz_dx )
        beta_y = -ax + mz * ( dM0_dy + tau * dMz_dy )
        beta_z = mz * ( dM0_dz + tau * dMz_dz )

        # energies
        xi = kx*kx + ky*ky + kz*kz
        Eplus  = xi + bnorm
        Eminus = xi - bnorm

        db_dx = np.array([0.0, -self.alpha_z, beta_x], dtype=float)
        db_dy = np.array([self.alpha_z, 0.0,  beta_y], dtype=float)
        db_dz = np.array([0.0,  self.alpha_x, beta_z], dtype=float)

        return b, b2, bnorm, (db_dx, db_dy, db_dz), Eminus, Eplus

def berry_vector_from_grads(b, b2, grads, band, eta_b):
    db_dx, db_dy, db_dz = grads
    tp_x = float(np.dot(b, np.cross(db_dy, db_dz)))  # Ω^x numerator
    tp_y = float(np.dot(b, np.cross(db_dz, db_dx)))  # Ω^y numerator
    tp_z = float(np.dot(b, np.cross(db_dx, db_dy)))  # Ω^z numerator
    denom = (b2 + eta_b*eta_b)**1.5
    if denom < 1e-24:
        return np.zeros(3)
    pref = +0.5 if band == 'lower' else -0.5
    return pref * np.array([tp_x, tp_y, tp_z]) / denom

def spin_weights_z(b, bnorm, band):
    cos_th = b[2]/bnorm if bnorm > 1e-18 else 0.0
    if band == 'upper':
        w_up = 0.5*(1.0 + cos_th)
        w_dn = 0.5*(1.0 - cos_th)
    else:
        w_up = 0.5*(1.0 - cos_th)
        w_dn = 0.5*(1.0 + cos_th)
    return w_up, w_dn

def f_occ(E, mu, kT):
    if kT <= 0.0:
        return 1.0 if (E <= mu) else 0.0
    # Fermi-Dirac
    x = (E - mu)/kT
    # stable numerics:
    if x > 40:  return 0.0
    if x < -40: return 1.0
    return 1.0/(1.0 + np.exp(x))

# --------------------------- Integration & plotting ---------------------------
def compute_sigma_vs_mu(par):
    """Return dict: res[(tau_label, spin_label)][comp] = array over μ.
       comp ∈ {'xy','yz','xz'}, tau_label ∈ {'tau=+','tau=-','total'}, spin_label ∈ {'up','dn','total'}.
       Units: e^2/h.
    """
    model = Model(par)

    ks = np.linspace(-par.K, par.K, par.N)
    dk = ks[1]-ks[0]
    w_cell = (dk**3) / (2*np.pi)**3
    mu_grid = np.linspace(par.mu_min, par.mu_max, par.n_mu)

    comp_idx = {'yz':0, 'xz':1, 'xy':2}
    cases = [('tau=+','up'),('tau=+','dn'),('tau=-','up'),('tau=-','dn'),('total','total')]
    res = {c: {comp: np.zeros_like(mu_grid) for comp in comp_idx} for c in cases}

    # Precompute entries; optionally symmetrize on the fly
    entries = []
    for kx in ks:
        for ky in ks:
            for kz in ks:
                for tau in (+1,-1):
                    b, b2, bnorm, grads, Emin, Emax = model.fields_grads(kx,ky,kz,tau)
                    OmL = berry_vector_from_grads(b, b2, grads, 'lower', model.eta_b)
                    OmU = berry_vector_from_grads(b, b2, grads, 'upper', model.eta_b)
                    wUL, wDL = spin_weights_z(b, bnorm, 'lower')
                    wUU, wDU = spin_weights_z(b, bnorm, 'upper')

                    if par.symmetrize_y:
                        # Mirror pair: (kx,-ky,kz,-tau)
                        b_p, b2_p, bn_p, grads_p, Emin_p, Emax_p = model.fields_grads(kx,-ky,kz,-tau)
                        OmL_p = berry_vector_from_grads(b_p, b2_p, grads_p, 'lower', model.eta_b)
                        OmU_p = berry_vector_from_grads(b_p, b2_p, grads_p, 'upper', model.eta_b)
                        wUL_p, wDL_p = spin_weights_z(b_p, bn_p, 'lower')
                        wUU_p, wDU_p = spin_weights_z(b_p, bn_p, 'upper')
                        # average both contributions (Ω and weights; energies map exactly by symmetry)
                        OmL = 0.5*(OmL + OmL_p)
                        OmU = 0.5*(OmU + OmU_p)
                        wUL = 0.5*(wUL + wUL_p); wDL = 0.5*(wDL + wDL_p)
                        wUU = 0.5*(wUU + wUU_p); wDU = 0.5*(wDU + wDU_p)

                    entries.append(dict(
                        tau=tau, Emin=Emin, Emax=Emax, w=w_cell,
                        OmL_up=wUL*OmL, OmL_dn=wDL*OmL,
                        OmU_up=wUU*OmU, OmU_dn=wDU*OmU
                    ))

    # Accumulate as function of μ with smearing
    for m_idx, mu in enumerate(mu_grid):
        sums = {c: {comp:0.0 for comp in comp_idx} for c in cases}
        for e in entries:
            fL = f_occ(e['Emin'], mu, par.kT)
            fU = f_occ(e['Emax'], mu, par.kT)
            for comp, idx in comp_idx.items():
                sums[('total','total')][comp] += (e['OmL_up'][idx] + e['OmL_dn'][idx]) * fL * e['w']
                sums[('total','total')][comp] += (e['OmU_up'][idx] + e['OmU_dn'][idx]) * fU * e['w']

                lab = 'tau=+' if e['tau']==+1 else 'tau=-'
                sums[(lab,'up')][comp] += e['OmL_up'][idx] * fL * e['w']
                sums[(lab,'dn')][comp] += e['OmL_dn'][idx] * fL * e['w']
                sums[(lab,'up')][comp] += e['OmU_up'][idx] * fU * e['w']
                sums[(lab,'dn')][comp] += e['OmU_dn'][idx] * fU * e['w']

        # Convert to e^2/h: σ/(e^2/h) = −2π * I
        for case in cases:
            for comp in comp_idx:
                res[case][comp][m_idx] = -2*np.pi * sums[case][comp]

    return res, mu_grid

#STYLE = {
#    ('tau=+','up'):   dict(color='blue', linestyle='-', linewidth=9.0),   # solid blue
#    ('tau=-','up'):   dict(color='mediumslateblue', linestyle='-', linewidth=9.0),  # naranja a trazos
#    ('tau=+','dn'):   dict(color='red', linestyle=(0, (5, 10)), linewidth=9.0),  # blue dashed
#    ('tau=-','dn'):   dict(color='magenta', linestyle=(0, (5, 10)), linewidth=9.0),   # solid orange
#    ('total','total'):dict(color='k',  linestyle='-', linewidth=11.0, marker=None) # thick black
#}


STYLE = {
    ('tau=+','up'):   dict(color='blue', linestyle='-', linewidth=9.0),   # solid blue
    ('tau=-','up'):   dict(color='cornflowerblue', linestyle='-', linewidth=9.0),  # naranja a trazos
    ('tau=+','dn'):   dict(color='red', linestyle=(0, (5, 5)), linewidth=9.0),  # blue dashed
    ('tau=-','dn'):   dict(color='darkorange', linestyle=(0, (5, 5)), linewidth=9.0),   # solid orange
    ('total','total'):dict(color='k',  linestyle='-', linewidth=10.0, marker=None) # thick black
}
#mediumslateblue

def make_plots_panel(res, mu_grid, outdir, show=True):
    os.makedirs(outdir, exist_ok=True)

    comps  = ['xy','yz','xz']
#    titles = [r'$\sigma_{xy}(\mu)$', r'$\sigma_{yz}(\mu)$', r'$\sigma_{xz}(\mu)$']
    curves = [
        (('tau=+','up'),    '+, ↑'),
        (('tau=-','up'),    '−, ↑'),
        (('tau=+','dn'),    '+, ↓'),
        (('tau=-','dn'),    '−, ↓'),
        (('total','total'), 'TOTAL'),
    ]

    # ① Figure WITHOUT constrained_layout and with an extra row for the legend
    fig = plt.figure(figsize=(30, 10), constrained_layout=False)
    gs  = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[0.8, 0.42])  # 24% for the legend band
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # ② Plots (sin tight_layout)
    for ax, comp in zip(axes, comps):
        for key, lab in curves:
            ax.plot(mu_grid, res[key][comp], **STYLE.get(key, {}))
        ax.set_xlabel(r'$\mu$')
#        ax.set_ylabel(rf'$\sigma_{{{comp}}}/(e^2/h)$')
#        ax.set_title(ttl)
        # some margin so ticks don't touch the legend band
        ax.margins(x=0.02, y=0.08)

    # ③ Legend on a dedicated axis (no axes → never overlaps)
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis('off')
    handles = [Line2D([0],[0], **STYLE[k], label=lab) for k, lab in curves]
#    labels  = ['+, ↑', '+, ↓', '−, ↑', '−, ↓', 'TOTAL']
    labels  = ['+, ↑', '−, ↑', '+, ↓', '−, ↓', 'TOTAL']
    ax_leg.legend(handles=handles, labels=labels, loc='center', ncol=5, frameon=False)

    # ④ Adjust margins between subplots and edges (do NOT use tight_layout here)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.10, wspace=0.36, hspace=0.0)

    fig.savefig(os.path.join(outdir, "sigma_panel_xy_yz_xz.png"), dpi=300)
    if not show:
        plt.close(fig)

## --- new function: horizontal figure with 3 subplots ---
#def make_plots_panel(res, mu_grid, outdir, show=True):
#    import os, matplotlib.pyplot as plt
#    os.makedirs(outdir, exist_ok=True)
#    comps = ['xy','yz','xz']       # orden: σ_xy, σ_yz, σ_xz
#    titles = [r'$\sigma_{xy}(\mu)$', r'$\sigma_{yz}(\mu)$', r'$\sigma_{xz}(\mu)$']
#
#    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False, constrained_layout=True)
#
#    curves = [
#        (('tau=+','up'),    '+, ↑'),
#        (('tau=+','dn'),    '+, ↓'),
#        (('tau=-','up'),    '−, ↑'),
#        (('tau=-','dn'),    '−, ↓'),
#        (('total','total'), 'TOTAL'),
#    ]
#
#    for ax, comp, ttl in zip(axes, comps, titles):
#        for key, lab in curves:
#            style = STYLE.get(key, {})
#            ax.plot(mu_grid, res[key][comp], **style)
#        ax.set_xlabel(r'$\mu$')
#        ax.set_ylabel(rf'$\sigma_{{{comp}}}/(e^2/h)$')
#        ax.set_title(ttl)
#
#
#    legend_elems = [Line2D([0],[0], **STYLE[k], label=lab) for k, lab in curves]
#    fig.legend(handles=legend_elems,
#               loc='upper center', bbox_to_anchor=(0.5, 1.02),
#               ncol=5, frameon=False)
#
#    # 4) leave space at the top for the legend
#    plt.subplots_adjust(top=0.86)
#
#    # global legend (single, below)
##    legend_elems = [Line2D([0],[0], **STYLE[k], label=lab) for k, lab in curves]
##    fig.legend(handles=legend_elems, loc='lower center', ncol=5, frameon=False)
##    plt.subplots_adjust(bottom=0.00)  # space for the legend
#
#    fig.savefig(os.path.join(outdir, "sigma_panel_xy_yz_xz.png"), dpi=300)
#    if not show:
#        plt.close(fig)



#
# --------------------------- Main ---------------------------
def main():
    parser = build_parser()
    par = parser.parse_args()
    res, mu_grid = compute_sigma_vs_mu(par)
#    make_plots(res, mu_grid, par.outdir, show=(not par.no_show))
    make_plots_panel(res, mu_grid, par.outdir, show=(not par.no_show))

    print(f"Saved figures in: {par.outdir}")
    print("Figures: sigma_xy_by_tau_spin.png, sigma_yz_by_tau_spin.png, sigma_xz_by_tau_spin.png")
    print(f"Settings: N={par.N}, K={par.K}, kT={par.kT}, eta_b={par.eta_b}, symmetrize_y={par.symmetrize_y}")

if __name__ == "__main__":
    main()