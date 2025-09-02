#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fermi contour series generator for the FULL Hamiltonian H = H_p + H_Q
with optional periodic (sin/cos) mapping or continuum k·p mapping.

- Spin space: Pauli σ; magnetic sublattice: Pauli τ.
- Eigenvalues (analytically): E_{s,τ}(k) = ξ(k) + s * m_z [ τ A(k) + B(k) ],
  where A = J p_z k_z + tilde{J}_{Q_z}(k) and B = J p_y k_y + tilde{J}_{Q_y}(k)
  (or their periodicized versions).
- This script plots Fermi contours E_{s,τ}(k_x, k_y, k_z) = μ in the (k_y,k_z) plane for
  a list of μ's and a fixed k_x. Also includes an optional 3D “lift” view.

The τ index is the MAGNETIC SUBLATTICE.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 38,          # base size
    "axes.titlesize": 38,     # titles of each subplot
    "axes.labelsize": 38,     # axis labels (xlabel/ylabel)
    "xtick.labelsize": 38,    # X-axis tick numbers
    "ytick.labelsize": 38,    # Y-axis tick numbers
    "legend.fontsize": 28,    # legend
})

# -----------------------
# Model configuration
# -----------------------
a = 1.0  # Lattice constant

# Kinetic prefactor for xi(k) = kx^2 + ky^2 + kz^2 (or its periodicization)
xi = 0.5

# Spin–sublattice exchange strength (multiplies σ_z)
mz = 1.0

# --- Linear magneto-electric exchange (H_p): J * p_i * k_i ---
# Keep J as a global scale so J*p_i reproduces your earlier "effective" constants.
J = 1.0
p_z = 1.0   # so J*p_z == 1.2  (your previous Jpar_pz)
p_y = 0.2   # so J*p_y == 0.4  (your previous Jper_py)

# --- Quadratic magneto-electric exchange (H_Q) ---
# tilde{J}_{Q_z} = 2J( -Q_xx kx^2 + Q_yy ky^2 - (Q_xx - Q_yy) kz^2 + Q_xz kx kz )
# tilde{J}_{Q_y} = 2J( Q_xy kx - Q_yz kz ) ky
# Set all Q's to zero to recover your original model.
Q_xx = 0.00
Q_yy = 0.00
Q_xz = 1.0
Q_xy = 1.00 
Q_yz = 0.0 #Affect

# Requested defaults for visualization
ELEV_DEFAULT = 15
AZIM_DEFAULT = 105
KX_DEFAULT = 0.0

# A single color for all bands (as requested)
#BAND_COLOR = "0.25"  # gray; can use "#555555" or "black"
#BAND_COLORS = [BAND_COLOR]*4
BAND_COLORS = ["blue", "cornflowerblue", "red", "darkorange"]
BAND_LABELS = [r"$s=+,\ \tau=+$", r"$s=+,\ \tau=-$",
               r"$s=-,\ \tau=+$", r"$s=-,\ \tau=-$"]
BAND_w = ["solid", "solid", "dashed", "dashed"]
STYLE = {
    ('tau=+','up'):   dict(color='blue', linestyle='-', linewidth=8.0),   # blue solid
    ('tau=-','up'):   dict(color='royalblue', linestyle='-', linewidth=8.0),  # orange dashed
    ('tau=+','dn'):   dict(color='red', linestyle=(0, (5, 10)), linewidth=8.0),  # azul a trazos
    ('tau=-','dn'):   dict(color='red', linestyle=(0, (5, 10)), linewidth=8.0),   # orange solid
    ('total','total'):dict(color='k',  linestyle='-', linewidth=6.0, marker=None) # thick black
}

# -----------------------
# Periodic mapping helpers
# -----------------------
def K2_periodic(kx, ky, kz, a=1.0):
    """Sum_i [2/a^2 (1 - cos(k_i a))]  -- periodic quadratic."""
    return (2.0/a**2) * ((1 - np.cos(kx*a)) + (1 - np.cos(ky*a)) + (1 - np.cos(kz*a)))

def sines(kx, ky, kz, a=1.0):
    """Periodic linear terms: sin(k_i a)/a."""
    return np.sin(kx*a)/a, np.sin(ky*a)/a, np.sin(kz*a)/a

def quad_axes_periodic(kx, ky, kz, a=1.0):
    """Axis-resolved periodic quadratics k_i^2 -> 2(1 - cos(k_i a))/a^2."""
    kx2 = 2.0*(1 - np.cos(kx*a))/a**2
    ky2 = 2.0*(1 - np.cos(ky*a))/a**2
    kz2 = 2.0*(1 - np.cos(kz*a))/a**2
    return kx2, ky2, kz2

def cross_periodic(kx, kz, a=1.0):
    """Periodic mapping for kx*kz -> (sin(kx a)/a)*(sin(kz a)/a)."""
    sx = np.sin(kx*a)/a
    sz = np.sin(kz*a)/a
    return sx*sz


# -----------------------
# Energy dispersion: full H = H_p + H_Q
# -----------------------
def E_bands(kx, ky, kz, periodic=False):
    """
    Four bands E_{s,τ}(k) for fixed kx over a (ky,kz) grid.
    periodic=True -> periodicized k-mapping (sin/cos)
    periodic=False -> continuum k·p mapping
    Returns array (..., 4) in order: [(s=+1,τ=+1), (+,-), (-,+), (-,-)].
    """
    if periodic:
        # base kinetic xi(k)
        K2 = K2_periodic(kx, ky, kz, a)
        base = xi * K2

        # linear components (H_p): J p_i k_i  with k_i -> sin(ka)/a
        sx, sy, sz = sines(kx, ky, kz, a)     # linear mapping
        A_lin = J * p_z * sz                   # ~ k_z
        B_lin = J * p_y * sy                   # ~ k_y

        # quadratic components (H_Q): axis-resolved + cross
        kx2, ky2, kz2 = quad_axes_periodic(kx, ky, kz, a)
        kxkz = cross_periodic(kx, kz, a)

        # for B_quad, the factor (Q_xy kx - Q_yz kz) * ky  -> use linear periodic mapping
        kx_lin = sx
        kz_lin = sz
        ky_lin = sy

    else:
        # Continuum mapping
        K2 = kx**2 + ky**2 + kz**2
        base = xi * K2

        # linear (H_p)
        A_lin = J * p_z * kz
        B_lin = J * p_y * ky

        # quadratic (H_Q)
        kx2, ky2, kz2 = kx**2, ky**2, kz**2
        kxkz = kx * kz

        kx_lin = kx
        kz_lin = kz
        ky_lin = ky

    # tilde{J}_{Q_z} and tilde{J}_{Q_y}
    A_quad = 2.0*J * ( -Q_xx*kx2 + Q_yy*ky2 - (Q_xx - Q_yy)*kz2 + Q_xz*kxkz )
    B_quad = 2.0*J * (  Q_xy*kx_lin - Q_yz*kz_lin ) * ky_lin

    # Collect A(k), B(k)
    A = A_lin + A_quad
    B = B_lin + B_quad

    # Eigenvalues: E_{s,τ} = xi + s*mz (τ A + B)
    E_pp = base + mz * ( +A + B )   # s=+1, τ=+1
    E_pm = base + mz * ( -A + B )   # s=+1, τ=-1
    E_mp = base - mz * ( +A + B )   # s=-1, τ=+1
    E_mm = base - mz * ( -A + B )   # s=-1, τ=-1

    return np.stack([E_pp, E_pm, E_mp, E_mm], axis=-1)


# -----------------------
# Plotting utilities
# -----------------------
def plot_fermi_contours(kyg, kzg, Egrid, mu, periodic=False, ax=None, title_prefix=""):
    """
    Plot 2D Fermi contours E_{s,τ}(kx, ky, kz) = μ in ky–kz plane.
    Returns the Matplotlib Axes used.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12.4, 11.2))

    for i in range(4):
        ax.contour(kyg, kzg, Egrid[:, :, i] - mu, levels=[0.0],
                   linewidths=8, colors=[BAND_COLORS[i]], linestyles=[BAND_w[i]])

    ax.axhline(0, linestyle="--", linewidth=0.8, color="0.6")
    ax.axvline(0, linestyle="--", linewidth=0.8, color="0.6")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_z$")
    per_str = "periodic" if periodic else "continuum"
#    ax.set_title(f"{title_prefix}Fermi contours at μ={mu:.3f} ({per_str})")
    ax.set_title(f"{title_prefix} μ={mu:.3f}")

#    legend_elements = [Line2D([0], [0], color=BAND_COLORS[i], lw=2.5) for i in range(4)]
#    ax.legend(legend_elements, BAND_LABELS, loc="best", fontsize=8, framealpha=0.9)
    return ax


def plot_fermi_contours_3d(kyg, kzg, Egrid, mu, periodic=False,
                           elev=ELEV_DEFAULT, azim=AZIM_DEFAULT, clean=True):
    """
    Optional 3D visualization: draw 2D contours and lift them to z=μ,
    using the requested camera angles. If clean=True, remove axes/box.
    """
    fig = plt.figure(figsize=(14.0, 11.6))
    ax2d = plt.axes([0.08, 0.10, 0.34, 0.80])
    ax3d = fig.add_axes([0.52, 0.08, 0.46, 0.84], projection="3d")

    if clean:
        ax3d.set_axis_off()

    # Left: 2D contours
    plot_fermi_contours(kyg, kzg, Egrid, mu, periodic=periodic, ax=ax2d, title_prefix="")

    # Extract the contour paths from the 2D Axes and lift to z=μ in 3D
    for coll in ax2d.collections:
        for path in coll.get_paths():
            v = path.vertices
            ax3d.plot(v[:, 0], v[:, 1], zs=mu, lw=2.0, color=coll.get_edgecolor()[0])

    if not clean:
        ax3d.set_xlabel(r"$k_y$")
        ax3d.set_ylabel(r"$k_z$")
        ax3d.set_zlabel("Energy")
    ax3d.view_init(elev=elev, azim=azim)
    per_str = "periodic" if periodic else "continuum"
    ax3d.set_title(f"3D: contours at z=μ={mu:.3f} ({per_str})")
    plt.tight_layout()
    return fig


# -----------------------
# Driver to make a series
# -----------------------
def fermi_series(mu_list, kx=KX_DEFAULT, periodic=False, N=601,
                 save_each=True, save_prefix="fermi_fullHQ_kx0_mu_",
                 make_panel=True, panel_cols=3,
                 plot_3d=True, elev=ELEV_DEFAULT, azim=AZIM_DEFAULT, clean3d=True):
    """
    Generate a series of Fermi contour plots for a list of μ.
    """
    # ky-kz grid over the first BZ (or a symmetric window for continuum)
    kyg = np.linspace(-np.pi, np.pi, N)
    kzg = np.linspace(-np.pi, np.pi, N)
    KY, KZ = np.meshgrid(kyg, kzg, indexing="xy")

    # Compute the 4-band energy grid once per (kx, periodic)
    Egrid = E_bands(kx, KY, KZ, periodic=periodic)

    # Save individual figures
    paths = []
    for mu in mu_list:
        fig, ax = plt.subplots(figsize=(12.4, 11.2))
        plot_fermi_contours(kyg, kzg, Egrid, mu, periodic=periodic, ax=ax,
                            title_prefix=f"kx={kx:.3f} — ")
        out = f"{save_prefix}{mu:.3f}.png"
        if save_each:
            fig.savefig(out, dpi=160, bbox_inches="tight")
            paths.append(out)
        plt.close(fig)

        if plot_3d:
            fig3d = plot_fermi_contours_3d(kyg, kzg, Egrid, mu, periodic=periodic,
                                           elev=elev, azim=azim, clean=clean3d)
            out3d = f"{save_prefix}{mu:.3f}_3d.png"
            fig3d.savefig(out3d, dpi=160, bbox_inches="tight")
            plt.close(fig3d)
            paths.append(out3d)

    # Multi-panel with all μ
    panel_path = None
    if make_panel:
        rows = int(np.ceil(len(mu_list) / panel_cols))
        fig, axes = plt.subplots(rows, panel_cols, figsize=(6.2*panel_cols, 5.6*rows))
        axes = np.atleast_2d(axes)
        for idx, mu in enumerate(mu_list):
            r, c = divmod(idx, panel_cols)
            plot_fermi_contours(kyg, kzg, Egrid, mu, periodic=periodic, ax=axes[r, c],
                                title_prefix="")
#            axes[r, c].set_title(axes[r, c].get_title() + f"\n$k_x={kx:.3f}$")
            axes[r, c].set_title(axes[r, c].get_title() )

        # Hide any unused axes
        for j in range(len(mu_list), rows*panel_cols):
            r, c = divmod(j, panel_cols)
            axes[r, c].axis("off")
        fig.tight_layout()
        panel_path = f"{save_prefix}panel.png"
        fig.savefig(panel_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    return paths, panel_path


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Define μ values (step of 0.1 as requested earlier)
    mu_values = np.arange(-1.2, 1.0, 0.1)

    # Toggle 'periodic' to True for periodicized model
    paths, panel = fermi_series(mu_values,
                                kx=KX_DEFAULT,
                                periodic=False,      # <-- set True to use periodic sin/cos mapping
                                N=401,
                                save_each=True,
                                save_prefix="fermi_fullHQ_kx0_mu_",
                                make_panel=True,
                                panel_cols=3,
                                plot_3d=True,
                                elev=ELEV_DEFAULT,
                                azim=AZIM_DEFAULT,
                                clean3d=True)

    print("Saved individual plots:", paths)
    print("Saved panel:", panel)
