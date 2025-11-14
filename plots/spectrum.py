import numpy as np
import matplotlib.pyplot as plt
import yt
from utils import *

def doit(ds):
    # a FFT operates on uniformly gridded data.  We'll use the yt
    # covering grid for this.

    max_level = ds.index.max_level

    # Determine the refinement factor: yt datasets often provide 'ref_factors'
    # as a list, while Parthenon-based datasets expose 'ires_factor'. Use
    # ref_factors if available, otherwise fall back to ires_factor, or 1.
    if hasattr(ds, "ref_factors"):
        ref = int(np.prod(ds.ref_factors[0:max_level]))
    elif hasattr(ds, "ires_factor"):
        ref = int(ds.ires_factor)
    else:
        ref = 1

    low = ds.domain_left_edge
    dims = ds.domain_dimensions * ref

    nx, ny, nz = dims

    nindex_rho = 1.0 / 3.0

    Kk = np.zeros((nx // 2 + 1, ny // 2 + 1, nz // 2 + 1))

    for vel in [("gas", "velocity_x"), ("gas", "velocity_y"), ("gas", "velocity_z")]:
        Kk += 0.5 * fft_comp(
            ds, ("gas", "density"), vel, nindex_rho, max_level, low, dims
        )

    # wavenumbers
    L = (ds.domain_right_edge - ds.domain_left_edge).d

    kx = np.fft.rfftfreq(nx) * nx / L[0]
    ky = np.fft.rfftfreq(ny) * ny / L[1]
    kz = np.fft.rfftfreq(nz) * nz / L[2]

    # physical limits to the wavenumbers
    kmin = np.min(1.0 / L)
    kmax = np.min(0.5 * dims / L)

    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)

    # bin the Fourier KE into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    whichbin = np.digitize(k.flat, kbins)
    ncount = np.bincount(whichbin)

    E_spectrum = np.zeros(len(ncount) - 1)

    for n in range(1, len(ncount)):
        E_spectrum[n - 1] = np.sum(Kk.flat[whichbin == n])

    k = 0.5 * (kbins[0 : N - 1] + kbins[1:N])
    E_spectrum = E_spectrum[1:N]

    index = np.argmax(E_spectrum)
    kmax = k[index]
    Emax = E_spectrum[index]

    plt.loglog(k , E_spectrum* (k / kmax) ** (5.0/3.0))
    #plt.loglog(k, Emax * (k / kmax) ** (-5.0 / 3.0), ls=":", color="0.5")
    #plt.loglog(k, Emax * (k / kmax) ** (-2.0), ls=":", color="red")

    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k) * 5/3 dk$")

    plt.savefig("spectrum_k45.png")


def fft_comp(ds, irho, iu, nindex_rho, level, low, delta):
    cube = ds.covering_grid(level, left_edge=low, dims=delta, fields=[irho, iu])

    rho = cube[irho].d
    u = cube[iu].d

    nx, ny, nz = rho.shape

    # do the FFTs -- note that since our data is real, there will be
    # too much information here.  fftn puts the positive freq terms in
    # the first half of the axes -- that's what we keep.  Our
    # normalization has an '8' to account for this clipping to one
    # octant.
    ru = np.fft.fftn(rho**nindex_rho * u)[
        0 : nx // 2 + 1, 0 : ny // 2 + 1, 0 : nz // 2 + 1
    ]
    ru = 8.0 * ru / (nx * ny * nz)

    return np.abs(ru) ** 2


if __name__ == "__main__":

    N_procs, user_args = get_n_procs_and_user_args()

    ds = yt.load(user_args[0])
    doit(ds)