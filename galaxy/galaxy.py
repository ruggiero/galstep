# Use: python galaxy.py [FILE]

from os import path
from sys import exit

import numpy as np
import numpy.random as nprand
from scipy.optimize import brentq

from optimized_functions import phi
from snapwrite import process_input, write_snapshot


halo_core = True
bulge_core = True
G = 43007.1


def main():
    init()
    galaxy_data = generate_galaxy()
    write_input_file(galaxy_data)


def generate_galaxy():
    coords_halo = set_halo_positions()
    coords_disk = set_disk_positions()
    coords_bulge = set_bulge_positions()
    coords = np.concatenate((coords_halo, coords_disk, coords_bulge))
    vels = set_velocities(coords)
    return [coords, vels]


def init():
    global M_halo, M_disk, M_bulge
    global N_halo, N_disk, N_bulge
    global a_halo, a_bulge, Rd, z0
    global N_total, M_total

    if not (path.isfile("header.txt") and path.isfile("galaxy_param.txt")):
        print "header.txt or galaxy_param.txt missing."
        exit(0)

    vars_ = process_input("galaxy_param.txt")
    M_halo, M_disk, M_bulge = (float(i[0]) for i in vars_[0:3])
    N_halo, N_disk, N_bulge = (float(i[0]) for i in vars_[3:6])
    a_halo, a_bulge, Rd, z0 = (float(i[0]) for i in vars_[6:10])
    M_total = M_disk + M_bulge + M_halo
    N_total = N_disk + N_bulge + N_halo


def dehnen_inverse_cumulative(Mc, M, a, core):
    if(core):
        return ((a * (Mc**(2/3.)*M**(4/3.) + Mc*M + Mc**(4/3.)*M**(2/3.))) /
                   (Mc**(1/3.) * M**(2/3.) * (M-Mc)))
    else:
        return (a * ((Mc*M)**0.5 + Mc)) / (M-Mc)


def set_halo_positions():
    # The factor M * 200^2 / 201^2 restricts the radius to 200 * a.
    radii = dehnen_inverse_cumulative(nprand.sample(N_halo) *
        ((M_halo*40000) / 40401), M_halo, a_halo, halo_core)
    thetas = np.arccos(nprand.sample(N_halo)*2 - 1)
    phis = 2 * np.pi * nprand.sample(N_halo)
    xs = radii * np.sin(thetas) * np.cos(phis)
    ys = radii * np.sin(thetas) * np.sin(phis)
    zs = radii * np.cos(thetas)

    # Older NumPy versions freak out without this line.
    coords = np.column_stack((xs, ys, zs))
    coords = np.array(coords, order='C')
    coords.shape = (1, -1) # Linearizing the array.
    return coords[0]


def set_bulge_positions():
    radii = dehnen_inverse_cumulative(nprand.sample(N_bulge) *
        ((M_bulge*40000) / 40401), M_bulge, a_bulge, bulge_core)
    thetas = np.arccos(nprand.sample(N_bulge)*2 - 1)
    phis = 2 * np.pi * nprand.sample(N_bulge)
    xs = radii * np.sin(thetas) * np.cos(phis)
    ys = radii * np.sin(thetas) * np.sin(phis)
    zs = radii * np.cos(thetas)
    coords = np.column_stack((xs, ys, zs))
    coords = np.array(coords, order='C')
    coords.shape = (1, -1)
    return coords[0]


def set_disk_positions():
    # TODO: restrict the maximum radius and height
    radii = np.zeros(N_disk)
    sample = nprand.sample(N_disk)
    for i, s in enumerate(sample):
        radii[i] = disk_radial_inverse_cumulative(s)
    
    zs = disk_height_inverse_cumulative(nprand.sample(N_disk))
    phis = 2 * np.pi * nprand.sample(N_disk)

    xs = radii * np.cos(phis)
    ys = radii * np.sin(phis)

    coords = np.column_stack((xs, ys, zs))
    coords = np.array(coords, order='C')
    coords.shape = (1, -1)
    return coords[0]


def disk_radial_cumulative(r):
    return (Rd**2-(Rd**2+r*Rd)*np.exp(-r/Rd))/Rd**2


# frac is a number between 0 and 1
def disk_radial_inverse_cumulative(frac):
    return brentq(lambda r: disk_radial_cumulative(r) - frac, 0, 1.0e10)


def disk_height_inverse_cumulative(frac):
    return 0.5 * z0 * np.log(frac/(1-frac))


def set_velocities(coords):
    return np.zeros(N_total)


def write_input_file(galaxy_data):
    coords = galaxy_data[0]
    vels = galaxy_data[1]
    masses = np.empty(N_total)
    masses.fill(M_total / N_total)
    ids = np.arange(1, N_total + 1, 1)
    write_snapshot(n_part=[0, N_halo, N_disk, N_bulge, 0, 0], from_text=False,
                   data_list=[coords, vels, ids, masses])


if __name__ == '__main__':
    main()
