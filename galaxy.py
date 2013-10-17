import numpy as np

from snapwrite import process_input, write_snapshot


N_disk = 3000
N_bulge = 1000
N_halo = 6000
N_total = N_disk + N_bulge + N_halo

M_disk = 10000
M_bulge = 10000
M_halo = 10000
M_total = M_disk + M_bulge + M_halo

halo_core = False
bulge_core = False

G = 43007.1


def main():
    galaxy_data = generate_galaxy()
    write_input_file(galaxy_data)


def generate_galaxy():
    coords_halo = set_halo_positions()
    coords_bulge = set_bulge_positions
    coords_disk = set_disk_positions()
    coords = np.concatenate((coords_disk, coords_bulge, coords_halo))
    vels = set_velocities(coords)
    return [coords, vels]

def dehnen_inverse_cumulative(Mc, M, a, core):
    if(core):
        return ((a * (Mc**(2/3.)*M**(4/3.) + Mc*M + Mc**(4/3.)*M**(2/3.))) /
                   (Mc**(1/3.) * M**(2/3.) * (M-Mc)))
    else:
        return (a * ((Mc*M)**0.5 + Mc)) / (M-Mc)

def disk_inverse_cumulative(radial=False):

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



def set_velocities(coords):
    potential_tabulated = []
    # yada gerar potential tabulated...

def write_input_file(galaxy_data):
    coords = galaxy_data[0]
    vels = galaxy_data[1]
    masses = np.empty(N_total)
    masses.fill(M_total / N_total)
    ids = np.arange(1, N_total + 1, 1)
    write_snapshot(n_part=[0, N_total, 0, 0, 0, 0], from_text=False,
                   data_list=[coords, vels, ids, masses])

