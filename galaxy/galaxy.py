# Use: python galaxy.py [FILE]

from os import path
from sys import exit

import numpy as np
import numpy.random as nprand
from numpy import cos, sin, pi, arccos, log10, exp, arctan, cosh
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from bisect import bisect_left

from optimized_functions import phi_disk
from snapwrite import process_input, write_snapshot


halo_core = False
bulge_core = False
G = 43007.1


def main():
    init()
    galaxy_data = generate_galaxy()
    write_input_file(galaxy_data)


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


def generate_galaxy():
    coords_halo = set_halo_positions()
    coords_disk = set_disk_positions()
    coords_bulge = set_bulge_positions()
    coords = np.concatenate((coords_halo, coords_disk, coords_bulge))
    vels = set_velocities(coords)
    coords = np.array(coords, order='C')
    coords.shape = (1, -1) # Linearizing the array.
    vels = np.array(vels, order='C')
    vels.shape = (1, -1)
    return [coords[0], vels[0]]


def dehnen_inverse_cumulative(Mc, M, a, core):
    if(core):
        return ((a * (Mc**(2/3.)*M**(4/3.) + Mc*M + Mc**(4/3.)*M**(2/3.))) /
                   (Mc**(1/3.) * M**(2/3.) * (M-Mc)))
    else:
        return (a * ((Mc*M)**0.5 + Mc)) / (M-Mc)


def dehnen_potential(r, M, a, core):
    if(core):
        return (G*M)/(2*a) * ((r/(r+a))**2 - 1)
    else:
        return (G*M)/a * (r/(r+a) - 1)


def halo_density(r):
    if(halo_core):
        return (3*M_halo)/(4*pi) * a_halo/(r+a_halo)**4
    else:
        return M_halo/(2*pi) * a_halo/(r*(r+a_halo)**3)


def disk_density(rho, z):
    cte = M_disk/(4*pi*z0*Rd**2)
    return cte * (1/cosh(z/z0))**2 * exp(-rho/Rd)
 

def bulge_density(r):
    if(bulge_core):
        return (3*M_bulge)/(4*pi) * a_bulge/(r+a_bulge)**4
    else:
        return M_bulge/(2*pi) * a_bulge/(r*(r+a_bulge)**3)


def set_halo_positions():
    # The factor M * 200^2 / 201^2 restricts the radius to 200 * a.
    radii = dehnen_inverse_cumulative(nprand.sample(N_halo) *
        ((M_halo*40000) / 40401), M_halo, a_halo, halo_core)
    thetas = np.arccos(nprand.sample(N_halo)*2 - 1)
    phis = 2 * pi * nprand.sample(N_halo)
    xs = radii * sin(thetas) * cos(phis)
    ys = radii * sin(thetas) * sin(phis)
    zs = radii * cos(thetas)

    # Older NumPy versions freak out without this line.
    coords = np.column_stack((xs, ys, zs))
    return coords


def set_bulge_positions():
    radii = dehnen_inverse_cumulative(nprand.sample(N_bulge) *
        ((M_bulge*40000) / 40401), M_bulge, a_bulge, bulge_core)
    thetas = np.arccos(nprand.sample(N_bulge)*2 - 1)
    phis = 2 * pi * nprand.sample(N_bulge)
    xs = radii * sin(thetas) * cos(phis)
    ys = radii * sin(thetas) * sin(phis)
    zs = radii * cos(thetas)
    coords = np.column_stack((xs, ys, zs))
    return coords


def set_disk_positions():
    # TODO: restrict the maximum radius and height
    radii = np.zeros(N_disk)
    sample = nprand.sample(N_disk)
    for i, s in enumerate(sample):
        radii[i] = disk_radial_inverse_cumulative(s)
    zs = disk_height_inverse_cumulative(nprand.sample(N_disk))
    phis = 2 * pi * nprand.sample(N_disk)
    xs = radii * cos(phis)
    ys = radii * sin(phis)
    coords = np.column_stack((xs, ys, zs))
    return coords


def disk_radial_cumulative(r):
    return (Rd**2-(Rd**2+r*Rd)*exp(-r/Rd))/Rd**2


# 'frac' is a number between 0 and 1.
def disk_radial_inverse_cumulative(frac):
    return brentq(lambda r: disk_radial_cumulative(r) - frac, 0, 1.0e10)


def disk_height_inverse_cumulative(frac):
    return 0.5 * z0 * np.log(frac/(1-frac))


def interpolate(value, axis):
    index = bisect_left(axis, value)
    if(index >= len(axis)-1):
        return len(axis)-1
    else:
        return index


def set_velocities(coords):
    N_rho = Nz = 100
    rho_max = 200 * a_halo
    z_max = 2000 * a_halo # This has to go far so I can estimate the integral.
    rho_axis = np.logspace(log10(0.1), log10(rho_max), N_rho)
    z_axis = np.logspace(log10(0.1), log10(z_max), Nz)
    phi_grid = np.zeros((N_rho, Nz))

    # Filling the potential grid
    for i in range(N_rho):
        print "Potential calculation, %d of %d..." % (i, N_rho)
        for j in range(Nz):
            r = (rho_axis[i]**2 + z_axis[j]**2)**0.5
            phi_grid[i][j] += dehnen_potential(r, M_halo, a_halo, halo_core)
            phi_grid[i][j] += phi_disk(rho_axis[i], z_axis[j], M_disk, Rd, z0)
            phi_grid[i][j] += dehnen_potential(r, M_bulge, a_bulge, bulge_core)

    # The [0], [1] and [2] components of this grid will refer to the halo,
    # disk and bulge, respectively. The calculation being performed here
    # follows the prescription found in Springel & White, 1999.
    sz_grid = np.zeros((3, N_rho, Nz))
    ys = np.zeros((3, N_rho, Nz)) # Integrand array.
    for i in range(N_rho):
        for j in range(1, Nz):
            r = (rho_axis[i]**2 + z_axis[j]**2)**0.5
            dz = z_axis[j] - z_axis[j-1]
            dphi = phi_grid[i][j] - phi_grid[i][j-1]

            # Filling the integrand array.
            ys[0][i][j] = halo_density(r) * dphi/dz 
            ys[1][i][j] = disk_density(rho_axis[i], z_axis[j]) * dphi/dz
            ys[2][i][j] = bulge_density(r) * dphi/dz 
        ys[0][i][0] = ys[0][i][1]
        ys[1][i][0] = ys[1][i][1]
        ys[2][i][0] = ys[2][i][1]
        for j in range(0, Nz-1):
            r = (rho_axis[i]**2 + z_axis[j]**2)**0.5
            sz_grid[0][i][j] = 1/halo_density(r) * np.trapz(ys[0][i][j:], z_axis[j:])
            sz_grid[1][i][j] = 1/disk_density(rho_axis[i], z_axis[j]) * np.trapz(ys[1][i][j:], z_axis[j:])
            sz_grid[2][i][j] = 1/bulge_density(r) * np.trapz(ys[2][i][j:], z_axis[j:])
        sz_grid[0][i][Nz-1] = sz_grid[0][i][Nz-2]
        sz_grid[1][i][Nz-1] = sz_grid[1][i][Nz-2]
        sz_grid[2][i][Nz-1] = sz_grid[2][i][Nz-2]

    sphi_grid = np.zeros((3, N_rho, Nz))
    for i in range(1, N_rho-1):
        for j in range(Nz):
            r0 = (rho_axis[i]**2 + z_axis[j]**2)**0.5
            r1 = (rho_axis[i+1]**2 + z_axis[j]**2)**0.5
            drho = rho_axis[i+1] - rho_axis[i]
            dphi = phi_grid[i+1][j] - phi_grid[i][j]
            d2phi = phi_grid[i+1][j] - 2*phi_grid[i][j] + phi_grid[i-1][j]
            kappa2 = 3/rho_axis[i] * dphi/drho + d2phi/drho**2
            gamma2 = 4/(kappa2*rho_axis[i]) * dphi/drho
            sphi_grid[0][i][j] = (sz_grid[0][i][j] + rho_axis[i]/halo_density(r0) *
                (halo_density(r1)*sz_grid[0][i+1][j] - 
                halo_density(r0)*sz_grid[0][i][j]) / drho +
                rho_axis[i] * dphi/drho)
            sphi_grid[1][i][j] = sz_grid[1][i][j] / gamma2
            sphi_grid[2][i][j] = (sz_grid[2][i][j] + rho_axis[i]/bulge_density(r0) *
                (bulge_density(r1)*sz_grid[2][i+1][j] - 
                bulge_density(r0)*sz_grid[2][i][j]) / drho +
                rho_axis[i] * dphi/drho)
            for k in range(3):
                sphi_grid[k][0][j] = sphi_grid[k][1][j]
                sphi_grid[k][N_rho-1][j] = sphi_grid[k][N_rho-3][j]

    # Dictionary to hold interpolator functions for the circular velocity
    # of the disk, one function per value of z. They are created on the run,
    # to avoid creating functions for values of z which are not used.
    vphis = {}
    vels = np.zeros((N_total, 3))
    for i, part in enumerate(coords):
        x = part[0]
        y = part[1]
        z = abs(part[2])
        rho = (x**2 + y**2)**0.5
        if(x > 0 and y > 0):
            phi = arctan(y/x)
        elif(x < 0 and y > 0):
            phi = pi - arctan(-y/x)
        elif(x < 0 and y < 0):
            phi = pi + arctan(y/x)
        elif(x > 0 and y < 0):
            phi = 2 * pi - arctan(-y/x)
        bestz = interpolate(z, z_axis)
        bestr = interpolate(rho, rho_axis)
        if(i < N_halo):
            sigmaz = sz_grid[0][bestr][bestz]
            sigmap = sphi_grid[0][bestr][bestz]
            vz = nprand.normal(scale=sigmaz**0.5)
            vr = nprand.normal(scale=sigmaz**0.5)
            vphi = nprand.normal(scale=sigmap**0.5)
        elif(i >= N_halo and i < N_halo+N_disk):
            if(bestz == 0):
                bestz += 1
            if(bestr == 0):
                bestr += 1
            sigmaz = sz_grid[1][bestr][bestz]
            sigmap = sphi_grid[1][bestr][bestz]
            vz = nprand.normal(scale=sigmaz**0.5)
            vr = nprand.normal(scale=sigmaz**0.5)
            vphi = nprand.normal(scale=sigmap**0.5)
            dphi = phi_grid[bestr][bestz] - phi_grid[bestr-1][bestz]
            drho = rho_axis[bestr] - rho_axis[bestr-1]
            vphi += (rho_axis[bestr]*dphi/drho)**0.5
        else:
            sigmaz = sz_grid[2][bestr][bestz]
            sigmap = sphi_grid[2][bestr][bestz]
            vz = nprand.normal(scale=sigmaz**0.5)
            vr = nprand.normal(scale=sigmaz**0.5)
            vphi = nprand.normal(scale=sigmap**0.5)
        vels[i][0] = vr*cos(phi) - vphi*sin(phi)
        vels[i][1] = vr*sin(phi) + vphi*cos(phi)
        vels[i][2] = vz
    return vels
 

def write_input_file(galaxy_data):
    coords = galaxy_data[0]
    vels = galaxy_data[1]
    m_halo = np.empty(N_halo)
    m_halo.fill(M_halo/N_halo)
    m_disk = np.empty(N_disk)
    m_disk.fill(M_disk/N_disk)
    m_bulge = np.empty(N_bulge)
    m_bulge.fill(M_bulge/N_bulge)
    masses = np.concatenate((m_halo, m_disk, m_bulge))
    ids = np.arange(1, N_total+1, 1)
    write_snapshot(n_part=[0, N_halo, N_disk, N_bulge, 0, 0], from_text=False,
                    data_list=[coords, vels, ids, masses])


if __name__ == '__main__':
    main()
