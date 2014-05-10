# Use: python galaxy.py [FILE]

from os import path, remove
from sys import exit
from sys import path as syspath

import numpy as np
import numpy.random as nprand
from numpy import cos, sin, pi, arccos, log10, exp, arctan, cosh
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from bisect import bisect_left
from multiprocessing import Process, Array
from argparse import ArgumentParser as parser

from optimized_functions import phi_disk
from snapwrite import process_input, write_snapshot
syspath.append(path.join(path.dirname(__file__), '..', 'misc'))
from units import temp_to_internal_energy, internal_energy_to_temp


G = 43007.1


def main():
    init()
    galaxy_data = generate_galaxy()
    write_input_file(galaxy_data)


def init():
    global M_halo, M_disk, M_bulge, M_gas
    global N_halo, N_disk, N_bulge, N_gas
    global a_halo, a_bulge, Rd, z0
    global N_total, M_total
    global phi_grid, rho_axis, z_axis, rho_max, z_max, N_rho, Nz
    global halo_core, bulge_core, N_CORES, force_yes, output
    flags = parser(description="Generates an initial conditions file\
                                for a galaxy simulation with halo, stellar\
                                disk, gaseous disk and bulge components.")
    flags.add_argument('--halo-core', help='Sets the density profile for the\
                       halo to have a core.', action='store_true')
    flags.add_argument('--bulge-core', help='The same, but for the bulge.',
                       action='store_true')
    flags.add_argument('-cores', help='The number of cores to use during the\
                       potential canculation. Default is 1. Make sure this\
                       number is a factor of N_rho and N_z.', default=1)
    flags.add_argument('--force-yes', help='Don\'t ask if you want to use the\
                        existing potential_data.txt file. Might be useful to\
                        run the script from another script.',
                        action='store_true')
    flags.add_argument('-o', help='The name of the output file.',
                       metavar="init.dat", default="init.dat")
    args = flags.parse_args()
    halo_core = args.halo_core
    bulge_core = args.bulge_core
    N_CORES = int(args.cores)
    force_yes = args.force_yes
    output = args.o

    if not (path.isfile("header.txt") and path.isfile("galaxy_param.txt")):
        print "header.txt or galaxy_param.txt missing."
        exit(0)

    vars_ = process_input("galaxy_param.txt")
    M_halo, M_disk, M_bulge, M_gas = (float(i[0]) for i in vars_[0:4])
    N_halo, N_disk, N_bulge, N_gas = (float(i[0]) for i in vars_[4:8])
    a_halo, a_bulge, Rd, z0 = (float(i[0]) for i in vars_[8:12])
    M_total = M_disk + M_bulge + M_halo + M_gas
    N_total = N_disk + N_bulge + N_halo + N_gas
    N_rho = Nz = 280 # Make sure N_CORES is a factor of this number!
    phi_grid = np.zeros((N_rho, Nz))
    rho_max = 200 * a_halo
    # This has to go far so I can estimate the integrals below.
    z_max = 2000 * a_halo 
    rho_axis = np.logspace(-2, log10(rho_max), N_rho)
    z_axis = np.logspace(-2, log10(z_max), Nz)


def generate_galaxy():
    global phi_grid
    print "Setting positions..."
    coords_halo = set_halo_positions()
    coords_disk = set_disk_positions(N_disk, z0)
    coords_gas = set_disk_positions(N_gas, z0/7)
    coords_bulge = set_bulge_positions()
    coords = np.concatenate((coords_gas, coords_halo, coords_disk,
                             coords_bulge))

    if path.isfile('potential_data.txt'):
        if not force_yes:
            print ("Use existing potential tabulation in potential_data.txt? "+
                   "Make sure it\nrefers to the current parameters. (y/n)")
            ans = raw_input()
            while ans not in "yn":
                print "Please give a proper answer. (y/n)"
                ans = raw_input()
        else: ans = "y"
        if ans == "y":
            phi_grid = np.loadtxt('potential_data.txt')
        else:
            remove('potential_data.txt')
            fill_potential_grid()
            np.savetxt('potential_data.txt', phi_grid)
    else:
        fill_potential_grid()
        np.savetxt('potential_data.txt', phi_grid)
    print "Setting temperatures..."
    U, T_cl_grid = set_temperatures(coords_gas) 
    print "Setting velocities..."
    vels = set_velocities(coords, T_cl_grid) 
    coords = np.array(coords, order='C')
    coords.shape = (1, -1) # Linearizing the array.
    vels = np.array(vels, order='C')
    vels.shape = (1, -1)
    rho = set_densities(coords_gas)
    return [coords[0], vels[0], U, rho]


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


def disk_density(rho, z, M, z0):
    cte = M/(4*pi*z0*Rd**2)
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


def set_disk_positions(N, z0):
    radii = np.zeros(N)
    # The maximum radius is restricted to 21 kpc
    sample = nprand.sample(N) * disk_radial_cumulative(21)
    for i, s in enumerate(sample):
        radii[i] = disk_radial_inverse_cumulative(s)
    zs = disk_height_inverse_cumulative(nprand.sample(N), z0)
    phis = 2 * pi * nprand.sample(N)
    xs = radii * cos(phis)
    ys = radii * sin(phis)
    coords = np.column_stack((xs, ys, zs))
    return coords


def disk_radial_cumulative(r):
    return (Rd**2-(Rd**2+r*Rd)*exp(-r/Rd))/Rd**2


# 'frac' is a number between 0 and 1.
def disk_radial_inverse_cumulative(frac):
    return brentq(lambda r: disk_radial_cumulative(r) - frac, 0, 1.0e10)


def disk_height_inverse_cumulative(frac, z0):
    return 0.5 * z0 * np.log(frac/(1-frac))


def interpolate(value, axis):
    index = bisect_left(axis, value)
    if(index >= len(axis)-1):
        return len(axis)-1
    else:
        return index


def fill_potential_grid():
    ps = []
    print ("Filling potential grid. This takes a while, even after thorough\n"
           "optimization.")
    def loop(n_loop, N_CORES):
        for i in range(n_loop*N_rho/N_CORES, (1+n_loop)*N_rho/N_CORES):
            print "%1.1f%% done at core %d" % (float(i-n_loop*N_rho/N_CORES) /
                (N_rho/N_CORES) / 0.01, n_loop + 1)
            for j in range(Nz):
                r = (rho_axis[i]**2 + z_axis[j]**2)**0.5
                shared_phi_grid[i][j] += dehnen_potential(r, M_halo, a_halo,
                    halo_core)
                shared_phi_grid[i][j] += phi_disk(rho_axis[i], z_axis[j],
                    M_disk, Rd, z0)
                shared_phi_grid[i][j] += phi_disk(rho_axis[i], z_axis[j],
                    M_gas, Rd, z0/7)
                shared_phi_grid[i][j] += dehnen_potential(r, M_bulge, a_bulge,
                    bulge_core)
    shared_phi_grid = [Array('f', phi_grid[i]) for i in range(len(phi_grid))]
    proc=[Process(target=loop, args=(n, N_CORES)) for n in range(N_CORES)]
    try:
        [p.start() for p in proc]
        [p.join() for p in proc]
    except KeyboardInterrupt:
        [p.terminate() for p in proc]
        [p.join() for p in proc]
        print "\nProcess canceled."
        exit(0)
    for i in range(N_rho):
        for j in range(Nz):
            phi_grid[i][j] = shared_phi_grid[i][j]


def set_velocities(coords, T_cl_grid):
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
            ys[1][i][j] = disk_density(rho_axis[i], z_axis[j], M_disk, z0) * dphi/dz
            ys[2][i][j] = bulge_density(r) * dphi/dz 
        ys[0][i][0] = ys[0][i][1]
        ys[1][i][0] = ys[1][i][1]
        ys[2][i][0] = ys[2][i][1]
        for j in range(0, Nz-1):
            r = (rho_axis[i]**2 + z_axis[j]**2)**0.5
            sz_grid[0][i][j] = 1/halo_density(r) * np.trapz(ys[0][i][j:], z_axis[j:])
            sz_grid[1][i][j] = 1/disk_density(rho_axis[i], z_axis[j], M_disk, z0) * np.trapz(ys[1][i][j:], z_axis[j:])
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
        if rho < rho_axis[0]:
            rho = rho_axis[0]
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
        if(i < N_gas):
            if(bestz not in vphis):
                ds = np.zeros(N_rho)
                for j in range(1, N_rho):
                    dphi = phi_grid[j][bestz]-phi_grid[j-1][bestz]
                    drho = rho_axis[j]-rho_axis[j-1]
                    ds[j] = dphi/drho
                ds[0] = ds[1]
                vphis[bestz] = interp1d(rho_axis, ds, kind='cubic')
            dP = (disk_density(rho_axis[bestr], z, M_gas, z0/7)*
                  T_cl_grid[bestr][bestz] - disk_density(rho_axis[bestr-1], 
                  z, M_gas, z0/7)*T_cl_grid[bestr-1][bestz])
            drho = rho_axis[bestr] - rho_axis[bestr-1]
            vphi2 = rho * (vphis[bestz](rho) + 1/disk_density(rho, z, M_gas,
                    z0/7) * dP/drho)
            vphi = abs(vphi2)**0.5
            vz = vr = 0
        elif(i >= N_gas and i < N_gas+N_halo):
            sigmaz = sz_grid[0][bestr][bestz]
            sigmap = sphi_grid[0][bestr][bestz]
            vz = nprand.normal(scale=sigmaz**0.5)
            vr = nprand.normal(scale=sigmaz**0.5)
            vphi = nprand.normal(scale=sigmap**0.5)
        elif(i >= N_gas+N_halo and i < N_gas+N_halo+N_disk):
            if(bestz == 0):
                bestz += 1
            if(bestr == 0):
                bestr += 1
            sigmaz = sz_grid[1][bestr][bestz]
            sigmap = sphi_grid[1][bestr][bestz]
            vz = nprand.normal(scale=sigmaz**0.5)
            vr = nprand.normal(scale=sigmaz**0.5)
            vphi = nprand.normal(scale=sigmap**0.5)
            if(bestz not in vphis):
                ds = np.zeros(N_rho)
                for j in range(1, N_rho):
                    dphi = phi_grid[j][bestz]-phi_grid[j-1][bestz]
                    drho = rho_axis[j]-rho_axis[j-1]
                    ds[j] = dphi/drho
                ds[0] = ds[1]
                vphis[bestz] = interp1d(rho_axis, ds, kind='cubic')
            vphi += (rho * vphis[bestz](rho))**0.5
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
 

def set_densities(coords_gas):
    rhos = np.zeros(N_gas)
    for i, part in enumerate(coords_gas):
        rho = (part[0]**2 + part[1]**2)**0.5
        z = abs(part[2])
        rhos[i] = disk_density(rho, z, M_gas, z0/7)
    return rhos


def set_temperatures(coords_gas):
    T_grid = np.zeros((N_rho, Nz))
    U = np.zeros(N_gas)
    # Constantless temperature, will be used in the circular
    # velocity determination for the gas.
    T_cl_grid = np.zeros((N_rho, Nz)) 
    MP_OVER_KB = 121.148
    HYDROGEN_MASSFRAC = 0.76
    meanweight_n = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC)
    meanweight_i = 4.0 / (3 + 5 * HYDROGEN_MASSFRAC)
    ys = np.zeros((N_rho, Nz)) # Integrand array.
    for i in range(N_rho):
        for j in range(1, Nz):
            dphi = phi_grid[i][j] - phi_grid[i][j-1]
            dz = z_axis[j] - z_axis[j-1]
            ys[i][j] = (disk_density(rho_axis[i], z_axis[j], M_gas, z0/7) *
                        dphi/dz)
        ys[i][0] = ys[i][1]
        for j in range(0, Nz-1):
            result = (np.trapz(ys[i][j:], z_axis[j:]) /
                      disk_density(rho_axis[i], z_axis[j], M_gas, z0/7))
            temp_i = MP_OVER_KB * meanweight_i * result
            temp_n = MP_OVER_KB * meanweight_n * result
            if(temp_i > 1.0e4):
                T_grid[i][j] = temp_to_internal_energy(temp_i)
            else:
                T_grid[i][j] = temp_to_internal_energy(temp_n)
            T_cl_grid[i][j] = result
        T_grid[i][-1] = T_grid[i][-2]
        T_cl_grid[i][-1] = T_cl_grid[i][-2]
    for i, part in enumerate(coords_gas):
        rho = (part[0]**2 + part[1]**2)**0.5
        z = abs(part[2])
        bestz = interpolate(z, z_axis)
        bestr = interpolate(rho, rho_axis)
        U[i] = T_grid[bestr][bestz]
    #U.fill(temp_to_internal_energy(1000))
    #T_cl_grid.fill(1000 / MP_OVER_KB / meanweight_n)
    return U, T_cl_grid


def write_input_file(galaxy_data):
    coords = galaxy_data[0]
    vels = galaxy_data[1]
    U = galaxy_data[2]
    rho = galaxy_data[3]
    m_gas = np.empty(N_gas)
    m_gas.fill(M_gas/N_gas)
    m_halo = np.empty(N_halo)
    m_halo.fill(M_halo/N_halo)
    m_disk = np.empty(N_disk)
    m_disk.fill(M_disk/N_disk)
    m_bulge = np.empty(N_bulge)
    m_bulge.fill(M_bulge/N_bulge)
    masses = np.concatenate((m_gas, m_halo, m_disk, m_bulge))
    ids = np.arange(1, N_total+1, 1)
    smooths = np.zeros(N_gas)
    write_snapshot(n_part=[N_gas, N_halo, N_disk, N_bulge, 0, 0],
                   from_text=False, outfile=output
                   data_list=[coords, vels, ids, masses, U, rho, smooths])


if __name__ == '__main__':
    main()
