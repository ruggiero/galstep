# Run python galstep.py --help for a description.

from os import path
from sys import exit, stdout, path
from time import sleep

import numpy as np
import numpy.random as nprand
from numpy import cos, sin, pi, log10, exp, arctan2, cosh
from scipy.optimize import brentq
from scipy import integrate
from bisect import bisect_left
from multiprocessing import Process, Array
from argparse import ArgumentParser as parser
import configparser
from itertools import product
import scipy.interpolate as inter

from treecode import oct_tree, potential
from snapwrite import write_snapshot
import units

import warnings
warnings.filterwarnings("ignore")

def main():
  init()
  galaxy_data = generate_galaxy()
  write_input_file(galaxy_data)


def init():
  global M_halo, M_disk, M_bulge, M_gas
  global N_halo, N_disk, N_bulge, N_gas
  global a_halo, a_bulge, gamma_halo, gamma_bulge, Rd, z0, z0_gas
  global halo_cut_r, halo_cut_M, bulge_cut_r, bulge_cut_M
  global disk_cut_r, disk_cut
  global N_total, M_total
  global phi_grid, rho_axis, z_axis, N_rho, Nz
  global N_CORES, output, input_, gas, bulge, factor, Z
  global file_format

  flags = parser(description="Generates an initial conditions file for a\
                              galaxy simulation with halo, stellar disk,\
                              gaseous disk and bulge components.")
  flags.add_argument('-cores', help='The number of cores to use during the\
                                     potential canculation. Make sure this\
                                     number is a factor of N_rho*N_z. Default\
                                     is 1.', default=1)
  flags.add_argument('--hdf5', help='Output initial conditions in HDF5\
                                     format.', action = 'store_true')
  flags.add_argument('-o', help='The name of the output file.',
                     metavar="init.dat", default="init.dat")
  flags.add_argument('-i', help='The name of the .ini file.',
                     metavar="params_galaxy.ini", default="params_galaxy.ini")
  args = flags.parse_args()
  N_CORES = int(args.cores)
  output = args.o
  input_ = args.i
  
  if args.hdf5:
    file_format = 'hdf5'
  else:
    file_format = 'gadget2'

  if not path.isfile(input_):
    print("Input file not found:", input_)
    exit(0)

  config = configparser.ConfigParser(inline_comment_prefixes=';')
  config.read(input_)
  # Halo
  M_halo = config.getfloat('halo', 'M_halo')
  a_halo = config.getfloat('halo', 'a_halo')
  N_halo = config.getint('halo', 'N_halo')
  gamma_halo = config.getfloat('halo', 'gamma_halo')
  halo_cut_r = config.getfloat('halo', 'halo_cut_r')
  # Disk
  M_disk = config.getfloat('disk', 'M_disk')
  N_disk = config.getint('disk', 'N_disk')
  Rd = config.getfloat('disk', 'Rd')
  z0 = config.getfloat('disk', 'z0')
  factor = config.getfloat('disk', 'factor')
  disk_cut_r = config.getfloat('disk', 'disk_cut_r')
  # Bulge
  bulge = config.getboolean('bulge', 'include')
  M_bulge = config.getfloat('bulge', 'M_bulge')
  a_bulge = config.getfloat('bulge', 'a_bulge')
  N_bulge = config.getint('bulge', 'N_bulge')
  gamma_bulge = config.getfloat('bulge', 'gamma_bulge')
  bulge_cut_r = config.getfloat('bulge', 'bulge_cut_r')
  # Gas
  gas = config.getboolean('gas', 'include')
  M_gas = config.getfloat('gas', 'M_gas')
  N_gas = config.getint('gas', 'N_gas')
  z0_gas = config.getfloat('gas', 'z0_gas')
  Z = config.getfloat('gas', 'Z')

  z0_gas *= z0
  if not gas:
    N_gas = 0
  if not bulge:
    N_bulge = 0
  M_total = M_disk + M_bulge + M_halo + M_gas
  N_total = N_disk + N_bulge + N_halo + N_gas
  N_rho = config.getint('global', 'N_rho')
  Nz = config.getint('global', 'Nz')
  phi_grid = np.zeros((N_rho, Nz))
  rho_max = config.getfloat('global', 'rho_max')*a_halo
  z_max = config.getfloat('global', 'z_max')*a_halo
  rho_axis = np.logspace(-2, log10(rho_max), N_rho)
  z_axis = np.logspace(-2, log10(z_max), Nz)


def generate_galaxy():
  global phi_grid
  print("Setting positions...")
  coords_halo = set_halo_positions()
  coords_stars = set_disk_positions(N_disk, z0)
  if(bulge):
    coords_bulge = set_bulge_positions()
  if(gas):
    coords_gas = set_disk_positions(N_gas, z0_gas)
    if(bulge):
      coords = np.concatenate((coords_gas, coords_halo, coords_stars,
                               coords_bulge))
    else:
      coords = np.concatenate((coords_gas, coords_halo, coords_stars))
  else:
    if(bulge):
      coords = np.concatenate((coords_halo, coords_stars, coords_bulge))
    else:
      coords = np.concatenate((coords_halo, coords_stars))
  if(gas):
    fill_potential_grid(coords_stars, coords_gas)
  else:
    fill_potential_grid(coords_stars) 
  if(gas):
    print("Setting temperatures...")
    U, T_cl_grid = set_temperatures(coords_gas) 
    print("Setting densitites...")
    rho = np.zeros(N_gas)
    print("Setting velocities...")
    vels = set_velocities(coords, T_cl_grid) 
  else:
    print("Setting velocities...")
    vels = set_velocities(coords, None)
  coords = np.array(coords, order='C')
  # Don't need this for HDF5 - D. Rennehan
  if file_format == 'gadget2':
    coords.shape = (1, -1) # Linearizing the array.
  vels = np.array(vels, order='C')
  if file_format == 'gadget2':
    vels.shape = (1, -1)

  # Again, don't need this if doing HDF5 - D. Rennehan
  if file_format == 'hdf5':
    if gas:
      return [coords, vels, U, rho]
    else:
      return [coords, vels]
  else:
    if(gas):
      return [coords[0], vels[0], U, rho]
    else:
      return [coords[0], vels[0]]


def dehnen_cumulative(r, M, a, gamma):
  return M * (r/(r+float(a)))**(3-gamma)


# Inverse cumulative mass function. Mc is a number between 0 and M.
def dehnen_inverse_cumulative(Mc, M, a, gamma):
  results = []
  for i in Mc:
    results.append(brentq(lambda r: dehnen_cumulative(r, M, a, gamma) - i, 0, 1.0e10))
  return np.array(results)


def dehnen_potential(r, M, a, gamma):
  if gamma != 2:
    return (units.G*M)/a * (-1.0/(2-gamma)) * (1-(r/(r+float(a)))**(2-gamma))
  else:
    return (units.G*M)/a * np.log(r/(r+float(a)))


def dehnen_density(r, M, a, gamma):
  return ((3-gamma)*M)/(4*np.pi) * a/(r**gamma * (r+float(a))**(4-gamma))


def disk_density(rho, z, M, z0):
  cte = M/(4*pi*z0*Rd**2)
  return cte * (1/cosh(z/z0))**2 * exp(-rho/Rd)
 

def set_halo_positions():
  global halo_cut_M
  halo_cut_M = dehnen_cumulative(halo_cut_r, M_halo, a_halo, gamma_halo)
  print( "%.0f%% of halo mass cut by the truncation..." % \
        (100*(1-halo_cut_M/M_halo)))
  if halo_cut_M < 0.9*M_halo:
    print( "  \_ Warning: this is more than 10%% of the total halo mass!")
  radii = dehnen_inverse_cumulative(nprand.sample(N_halo) * halo_cut_M,
    M_halo, a_halo, gamma_halo)
  thetas = np.arccos(nprand.sample(N_halo)*2 - 1)
  phis = 2 * pi * nprand.sample(N_halo)
  xs = radii * sin(thetas) * cos(phis)
  ys = radii * sin(thetas) * sin(phis)
  zs = radii * cos(thetas)
  coords = np.column_stack((xs, ys, zs))
  return coords


def set_bulge_positions():
  global bulge_cut_M
  bulge_cut_M = dehnen_cumulative(bulge_cut_r, M_bulge, a_bulge, gamma_bulge)
  print( "%.0f%% of bulge mass cut by the truncation..." % \
        (100*(1-bulge_cut_M/M_bulge)))
  if bulge_cut_M < 0.9*M_bulge:
    print ("  \_ Warning: this is more than 10%% of the total bulge mass!")
  radii = dehnen_inverse_cumulative(nprand.sample(N_bulge) * bulge_cut_M,
    M_bulge, a_bulge, gamma_bulge)
  thetas = np.arccos(nprand.sample(N_bulge)*2 - 1)
  phis = 2 * pi * nprand.sample(N_bulge)
  xs = radii * sin(thetas) * cos(phis)
  ys = radii * sin(thetas) * sin(phis)
  zs = radii * cos(thetas)
  coords = np.column_stack((xs, ys, zs))
  return coords


def set_disk_positions(N, z0):
  global disk_cut
  radii = np.zeros(N)
  disk_cut = disk_radial_cumulative(disk_cut_r)
  print( "%.0f%% of disk mass cut by the truncation..." % \
        (100*(1-disk_cut)))
  if disk_cut < 0.9:
    print( "  \_ Warning: this is more than 10% of the total disk mass!")
  sample = nprand.sample(N) * disk_cut
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


def fill_potential_grid(coords_stars, coords_gas=None):
  ps = []
  # Indexes are randomly distributed across processors for higher
  # performance. The tree takes longer to calculate the potential
  # at small radii. ip stands for 'index pair'.
  ip = nprand.permutation(list(product(range(N_rho), range(Nz))))
  print("Building gravity tree...")
  gravtree = oct_tree(200*a_halo*2)
  for i, part in enumerate(coords_stars):
    prog = 100*float(i)/len(coords_stars)
    stdout.write("%.2f%% done for the stellar disk\r" % prog)
    stdout.flush()
    gravtree.insert(part, M_disk/N_disk)
  if(coords_gas is not None):
    for i, part in enumerate(coords_gas):
      prog = 100*float(i)/len(coords_gas)
      stdout.write("%.2f%% done for the gaseous disk\r" % prog)
      stdout.flush()
      gravtree.insert(part, M_gas/N_gas)
 
  print ("Filling potential grid...")
  def loop(n_loop, N_CORES):
      
    for i in range(n_loop*N_rho*Nz//N_CORES, (1+n_loop)*N_rho*Nz//N_CORES):
      prog[n_loop] = 100*float(i-n_loop*N_rho*Nz//N_CORES)//(N_rho*Nz//N_CORES)
      m = ip[i][0]
      n = ip[i][1]
      r = (rho_axis[m]**2 + z_axis[n]**2)**0.5
      shared_phi_grid[m][n] += dehnen_potential(r, M_halo, a_halo, gamma_halo)
      shared_phi_grid[m][n] += potential(np.array((rho_axis[m], 0, z_axis[n])),
        gravtree)
      if(bulge):
        shared_phi_grid[m][n] += dehnen_potential(r, M_bulge, a_bulge, gamma_bulge)
  shared_phi_grid = [Array('f', phi_grid[i]) for i in range(len(phi_grid))]
  prog = Array('f', [0]*N_CORES)
  proc=[Process(target=loop, args=(n, N_CORES)) for n in range(N_CORES)]
  try:
    [p.start() for p in proc]
    while np.all([p.is_alive() for p in proc]):
      for i in range(N_CORES):
        if i == N_CORES - 1:
          stdout.write("%1.1f%% done at core %d\r" % (prog[N_CORES-1], N_CORES))
        else:
          stdout.write("%1.1f%% done at core %d, " % (prog[i], i+1))
      stdout.flush()
      sleep(0.1)
    [p.join() for p in proc]
  except KeyboardInterrupt:
    [p.terminate() for p in proc]
    [p.join() for p in proc]
    print("\nProcess canceled.")
    exit(0)
  for i in range(N_rho):
    for j in range(Nz):
      phi_grid[i][j] = shared_phi_grid[i][j]


# Calculates the second radial partial derivative of the potential
# at the point (rho_axis[i], z_axis[j]). As the grid is unevenly spaced,
# a more complicated formula must be used. Formula taken from
# http://mathformeremortals.wordpress.com/
# 2013/01/12/a-numerical-second-derivative-from-three-points/
def d2phi_drho2(i, j):
  x1, x2, x3 = rho_axis[i-1], rho_axis[i], rho_axis[i+1]
  y1, y2, y3 = phi_grid[i-1][j], phi_grid[i][j], phi_grid[i+1][j]
  v1 = np.array((2/((x2-x1)*(x3-x1)),-2/((x3-x2)*(x2-x1)),2/((x3-x2)*(x3-x1))))
  v2 = np.array((y1, y2, y3))
  return np.dot(v1, v2)


def generate_sigma_grids():
  # The [0], [1] and [2] components of this grid will refer to the halo,
  # disk and bulge, respectively. The calculation being performed here
  # follows the prescription found in Springel & White, 1999.
  sz_grid = np.zeros((3, N_rho, Nz))
  ys = np.zeros((3, N_rho, Nz)) # Integrand array.
  # ys is the integrand array. Filling it.
  for i in range(N_rho):
    for j in range(0, Nz-1):
      r = (rho_axis[i]**2 + z_axis[j]**2)**0.5
      dz = z_axis[j+1] - z_axis[j]
      dphi = phi_grid[i][j+1] - phi_grid[i][j]
      ys[0][i][j] = dehnen_density(r, M_halo, a_halo, gamma_halo) * dphi/dz 
      ys[1][i][j] = disk_density(rho_axis[i], z_axis[j], M_disk, z0) * dphi/dz
      if(bulge):
        ys[2][i][j] = dehnen_density(r, M_bulge, a_bulge, gamma_bulge) * dphi/dz 
    for j in range(0, Nz-1):
      r = (rho_axis[i]**2 + z_axis[j]**2)**0.5
      sz_grid[0][i][j] = 1/dehnen_density(r, M_halo, a_halo, gamma_halo) * integrate.simps(ys[0][i][j:], z_axis[j:])
      sz_grid[1][i][j] = (1/disk_density(rho_axis[i], z_axis[j], M_disk, z0) * 
        integrate.simps(ys[1][i][j:], z_axis[j:]))
      if(bulge):
        sz_grid[2][i][j] = 1/dehnen_density(r, M_bulge, a_bulge, gamma_bulge) * integrate.simps(ys[2][i][j:], z_axis[j:])

  sphi_grid = np.zeros((3, N_rho, Nz))
#  aux_grid = np.zeros(N_rho)
  for i in range(1, N_rho-1):
    for j in range(Nz):
      r0 = (rho_axis[i]**2 + z_axis[j]**2)**0.5
      r1 = (rho_axis[i+1]**2 + z_axis[j]**2)**0.5
      drho = rho_axis[i+1] - rho_axis[i]
      dphi = phi_grid[i+1][j] - phi_grid[i][j]
      sphi_grid[0][i][j] = (sz_grid[0][i][j] + rho_axis[i]/dehnen_density(r0, M_halo, a_halo, gamma_halo) * 
        (dehnen_density(r1, M_halo, a_halo, gamma_halo)*sz_grid[0][i+1][j] - 
        dehnen_density(r0, M_halo, a_halo, gamma_halo)*sz_grid[0][i][j]) / drho + rho_axis[i] * dphi/drho)
      if(j == 0):
        kappa2 = 3/rho_axis[i] * dphi/drho + d2phi_drho2(i, j)
        gamma2 = 4/(kappa2*rho_axis[i]) * dphi/drho
        sphi_grid[1][i][j] = sz_grid[1][i][j] / gamma2
#        aux_grid[i] = (sz_grid[1][i][j] + 
#          rho_axis[i]/disk_density(rho_axis[i], z_axis[j], M_disk, z0) * 
#          (sz_grid[1][i+1][j]*disk_density(rho_axis[i+1], z_axis[j], M_disk, z0) -
#          sz_grid[1][i][j]*disk_density(rho_axis[i], z_axis[j], M_disk, z0)) /
#          drho + rho_axis[i] * dphi/drho)
        if i == N_rho-2:
          sphi_grid[1][0][j] = sphi_grid[1][1][j]
#          aux_grid[0] = aux_grid[1]
#          aux_grid[N_rho-1] = aux_grid[N_rho-2]
      if(bulge):
        sphi_grid[2][i][j] = (sz_grid[2][i][j] + rho_axis[i]/dehnen_density(r0, M_bulge, a_bulge, gamma_bulge) * 
          (dehnen_density(r1, M_bulge, a_bulge, gamma_bulge)*sz_grid[2][i+1][j] - 
          dehnen_density(r0, M_bulge, a_bulge, gamma_bulge)*sz_grid[2][i][j]) / drho + rho_axis[i] * dphi/drho)
      for k in [0, 2]:
        sphi_grid[k][0][j] = sphi_grid[k][1][j]
#  return sz_grid, sphi_grid, aux_grid
  return sz_grid, sphi_grid


def set_velocities(coords, T_cl_grid):
  sz_grid, sphi_grid = generate_sigma_grids()
  # Avoiding numerical problems. They only occur at a minor amount
  # of points, anyway. I set the values to a small number so I can
  # successfuly sample from the gaussian distributions ahead.
  sphi_grid[np.isnan(sphi_grid)] = 1.0e-5;
  sphi_grid[sphi_grid == np.inf] = 1.0e-5;
  sphi_grid[sphi_grid <= 0] = 1.0e-5;
  sz_grid[sz_grid == 0] = 1.0e-5;
#  aux_grid[np.isnan(aux_grid)] = 1.0e-5;
#  aux_grid[aux_grid == np.inf] = 1.0e-5;
#  aux_grid[aux_grid <= sphi_grid[1][0]] = (sphi_grid[1][aux_grid <= sphi_grid[1][0]] + 1.0e-5)

  vels = np.zeros((N_total, 3))
  vphis = {}
  phis = arctan2(coords[:,1], coords[:,0])
  for i, part in enumerate(coords):
    x = part[0]
    y = part[1]
    z = abs(part[2])
    rho = (x**2 + y**2)**0.5
    r = (rho**2 + z**2)**0.5
    phi = phis[i]
    bestz = interpolate(z, z_axis)
    bestr = interpolate(rho, rho_axis)
    if(i < N_gas):
      dphi = phi_grid[bestr][bestz]-phi_grid[bestr-1][bestz]
      drho = rho_axis[bestr]-rho_axis[bestr-1]
      dP = (disk_density(rho_axis[bestr], z, M_gas, z0_gas)*
          T_cl_grid[bestr][bestz] - disk_density(rho_axis[bestr-1], 
          z, M_gas, z0_gas)*T_cl_grid[bestr-1][bestz])
      vphi2 = rho * (dphi/drho + 1/disk_density(rho, z, M_gas,
          z0_gas) * dP/drho)
      vphi = abs(vphi2)**0.5
      vz = vr = 0
    elif(i >= N_gas and i < N_gas+N_halo):
      sigmaz = sz_grid[0][bestr][bestz]
      sigmap = sphi_grid[0][bestr][bestz]
      vz = nprand.normal(scale=sigmaz**0.5)
      vr = nprand.normal(scale=sigmaz**0.5)
      vphi = nprand.normal(scale=sigmap**0.5)
    elif(i >= N_gas+N_halo and i < N_gas+N_halo+N_disk):
      sigmaz = sz_grid[1][bestr][bestz]
      sigmap = sphi_grid[1][bestr][0]
      vz = nprand.normal(scale=sigmaz**0.5)
      vr = nprand.normal(scale=factor*sigmaz**0.5)
      vphi = nprand.normal(scale=factor*sigmap**0.5)
      if(bestz not in vphis):
        tck = inter.splrep(rho_axis, phi_grid[:, bestz])
        vphis[bestz] = inter.interp1d(rho_axis, inter.splev(rho_axis, tck, der=1), bounds_error=False, fill_value=0)
      if(vphis[bestz](rho) > 0):
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
 

def set_temperatures(coords_gas):
  U_grid = np.zeros((N_rho, Nz))
  U = np.zeros(N_gas)
  # Constantless temperature, will be used in the circular
  # velocity determination for the gas.
  T_cl_grid = np.zeros((N_rho, Nz)) 
  MP_OVER_KB = 121.148
  HYDROGEN_MASSFRAC = 0.76
  meanweight_n = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC)
  meanweight_i = 4.0 / (3 + 5 * HYDROGEN_MASSFRAC)
  disk_temp = 10000
  U.fill(units.temp_to_internal_energy(disk_temp))
  if(disk_temp >= 1.0e4):
    T_cl_grid.fill(disk_temp / MP_OVER_KB / meanweight_i)
  else:
    T_cl_grid.fill(disk_temp / MP_OVER_KB / meanweight_n)
  return U, T_cl_grid


def write_input_file(galaxy_data):
  coords = galaxy_data[0]
  vels = galaxy_data[1]
  ids = np.arange(1, N_total+1, 1)
  m_halo = np.empty(N_halo)
  m_halo.fill(halo_cut_M/N_halo)
  m_disk = np.empty(N_disk)
  m_disk.fill(M_disk*disk_cut/N_disk)
  if(bulge):
    m_bulge = np.empty(N_bulge)
    m_bulge.fill(bulge_cut_M/N_bulge)
  if(gas):
    U = galaxy_data[2]
    rho = galaxy_data[3]
    m_gas = np.empty(N_gas)
    m_gas.fill(M_gas*disk_cut/N_gas)
    if(bulge):
      masses = np.concatenate((m_gas, m_halo, m_disk, m_bulge))
    else:
      masses = np.concatenate((m_gas, m_halo, m_disk))
    smooths = np.zeros(N_gas)
    if Z > 0:
      Zs = np.zeros(N_gas + N_disk + N_bulge)
      Zs.fill(Z)
      write_snapshot([N_gas, N_halo, N_disk, N_bulge, 0, 0],
        data_list=[coords, vels, ids, masses, U, rho, smooths, Zs],
        outfile=output)
    else:
      write_snapshot([N_gas, N_halo, N_disk, N_bulge, 0, 0],
        data_list=[coords, vels, ids, masses, U, rho, smooths],
        outfile=output)
  else:
    if(bulge):
      masses = np.concatenate((m_halo, m_disk, m_bulge))
    else:
      masses = np.concatenate((m_halo, m_disk))
    write_snapshot([0, N_halo, N_disk, N_bulge, 0, 0],
      data_list=[coords, vels, ids, masses],
      outfile=output)


if __name__ == '__main__':
  main()
