import sys
import os
import struct
import numpy as np
from ConfigParser import ConfigParser


config = ConfigParser()
config.read("params_galaxy.ini")


def read_header(folder, n_part):
  h_data = []
  for j in n_part: # n_part
    h_data.append(int(j))
  for j in config.get('header', 'mass_array').split(','):
    h_data.append(float(j))
  h_data.append(config.getfloat('header', 'time'))
  h_data.append(config.getfloat('header', 'redshift'))
  h_data.append(config.getint('header', 'flag_sfr'))
  h_data.append(config.getint('header', 'flag_feedback'))
  for j in n_part: # n_part_total, assuming equal to n_part
    h_data.append(int(j))
  h_data.append(config.getint('header', 'flag_cooling'))
  h_data.append(config.getint('header', 'num_files'))
  h_data.append(config.getfloat('header', 'boxsize'))
  h_data.append(config.getfloat('header', 'omega0'))
  h_data.append(config.getfloat('header', 'omega_lambda'))
  h_data.append(config.getfloat('header', 'hubble_param'))
  h_data.append(config.getint('header', 'flag_age'))
  h_data.append(config.getint('header', 'flag_metals'))

  # blank, present in the header
  for i in np.arange(88):
    h_data.append('\0')
  s = struct.Struct('iiiiii dddddd d d i i iiiiii i i dddd ii cccc\
           cccccccccccccccccccccccccccccccccccccccccccccccccc\
           cccccccccccccccccccccccccccccccccc')
  packed_data = s.pack(*h_data)

  # Need raw h_data as well - D. Rennehan
  return packed_data, h_data


def write_dummy(f, values_list):
  for i in values_list:
    dummy = [i]
    s = struct.Struct('i')
    d = s.pack(*dummy)
    f.write(d)


def write_block(f, block_data, data_type, block_name):
  write_dummy(f, [8])
  f.write(struct.pack('c' * 4, *block_name))
  if(block_name == 'HEAD'):
    nbytes = 256
  else:
    fmt = data_type * len(block_data)
    nbytes = len(block_data) * 4
  write_dummy(f, [nbytes + 8, 8, nbytes]) 
  if(block_name == 'HEAD'):
    f.write(block_data) 
  else:
    f.write(struct.pack(fmt, *block_data))
  write_dummy(f, [nbytes])


def write_snapshot(n_part, folder=None, data_list=None, outfile='init.dat',
          file_format='gadget2'):
  N_gas = n_part[0]
  folder = os.getcwd()

  # Erasing the output file before opening it.
  pos_data = data_list[0]
  vel_data = data_list[1]
  ID_data = data_list[2]
  mass_data = data_list[3]
  if (N_gas > 0):
    U_data = data_list[4]
    rho_data = data_list[5]
    smoothing_data = data_list[6]

  if len(data_list) > 7:
    Z = data_list[7]
  header_data, raw_hdata = read_header(folder, n_part)

  if file_format == 'hdf5':
    import h5py
    if outfile == 'init.dat':
      outfile = 'init.hdf5'

    if raw_hdata[29] != 0:
      Z = np.zeros((np.sum(n_part), raw_hdata[29]))

      # If more than 1 metal field, data_list[7][0] acts as a fractional
      # multiplier of Z_sun instead of Z_sun itself. 
      if raw_hdata[29] == 11:
        # TODO put this in the configuration file
        # All, He, C, N, O, Ne, Mg, Si, S, Ca, Fe (Asplund 2009)
        solar_abundances = [0.02, 0.28, 3.26e-3, 1.32e-3, 8.65e-3,
                  2.22e-3, 9.31e-4, 1.08e-3, 6.44e-4,
                  1.01e-4, 1.73e-3]

        # values[0] is Z (from X + Y + Z = 1)
        # values[1] is Y
        values = [data_list[7][0] * solar_abundances[0]]

        for j in range(len(solar_abundances)):
          if j == 0:
            continue
          values.append(solar_abundances[j] * values[0])
          values[-1] /= solar_abundances[0]

        # Allow for primordial Helium abundance
        he_fac = values[0] / solar_abundances[0]
        values[1] = 0.25
        values[1] += (solar_abundances[1] - 0.25) * he_fac

        for i in range(len(Z)):
          for j in range(len(solar_abundances)):
            Z[i, j] = values[j]

      else:
        for i in range(len(Z)):
          Z[i, 0] = data_list[7][0]

    f = h5py.File(outfile, 'w')
    
    # TODO put this in header.txt
    double_precision = config.getint('global', 'double_precision_hdf5')

    header = f.create_group('Header')
    header.attrs['NumPart_ThisFile'] = np.asarray(n_part)
    header.attrs['NumPart_Total'] = np.asarray(n_part)
    header.attrs['NumPart_Total_HighWord'] = 0 * np.asarray(n_part)
    header.attrs['MassTable'] = np.zeros(6)
    header.attrs['Time'] = float(raw_hdata[12])
    header.attrs['Redshift'] = float(raw_hdata[13])
    header.attrs['BoxSize'] = float(raw_hdata[24])
    header.attrs['NumFilesPerSnapshot'] = int(raw_hdata[23])
    header.attrs['Omega0'] = float(raw_hdata[25])
    header.attrs['OmegaLambda'] = float(raw_hdata[26])
    header.attrs['HubbleParam'] = float(raw_hdata[27])
    header.attrs['Flag_Sfr'] = int(raw_hdata[14])
    header.attrs['Flag_Cooling'] = int(raw_hdata[22])
    header.attrs['Flag_StellarAge'] = int(raw_hdata[28])
    header.attrs['Flag_Metals'] = int(raw_hdata[29])
    header.attrs['Flag_Feedback'] = int(raw_hdata[15])
    # TODO Make double precision optional
    header.attrs['Flag_DoublePrecision'] = double_precision
    header.attrs['Flag_IC_Info'] = 0

    if double_precision:
      dtype = 'float64'
    else:
      dtype = 'float32'

    running_idx = 0
    for i in range(len(n_part)):
      # HDF5 format doesn't require info for particles that don't exist
      if n_part[i] == 0:
        continue

      p = f.create_group('PartType' + str(i))

      # Next set of indices
      start = running_idx
      end = start + n_part[i]
      running_idx += n_part[i]

      assert np.shape(pos_data[start:end]) == (n_part[i], 3)
      assert np.shape(vel_data[start:end]) == (n_part[i], 3)
      assert np.shape(ID_data[start:end]) == (n_part[i], )
      assert np.shape(mass_data[start:end]) == (n_part[i], )

      p.create_dataset('Coordinates', data = pos_data[start:end],
                dtype = dtype)
      p.create_dataset('Velocities', data = vel_data[start:end],
                dtype = dtype)
      p.create_dataset('ParticleIDs', data = ID_data[start:end],
                dtype = 'uint32')
      p.create_dataset('Masses', data = mass_data[start:end],
                dtype = dtype)
     
      # TODO currently all gas+stars get the same metallicity.
      # this should be an option in the configuration as well.
      if i in [0, 2, 3, 4]:
        if raw_hdata[29] > 1:
          assert np.shape(Z[start:end]) == (n_part[i], raw_hdata[29])

        p.create_dataset('Metallicity', data = Z[start:end],
                  dtype = dtype)

      if i == 0 and N_gas > 0:
        assert np.shape(U_data[start:end]) == (n_part[i], )
        assert np.shape(rho_data[start:end]) == (n_part[i], )
        assert np.shape(smoothing_data[start:end]) == (n_part[i], )

        p.create_dataset('InternalEnergy', data = U_data[start:end],
                  dtype = dtype)
        p.create_dataset('Density', data = rho_data[start:end],
                  dtype = dtype)
        p.create_dataset('SmoothingLength', 
                  data = smoothing_data[start:end],
                  dtype = dtype)

    f.close()
  else:
    open(outfile, 'w').close()

    with open(outfile, 'a') as f:
      write_block(f, header_data, None, 'HEAD')
      write_block(f, pos_data, 'f', 'POS ')
      write_block(f, vel_data, 'f', 'VEL ')
      write_block(f, ID_data, 'i', 'ID  ')
      write_block(f, mass_data, 'f', 'MASS')

      if (N_gas > 0):
        write_block(f, U_data, 'f', 'U   ')

        if (len(data_list) > 7):
          write_block(f, Z, 'f', 'Z   ')

        write_block(f, rho_data, 'f', 'RHO ')
        write_block(f, smoothing_data, 'f', 'HSML')
