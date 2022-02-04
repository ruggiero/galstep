import struct
import numpy as np

def read_header(n_part):
    h_data = []

    #TODO enable funtionalities from the config parser
    #TODO make it less hardcoded
    for j in n_part: # n_part
        h_data.append(int(j))
    for j in range(6): # mass table
        h_data.append(0.0)
    h_data.append(0.0) # time
    h_data.append(0.0) # redshift
    h_data.append(0) # flag_sfr
    h_data.append(0) # flag_feedback
    for j in n_part:
        h_data.append(int(j)) # n_part_total
    h_data.append(0) # flag_coooling
    h_data.append(1) # num_files
    h_data.append(0.0) # box_size
    h_data.append(0.0) # omega0
    h_data.append(0.0) # omega_lambda
    h_data.append(1.0) # hubble_param
    h_data.append(0) #flag_age
    h_data.append(0) #flag_metals
    for i in np.arange(88):
        h_data.append(b'\x00')
    s = struct.Struct('iiiiii dddddd d d i i iiiiii i i dddd ii cccc\
    cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\
    cccccccccccccccccccc')
    packed_data = s.pack(*h_data)
    return packed_data

def read_header_hdf5(file, n_part, double_precision=1):
    #TODO enable funtionalities from the config parser
    #TODO make it less hardcoded
    header = file.create_group('Header')
    header.attrs['NumPart_ThisFile'] = np.asarray(n_part)
    header.attrs['NumPart_Total'] = np.asarray(n_part)
    header.attrs['NumPart_Total_HighWord'] = 0 * np.asarray(n_part)
    header.attrs['MassTable'] = np.zeros(6)
    header.attrs['Time'] = float(0.0)
    header.attrs['Redshift'] = float(0.0)
    header.attrs['BoxSize'] = float(0.0)
    header.attrs['NumFilesPerSnapshot'] = int(1)
    header.attrs['Omega0'] = float(0.0)
    header.attrs['OmegaLambda'] = float(0.0)
    header.attrs['HubbleParam'] = float(1.0)
    header.attrs['Flag_Sfr'] = int(0.0)
    header.attrs['Flag_Cooling'] = int(0)
    header.attrs['Flag_StellarAge'] = int(0)
    header.attrs['Flag_Metals'] = int(0)
    header.attrs['Flag_Feedback'] = int(0)
    header.attrs['Flag_DoublePrecision'] = double_precision
    header.attrs['Flag_IC_Info'] = 0

def write_dummy(f, values_list):
    for i in values_list:
        dummy = [i]
        s = struct.Struct('i')
        d = s.pack(*dummy)
        f.write(d)

def write_block(f, block_data, data_type, block_name):
    write_dummy(f, [8])
    block_name_binary = [bytes(i, 'utf-8') for i in block_name]
    f.write(struct.pack('c' * 4, *block_name_binary))
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

def write_snapshot(n_part, data_list, outfile='init.dat',
                    file_format='gadget2'):
    N_gas = n_part[0]

    #Getting data
    pos_data = data_list[0]
    vel_data = data_list[1]
    ID_data = data_list[2]
    mass_data = data_list[3]
    if(N_gas > 0):
        U_data = data_list[4]
        rho_data = data_list[5]
        smoothing_data = data_list[6]
    if len(data_list) > 7:
        Z = data_list[7]
    else:
        Z = None

    if file_format == 'gadget2':
        header_data = read_header(n_part)
        with open(outfile, 'wb') as f:
            write_block(f, header_data, None, 'HEAD')
            write_block(f, pos_data, 'f', 'POS ')
            write_block(f, vel_data, 'f', 'VEL ')
            write_block(f, ID_data, 'i', 'ID  ')
            write_block(f, mass_data, 'f', 'MASS')
            if(N_gas > 0):
               write_block(f, U_data, 'f', 'U   ')
               write_block(f, rho_data, 'f', 'RHO ')
               write_block(f, smoothing_data, 'f', 'HSML')
    
    elif file_format == 'hdf5':
        import h5py

        pos_data.shape = (len(pos_data)//3, 3)
        vel_data.shape = (len(vel_data)//3, 3)

        # At this point in the original galstep file there are what
        # looks like random definition of values, which shouldn't
        # be there in order to keep generality.
        # The code works fine without it.

        with h5py.File(outfile, 'w') as f:
            # TODO Make double precision optional
            double_precision = 1

            if double_precision:
                dtype = 'float64'
            else:
                dtype = 'float32'
            #Header
            read_header_hdf5(f, n_part, double_precision)

            #Particle families
            for i, j in enumerate(n_part):
                # HDF5 format doesn't require info for particles that
                # don't exist
                if j == 0:
                    continue
                else:
                    current_family = f.create_group('PartType'+str(i))
                    start_index = sum(n_part[:i])
                    end_index = sum(n_part[:i+1])
                    current_family.create_dataset('Coordinates',
                                data = pos_data[start_index:end_index],
                                dtype = dtype)
                    current_family.create_dataset('Velocities',
                                data = vel_data[start_index:end_index],
                                dtype = dtype)
                    current_family.create_dataset('ParticleIDs',
                                data = ID_data[start_index:end_index],
                                dtype = 'uint32')
                    current_family.create_dataset('Masses',
                                data = mass_data[start_index:end_index],
                                dtype = dtype)
                    # TODO currently all gas+stars get the same metallicity.
                    # this should be an option in the configuration as well.

                    # Metallicity properties - not needed for now
                    #if (i in [0, 2, 3, 4]) and (Z != None):
                    #    current_family.create_dataset('Metallicity',
                    #                data = Z[start_index:end_index],
                    #                dtype = dtype)

                    #Gas specific properties
                    if i == 0 and N_gas > 0:
                        current_family.create_dataset('InternalEnergy',
                                data = U_data[start_index:end_index],
                                dtype = dtype)
                        current_family.create_dataset('Density',
                                data = rho_data[start_index:end_index],
                                dtype = dtype)
                        current_family.create_dataset('SmoothingLength',
                                data = smoothing_data[start_index:end_index],
                                dtype = dtype)

                    

    else:
        raise ValueError(f'{file_format} is not a supported file format.')


