import sys
import os
import struct
import numpy as np

def read_header(n_part):
    h_data = []

    #TODO enable funtionalities from the config parser
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
    return packed_data, h_data

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
    header_data, raw_hdata = read_header(n_part)
    pos_data = data_list[0]
    vel_data = data_list[1]
    ID_data = data_list[2]
    mass_data = data_list[3]
    if(N_gas > 0):
        U_data = data_list[4]
        rho_data = data_list[5]
        smoothing_data = data_list[6]
    #Metalicity?
    if len(data_list) > 7:
        Z = data_list[7]

    if file_format is 'gadget2':
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
    
    elif file_format is 'hdf5':
        import h5py

        # At this point in the original galstep file there are what
        # looks like random definition of values, which shouldn't
        # be there in order to keep generality.

        with h5py.File(outfile, 'w') as f:
            # TODO Make double precision optional
            double_precision = True
            
            if double_precision:
                dtype = 'float64'
            else:
                dtype = 'float32'

            #Header
            #TODO Insert this in the header function
            #create a function called here that receives the object
            # f and a file_format flag, generate the group 'Header'
            # and insert all information *if* the flag file_format
            # is 'hdf5'
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
            header.attrs['Flag_DoublePrecision'] = double_precision
            header.attrs['Flag_IC_Info'] = 0
            
            
            #Particle families
            for i, j in enumerate(n_part):
                # HDF5 format doesn't require info for particles that
                # don't exist
                if j:
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
                                dtype = dtype)
                    current_family.create_dataset('Masses',
                                data = mass_data[start_index:end_index],
                                dtype = dtype)

                    # TODO currently all gas+stars get the same metallicity.
                    # this should be an option in the configuration as well.
                    
                    #Metallicity properties
                    if i in [0, 2, 3, 4]:
                        current_family.create_dataset('Metallicity',
                                    data = Z[start_index:end_index],
                                    dtype = dtype)


                    #Gas specific properties
                    if i is 0 and N_gas > 0:
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


