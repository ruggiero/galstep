import sys
import os
import struct
import numpy as np

def read_header(n_part):
    h_data = []
    for j in n_part: # n_part
        h_data.append(int(j))
    for j in range(6): # mass table
        h_data.append(0.0)
    h_data.append(0.0) # time
    h_data.append(0.0) # redshift
    h_data.append(int(0)) # flag_sfr
    h_data.append(int(0)) # flag_feedback
    for j in n_part:
        h_data.append(int(j)) # n_part_total
    h_data.append(int(0)) # flag_coooling
    h_data.append(int(1)) # num_files
    h_data.append(0.0) # box_size
    h_data.append(0.0) # omega0
    h_data.append(0.0) # omega_lambda
    h_data.append(1.0) # hubble_param
    for i in np.arange(96):
        h_data.append(b'\x00')
    s = struct.Struct('iiiiii dddddd d d i i iiiiii i i dddd \
    ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\
    ccccccccccccccccccccccccccccccccccccccc')
    packed_data = s.pack(*h_data)
    return packed_data

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

def write_snapshot(n_part, data_list=None, outfile='init.dat'):
    N_gas = n_part[0]
    with open(outfile, 'wb') as f:
        header_data = read_header(n_part)
        pos_data = data_list[0]
        vel_data = data_list[1]
        ID_data = data_list[2]
        mass_data = data_list[3]
        if(N_gas > 0):
            U_data = data_list[4]
            rho_data = data_list[5]
            smoothing_data = data_list[6]
        write_block(f, header_data, None, 'HEAD')
        write_block(f, pos_data, 'f', 'POS ')
        write_block(f, vel_data, 'f', 'VEL ')
        write_block(f, ID_data, 'i', 'ID  ')
        write_block(f, mass_data, 'f', 'MASS')
        if(N_gas > 0):
            write_block(f, U_data, 'f', 'U   ')
            write_block(f, rho_data, 'f', 'RHO ')
            write_block(f, smoothing_data, 'f', 'HSML')
