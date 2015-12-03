import sys
import os
import struct

import numpy as np


# Creates a list containing the non-commented, non-empty lines
# of the input file for the header.
def process_input(file_):
    h_file = []
    input_ = open(file_, 'r')
    for line in input_:
        if line.find("#") != -1:
            continue
        elif line.find("\n") == 0:
            continue
        else:
            h_file.append(line.split('\t'))
    return h_file


def read_header(folder, n_part, flag_metals=False):
    h_file = process_input(folder + "/header.txt")
    h_data = []
    for j in n_part: # n_part
        h_data.append(int(j))
    for j in h_file[0][0:6]: # mass
        h_data.append(float(j))
    h_data.append(float(h_file[1][0])) # time
    h_data.append(float(h_file[2][0])) # redshift
    h_data.append(int(h_file[3][0])) # flag_sfr
    h_data.append(int(h_file[4][0])) # flag_feedback
    for j in n_part:
        h_data.append(int(j)) # n_part_total
    h_data.append(int(h_file[5][0])) # flag_coooling
    h_data.append(int(h_file[6][0])) # num_files
    h_data.append(float(h_file[7][0])) # box_size
    h_data.append(float(h_file[8][0])) # omega0
    h_data.append(float(h_file[9][0])) # omega_lambda
    h_data.append(float(h_file[10][0])) # hubble_param
    h_data.append(int(h_file[11][0])) # flag_age
    if(flag_metals): # flag_metals
      h_data.append(1) 
    else:
      h_data.append(int(h_file[12][0]))

    # blank, present in the header
    for i in np.arange(88):
        h_data.append('\0')
    s = struct.Struct('iiiiii dddddd d d i i iiiiii i i dddd ii cccc\
                       cccccccccccccccccccccccccccccccccccccccccccccccccc\
                       cccccccccccccccccccccccccccccccccc')
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


def write_snapshot(n_part, folder=None, data_list=None, outfile='init.dat'):
    N_gas = n_part[0]
    folder = os.getcwd()

    # Erasing the output file before opening it.
    open(outfile, 'w').close()
    f = file(outfile, 'a')
    pos_data = data_list[0]
    vel_data = data_list[1]
    ID_data = data_list[2]
    mass_data = data_list[3]
    if len(data_list) > 7:
        Z = data_list[7]
        header_data = read_header(folder, n_part, flag_metals=True)
    else:
        header_data = read_header(folder, n_part)
    write_block(f, header_data, None, 'HEAD')
    write_block(f, pos_data, 'f', 'POS ')
    write_block(f, vel_data, 'f', 'VEL ')
    write_block(f, ID_data, 'i', 'ID  ')
    write_block(f, mass_data, 'f', 'MASS')
    if(N_gas > 0):
        U_data = data_list[4]
        rho_data = data_list[5]
        smoothing_data = data_list[6]
        write_block(f, U_data, 'f', 'U   ')
        if(len(data_list) > 7):
            write_block(f, Z, 'f', 'Z   ')
        write_block(f, rho_data, 'f', 'RHO ')
        write_block(f, smoothing_data, 'f', 'HSML')
    f.close()
