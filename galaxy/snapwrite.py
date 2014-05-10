'''
DESCRIPTION:

Script that actually writes a snapshot, either from a set of text files,
or by directly passing the necessary data to the function write_snapshot
(for details on the syntax, read the function).

In the first case, just run the following:

python snapwrite.py FOLDER

FOLDER: folder containing all the input files, which are:

    header.txt position.txt velocity.txt id.txt masses.txt (always)
    energy.txt density.txt smoothing.txt (in case there is gas)

A file named 'init.dat' will be created.  Note that the columns of the
positions and velocities files should be separated with tabs, as well
as the columns of the header. Among the input files, header.txt is the
only one that accepts comments, which should be preceded by a #.
'''

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


def read_header(folder, n_part):
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

    # blank, present in the header
    for i in np.arange(96):
        h_data.append('\0')
    s = struct.Struct('iiiiii dddddd d d i i iiiiii i i dddd cccccccccccc\
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


def write_snapshot(n_part, folder=None, from_text=True, data_list=None,
                   outfile='init.dat'):
    N_gas = n_part[0]
    if(from_text and not folder):
        print ("error: can't call write_snapshot with from_text=True\n"
               "and without an input files folder.")
    if(not from_text):
        folder = os.getcwd()

    # Erasing the input file before opening it.
    open(outfile, 'w').close()
    f = file(outfile, 'a')
    header_data = read_header(folder, n_part)
    if(from_text):
        pos_data = np.fromfile(folder + "position.txt", sep='\t')
        vel_data = np.fromfile(folder + "velocity.txt", sep='\t')
        ID_data = np.fromfile(folder + "id.txt", dtype=int, sep='\t')
        mass_data = np.fromfile(folder + "masses.txt", sep='\t')
        if(N_gas > 0):
            U_data = np.fromfile(folder + "energy.txt", sep='\t')
            rho_data = np.fromfile(folder + "density.txt", sep='\t')
            smoothing_data = np.fromfile(folder + "smoothing.txt", sep='\t')
    else:
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
    f.close()
