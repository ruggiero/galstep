'''
Execution: python temperatures.py SNAPSHOT | tee outfile.txt
'''

import sys

import numpy as np

from os import path
from sys import path as syspath
syspath.append(path.join(path.dirname(__file__), '..', 'misc'))
from units import temp_to_internal_energy, internal_energy_to_temp
import matplotlib.pyplot as plt



# For reading and storing the header.
class header:
    def __init__(self, snapshot):
        initialize_block(snapshot)
        self.n_part = np.fromfile(snapshot, 'int32', 6)
        self.mass = np.fromfile(snapshot, 'float64', 6)
        self.time = np.fromfile(snapshot, 'float64', 1)
        self.redshift = np.fromfile(snapshot, 'float64', 1)
        self.flag_sfr = np.fromfile(snapshot, 'int32', 1)
        self.flag_feedback = np.fromfile(snapshot, 'int32', 1)
        self.n_part_total = np.fromfile(snapshot, 'int32', 6)
        self.flag_cooling = np.fromfile(snapshot, 'int32', 1)
        self.num_files = np.fromfile(snapshot, 'int32', 1)
        self.box_size = np.fromfile(snapshot, 'float64', 1)
        self.omega0 = np.fromfile(snapshot, 'float64', 1)
        self.omega_lambda = np.fromfile(snapshot, 'float64', 1)
        self.hubble_param = np.fromfile(snapshot, 'float64', 1)
        self.fill = np.fromfile(snapshot, 'int8', 96)
        read_dummy(snapshot, 1)


# These declared variables are for clarity purposes.
class particle:
    pos = None
    vel = None
    mass = 0.0
    U = None
    rho = None
    smoothing = None
    ID = None


# For reading the leading and trailing ints that are present in each block.
def read_dummy(snapshot, n_dummies):
    for i in np.arange(n_dummies):
        dummy = np.fromfile(snapshot, 'int32', 1)


def initialize_block(snapshot):
    read_dummy(snapshot, 1)
    block_ID = np.fromfile(snapshot, 'int8', 4)
    read_dummy(snapshot, 3)
    return ''.join([chr(i) for i in block_ID])

    
def read_data(snapshot, h):
    p_list = []
    n_part = sum(h.n_part)
    range_ = np.arange(n_part)
    
    # Positions
    initialize_block(snapshot)
    for i in range_:
        p = particle()
        p.pos = np.fromfile(snapshot, 'float32', 3)
        p_list.append(p)
    read_dummy(snapshot, 1)
    
    # Velocities
    initialize_block(snapshot)
    for i in range_:
        p_list[i].vel = np.fromfile(snapshot, 'float32', 3)
    read_dummy(snapshot, 1)

    # IDs
    initialize_block(snapshot)
    for i in range_:
        p_list[i].ID = np.fromfile(snapshot, 'int32', 1)[0]
    read_dummy(snapshot, 1)

    # Variable masses, which are read in case the mass of the
    # particle of type 'i' is declared as 0, in the header.
    cur = 0
    read_something = 0
    for i in np.arange(6):
        if(h.mass[i] != 0):
            cur += h.n_part[i]
            continue
        else:
            if(read_something == 0):
                read_something = 1
                initialize_block(snapshot)
            for j in np.arange(h.n_part[i]):
                p_list[cur].mass = np.fromfile(snapshot, 'float32', 1)[0]
                cur += 1

    # The variable masses block might not exist.
    if(read_something):
        read_dummy(snapshot, 1)
    
    # Blocks related to the internal energies, densities and smoothing
    # lengths of the gas particles, in case there is any.
    if(h.n_part[0] > 0):
        range_ = np.arange(h.n_part[0])

        # First the energies
        initialize_block(snapshot)
        for i in range_:
            p_list[i].U = np.fromfile(snapshot, 'float32', 1)[0]
        read_dummy(snapshot, 1)

        # Then the densities
        initialize_block(snapshot)
        for i in range_:
            p_list[i].rho = np.fromfile(snapshot, 'float32', 1)[0]
        read_dummy(snapshot, 1)

        # And the smoothing lengths
        initialize_block(snapshot)
        for i in range_:
            p_list[i].smoothing = np.fromfile(snapshot, 'float32', 1)[0]
        read_dummy(snapshot, 1)
    
    # There are some optional blocks that are yet to be implemented. So,
    # in case any of them is present, send a warning.
    chunk = snapshot.read()
    if chunk:
        print "There still were things to be read..."
    return p_list


def main():
    snapshot = open(sys.argv[1], 'r')
    h = header(snapshot)
    p_list = read_data(snapshot, h)
    snapshot.close()

    # From here on, the data for all the particles is accessible
    # As an example, showing the positions of all the particles
    points = []
    for i, p in enumerate(p_list):
        if(i < h.n_part[0]):
            points.append([(p.pos[0]**2 + p.pos[1]**2)**0.5, internal_energy_to_temp(p.U)])
        else:
            break
    points = sorted(points)
    plt.plot([p[0] for p in points], [p[1] for p in points])
    plt.savefig(sys.argv[1] + "_temp.png")

if __name__ == '__main__':
    main()
