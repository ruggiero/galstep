'''

DESCRIPTION:

Script that plots density, temperature and radial velocity profiles,
given as an input a snapshot containing a galaxy cluster halo, with both
gas and dark matter components following a Dehnen density profile with
either gamma = 0 or gamma = 1.


USAGE:

profiles.py [-h] [--gas-core] [--dm-core] -i file.dat

optional arguments:
  -h, --help   show this help message and exit
  --gas-core   Sets the density profile for the gas to have a core.
  --dm-core    The same, but for the dark matter.
  -i file.dat  The name of the input file.

'''


from sys import path as syspath
from os import path
from bisect import bisect_left
from argparse import ArgumentParser as parser

import numpy as np
from numpy import pi, cos, sin, arctan
from scipy import integrate
import matplotlib
matplotlib.use('Agg') # To be able to plot under an SSH session.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from snapread import read_data, header
import centers


G = 43007.1

# These variables must be updated manually.
a_halo = 50
a_bulge = 5
M_halo = 10
M_bulge = 1
time = 0.0


def main():
    input_ = init()
    print "reading..."
    data_halo, data_disk, data_bulge  = process_data(input_)
    part_halo, aux_halo = log_partition(data_halo, 1.3)
    part_disk, aux_disk = log_partition(data_disk, 1.3)
    part_bulge, aux_bulge = log_partition(data_bulge, 1.3)
    density_plot(input_, data_halo, part_halo, aux_halo)
    density_plot(input_, data_bulge, part_bulge, aux_bulge, bulge=True)
    circular_velocity_plot(input_, data_disk)


def init():
    global halo_core, bulge_core
    flags = parser(description="Plots stuff.")
    flags.add_argument('--halo-core', help='Sets the density profile for the\
                       halo to have a core.', action='store_true')
    flags.add_argument('--bulge-core', help='The same, but for the bulge.',
                       action='store_true')
    flags.add_argument('i', help='The name of the input file.',
                       metavar="file.dat")
    args = flags.parse_args()
    halo_core = args.halo_core
    bulge_core = args.bulge_core
    input_ = args.i
    return input_


def process_data(input_):
    global time, N_halo, N_disk, N_bulge
    snapshot = open(input_, 'r')
    h = header(snapshot)
    time = h.time
    N_halo = h.n_part_total[1]
    N_disk = h.n_part_total[2]
    N_bulge = h.n_part_total[3]
    p_list = read_data(snapshot, h)
    snapshot.close()
    data_halo = []
    data_disk = []
    data_bulge = []
    COD = centers.COD(p_list)
    for i in p_list:
        i.pos -= COD
        if(i.ID <= N_halo):
            r = np.linalg.norm(i.pos)
            data_halo.append([r])
        elif(i.ID > N_halo and i.ID <= N_halo+N_disk):
            x = i.pos[0]
            y = i.pos[1]
            z = i.pos[2]
            #if(abs(z) > 0.5):
            #    continue
            if(x > 0 and y > 0):
                phi = arctan(y/x)
            elif(x < 0 and y > 0):
                phi = pi - arctan(-y/x)
            elif(x < 0 and y < 0):
                phi = pi + arctan(y/x)
            elif(x > 0 and y < 0):
                phi = 2 * pi - arctan(-y/x)
            vphi = i.vel[1]*np.cos(phi) - i.vel[0]*np.sin(phi)
            rho = (x**2 + y**2)**0.5
            data_disk.append([rho, vphi])
        elif(i.ID > N_halo+N_disk):
            r = np.linalg.norm(i.pos)
            data_bulge.append([r])
    del(p_list)
    data_halo = sorted(data_halo)
    data_disk = sorted(data_disk)
    data_bulge = sorted(data_bulge)
    return data_halo, data_disk, data_bulge
 

def density(r, core=False, bulge=False):
    if(bulge):
        if(core):
            return (3*M_bulge*a_bulge) / (4*np.pi*(r+a_bulge)**4)
        else:
            if(r == 0):
                return 0
            else:
                return (M_bulge*a_bulge) / (2*np.pi*r*(r+a_bulge)**3)
    else:
        if(core):
            return (3*M_halo*a_halo) / (4*np.pi*(r+a_halo)**4)
        else:
            if(r == 0):
                return 0
            else:
                return (M_halo*a_halo) / (2*np.pi*r*(r+a_halo)**3)




# Given a data vector, in which each element represents a different
# PARTICLe by a list of the form [radius, radial_velocity^2], ordered
# according to the radii; and a multiplication factor, returns the right
# indexes of a log partition of the vector. Also returns an auxiliary
# vector, which will be useful in the functions that calculate the
# distribution functions.
def log_partition(data, factor):
    limits = []
    auxiliary = []
    radii = [i[0] for i in data]
    left_limit = 0
    right_limit = 0.01
    left_index = 0
    while(right_limit < 200 * a_halo):
        # Before right_index, everybody is smaller than right_limit.
        right_index = left_index + bisect_left(radii[left_index:], right_limit)
        limits.append(right_index)
        auxiliary.append([right_index - left_index, (right_limit + left_limit) /
                          2])
        left_limit = right_limit
        left_index = right_index
        right_limit *= factor
    return limits, auxiliary


# Returns a list containing elements of the form [radius, density].
def density_distribution(data, partition, aux, bulge=False):
    distribution = []
    left = 0
    if(bulge):
        cte = (10**10*3*M_bulge) / (4*np.pi*N_bulge)
    else:
        cte = (10**10*3*M_halo) / (4*np.pi*N_halo)
    for j in np.arange(len(partition)):
        right = partition[j]
        if(right >= len(data)):
            break
        count = aux[j][0]
        middle_radius = aux[j][1]
        if(count > 0):
            density = (cte * count) / (data[right][0]**3 - data[left][0]**3)
            distribution.append([middle_radius, density])
        else:
            distribution.append([middle_radius, 0])
        left = right
    return distribution



def density_plot(input_, data, part, aux, bulge=False):
    dist = density_distribution(data, part, aux, bulge)
    x_axis = np.logspace(np.log10(dist[0][0]), np.log10(dist[-1][0]), num=500)
    p1, = plt.plot([i[0] for i in dist], [i[1] for i in dist], 'o')
    if(bulge):
        p2, = plt.plot(x_axis, [10**10 * density(i, core=bulge_core, bulge=True) for i in x_axis])
    else:
        p2, = plt.plot(x_axis, [10**10 * density(i, core=halo_core) for i in x_axis])
    plt.legend([p1, p2], ["Simulation", "Theoretical value"], loc=1)
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Density ( M$_{\odot}$/kpc$^3$)")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1, 10**4])
    plt.ylim([1, 10**10])
    if(bulge):
        plt.title("Bulge, t = %1.2f Gyr" % time)
        plt.savefig(input_ + "-bulge-density.png")
        print "Done with bulge density for " + input_
    else:
        plt.title("Halo, t = %1.2f Gyr" % time)
        plt.savefig(input_ + "-halo-density.png")
        print "Done with halo density for " + input_
    plt.close()


def circular_velocity_plot(input_, data_disk):
    #formatter = FuncFormatter(lambda x, pos : "%1.2f" % (x / 10**6))
    #ax = plt.subplot(111)
    #ax.yaxis.set_major_formatter(formatter)
    p1, = plt.plot([i[0] for i in data_disk], [i[1] for i in data_disk], 'o')
    plt.legend('Circular velocity', loc=1)
    plt.xlabel("Radius (kpc)")
    plt.ylabel("$v_c$ (km/s)")
    #plt.xscale('log')
    plt.title("t = %1.2f Gyr" % time)
    #plt.gcf().subplots_adjust(left=0.17)
    #plt.yscale('log')
    plt.xlim([1, 200])
    #plt.ylim([0, 2.5 * 10**6])
    plt.ylim([0, 100])
    plt.savefig(input_ + "-circular-velocity.png")
    plt.close()
    print "Done with circular velocity for " + input_


if __name__ == '__main__':
    main()
