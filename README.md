## About

This code uses the algorithm found in Springel, Di Matteo & Hernquist
(2005) for generating the initial conditions for a disk galaxy simulation
with the codes GADGET-2 or RAMSES (using the [DICE patch](https://bitbucket.org/vperret/dice/wiki/RAMSES%20simulation)), including a thin isothermal gas component which
must me relaxed for a few hundred Myr to reach equilibrium. If you are
using a simulation code which includes radiative cooling, the relaxation
will be a lot faster (a few tens of Myr at most).

Important: this method fails to generate low mass halos (~10^10 solar
masses) in equilibrium, since the velocity distribution for these near
the center is strongly non gaussian. Also, note that if you run GADGET-2
without any gas cooling, the gaseous disk will gain lots of energy over
time due to the artificial viscosity, and will get a lot thicker.


## Required libraries
 
* NumPy (python-numpy)
* SciPy (python-scipy)
* [pyGadgetReader](https://bitbucket.org/rthompson/pygadgetreader)


## Usage

You can run `python galstep.py --help` to see the message below. 

    usage: galstep.py [-h] [--halo-core] [--bulge-core] [-cores CORES] [-temp TEMP]
                     [--force-yes] [-o init.dat]

    Generates an initial conditions file for a galaxy simulation with halo,
    stellar disk, gaseous disk and bulge components.

    optional arguments:
      -h, --help    show this help message and exit
      --nogas       Generates a galaxy without gas.
      -cores CORES  The number of cores to use during the potential canculation.
                    Default is 1. Make sure this number is a factor of N_rho*N_z.
      --force-yes   Don't ask if you want to use the existing potential_data.txt
                    file. Might be useful for automating the execution of the
                    script.
      -o init.dat   The name of the output file.


## Troubleshooting

If you are getting *OSError: [Errno 24] Too many open files* while trying
to run this code in OSX, please try [this](https://superuser.com/questions/302754/increase-the-maximum-number-of-open-file-descriptors-in-snow-leopard/514049#514049).


## Works which used this code

* [Ruggiero & Lima Neto (2017)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1703.08550)


## Author

    Rafael Ruggiero
    Ph.D student at Universidade de São Paulo (USP), Brazil
    Contact: rafael.ruggiero [at] usp.br


## Disclaimer

Feel free to use this code in your work, but please link this page
in your paper.
