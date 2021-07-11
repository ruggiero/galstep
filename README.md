## About this specific version

Modified in order to be compatible with Python3 among other improvements, possibly making it easier for future usage and maintenance.

Made possible with advice and contributions from [Prof. Dr. Rubens Machado](https://sites.google.com/professores.utfpr.edu.br/rubensmachado/home).


## About

This code uses the algorithm described in Springel, Di Matteo & Hernquist
(2005) for generating the initial conditions for a disk galaxy simulation
with the codes GADGET-2 or RAMSES (using the [DICE patch](https://bitbucket.org/vperret/dice/wiki/RAMSES%20simulation)), including 
a stellar disk, a gaseous disk, a dark matter halo and a stellar bulge. The
first two components follow an exponential density profile, and the last
two a Dehnen density profile with gamma=1 by default, corresponding to a 
Hernquist profile. You can check out the expressions in
[Ruggiero & Lima Neto (2017)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1703.08550).

Some notes which might save you some time:

1- Using this method, some combinations of component masses lead the dark
matter halo to be unrelaxated near the very center of the galaxy. If
that is the case for your galaxy, you will see a ring appearing in
its stellar disk in less than one rotation time. When that happens,
I like to relaxate the halo for a few hundred Myr while keeping the
other components frozen (both Gadget2 and RAMSES can do that) in order
to ensure a good state of equilibrium.

2- If you run GADGET-2 without any gas cooling, the gaseous disk will
gain lots of energy over time due to the artificial viscosity, and will
get a lot thicker.

3- The vertical equilibrium of the gas component depends on your
simulation setup.  If you use radiative cooling, it will settle on a
very short timescale (a few tens of Myr at most).  If not, it will take
about one rotation period to reach equilibrium.

About units: the value for the gravitational constant G used in this code
is such that the unit for length is 1.0 kpc, for mass 1.0e10 solar masses,
and for velocity 1.0 km/s. This is the default for GADGET-2, and works out
of the box in RAMSES with the DICE patch.


## Required libraries
 
* NumPy (python-numpy)
* SciPy (python-scipy)
* [pyGadgetReader](https://bitbucket.org/rthompson/pygadgetreader)
* h5py (only if you need the HDF5 file format, python-h5py)


## Usage

You can run `python galstep.py --help` to see the message below. Also please
check out the `galaxy_params.ini` file to see the available free parameters.

    usage: galstep.py [-h] [-cores CORES] [--force-yes] [--force-no] [--hdf5]
                      [-o init.dat] [-i params_galaxy.ini]
    
    Generates an initial conditions file for a galaxy simulation with halo,
    stellar disk, gaseous disk and bulge components.
    
    optional arguments:
      -h, --help            show this help message and exit
      -cores CORES          The number of cores to use during the potential
                            canculation. Make sure this number is a factor of
                            N_rho*N_z. Default is 1.
      --force-yes           Don't ask if you want to use the existing
                            potential_data.txt file. Useful for automating the
                            execution of the script.
      --force-no            Same as above, but with the opposite effect.
      --hdf5                Output initial conditions in HDF5 format.
      -o init.dat           The name of the output file.
      -i params_galaxy.ini  The name of the .ini file.

## Troubleshooting

If you are getting *OSError: [Errno 24] Too many open files* while trying
to run this code in OSX, please try [this](https://superuser.com/questions/302754/increase-the-maximum-number-of-open-file-descriptors-in-snow-leopard/514049#514049).


## Works which used this code

* [Ruggiero & Lima Neto (2017)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1703.08550)


## Disclaimer

Feel free to use this code in your work, but please link this page
in your paper.
