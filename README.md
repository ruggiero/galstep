## DEPRECATED!

After dedicating a lot of time to this method, I have found that,
for the parameters I was interested in, the results it generates for
the halo are quite gross near its center. The hyphothesis that the halo
velocity distributions are approximately gaussian is far from reasonable,
as they display increasingly high kurtosis as you get closer to r =
0. In practice, the nucleus of the generated halo simply explodes when
you actually simulate it.

I moved on from this method to the iteractive method from Radionov,
Athanassoula & Sotnikova (2009), which is both easier to implement and
more precise, although it demands more computing power. You can find my
new code under the repository 'galaxy-iter'.


## About

This code uses the algorithm found in Springel & Di Matteo & Hernquist
(2005) for generating the initial conditions for a galaxy simulation
with the code GADGET-2, including a gas component with temperatures
calculated for guaranteeing hydrodynamic equilibrium.


## Required libraries
 
* NumPy (python-numpy)
* SciPy (python-scipy)


## Usage

### galaxy.py

    usage: galaxy.py [-h] [--halo-core] [--bulge-core] [-cores CORES] [-temp TEMP]
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


A sample potential data file is provided for the parameters file included.


## Author

    Rafael Ruggiero
    Undergraduate student at Universidade de São Paulo (USP), Brazil
    Contact: bluewhale [at] cecm.usp.br


## Disclaimer

Feel free to use this code in your work, but please link this page
in your paper.
