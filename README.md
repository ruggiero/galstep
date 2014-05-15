## Required libraries
(and the names of the packages in Ubuntu)
 
* NumPy (python-numpy)
* SciPy (python-scipy)
* Cython (cython), higher than 0.17.4
* python-dev


## Installation

galaxy.py uses a custom Cython library that has to be compiled.
For compiling, just cd to /galaxy and type 'make'. A new file, named
optimized_funcions.so, will be created, and then galaxy.py will
be ready for execution.


## Usage

### galaxy.py

    usage: galaxy.py [-h] [--halo-core] [--bulge-core] [-cores CORES] [-temp TEMP]
                     [--force-yes] [-o init.dat]

    Generates an initial conditions file for a galaxy simulation with halo,
    stellar disk, gaseous disk and bulge components.

    optional arguments:
      -h, --help    show this help message and exit
      --halo-core   Sets the density profile for the halo to have a core.
      --bulge-core  The same, but for the bulge.
      -cores CORES  The number of cores to use during the potential canculation.
                    Default is 1. Make sure this number is a factor of N_rho and
                    N_z.
      -temp TEMP    Initial gaseous disk temperature. Must be chosen wisely to
                    guarantee that the disk won't explode or collapse. Default is
                    8000.
      --force-yes   Don't ask if you want to use the existing potential_data.txt
                    file. Might be useful to run the script from another script.
      -o init.dat   The name of the output file.


A sample potential data file is provided for the parameters file included.


## Author

    Rafael Ruggiero
    Undergraduate student at Universidade de São Paulo (USP), Brazil
    Contact: bluewhale [at] cecm.usp.br
