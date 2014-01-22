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

    python galaxy.py [OUTPUT]

## Author

    Rafael Ruggiero
    Undergraduate student at Universidade de São Paulo (USP), Brazil
    Contact: bluewhale [at] cecm.usp.br
