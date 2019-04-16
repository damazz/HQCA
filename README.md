# Hybrid Quantum Classical Algorithm for Quantum Chemistry Computation
Program which can perform a variety of quantum calculations for use in quantum
chemistry. Ideal for small molecules and small circuits which do not require
extensive fermionic mapping. Likely 8 qubits (or 4 spin orbitals) is maximum
suggested simulation, although the architecture can support larger runs on the
actual quantum computer. `Cheap' simlations on more than 8 qubits likely would
have too much error on the quantum computer however. 

Jordan-wigner mapping and some algorithms related to this are included. Notably,
generalizable entanglement schemes are included. 

## Getting Started:

### Prerequisites:
python >= 3.6
qiskit >= 0.8.0
pyscf (and prerequisite packs) >= 1.5.4
nevergrad >= 0.2.0
 
### Installing:
Nothing too important besides having python and the corresponding modules. Need
to configure the Qconfig, and or load your own configuration with IBMQ provider
from qiskit if you want to run actual tests. 

Note, qiskit-aqua should be installed for access to the C++ qasm simulator, and
ibmq-provider should be obtained for running results on the actualy quantum
computer. qiskit is inclusive of terra, aqua, and ibmq-provider, although the 
latter two are optional. 

## Running the tests

Tests are located in the examples directory, which has some simple diagnostic
tests for looking at noise, and isolating different 

Two types of runs are included. The `sp' and `scan' options. The latter is
designed to generate a potential surface over the optimizing parameters. Can be
used for diagnostic purposes. The `sp' is a single point run and is used for
evaluating a single point on a general potential energy surface. 

# Authors
Scott Smart

# License
Project licensed under the MIT License. See LICENSE.txt for more details. 

# Acknowledgement

Acknowledgements to David Mazziotti, for supporting the work. 





