# Hybrid Quantum Classical Algorithm for Quantum Chemistry Computation

Program is a compilation of different programs and utilites I have developed
along the course of my doctoral studies under Prof. David A. Mazziotti at the 
University of Chicago to run different types of quantum chemistry calculations 
on a quantum computer. 

Primarily, the focus is on approaches which incorporate reduced density matrix
theory, either in the construction or error correction, and which only need to
measure or utilize the 1-electron RDM for constructing the wavefunction. In
terms of the quantum computer, the program utilizes the IBM Quantum Experience,
which provides cloud-based access to quantum devices. 

The project originated with verifying the generalized Pauli-constraints in a
3-qubit system, and then eventually we used the pinning effect in the 3-electron
system to perform quantum chemistry calculations in the reduced space.

Currently, the program is being used to develop more accurate, but generalized
calculations for quantum computers. For instance, different error mitigation
schemes have been implemented, and in general the Jordan-Wigner mapping is used
in a variety of contexts. 

The focus is still on how the GPC's can be used to improve accuracy of quantum
chemistry calculations, but there are several different applciations which have
been developed along the way. A folder of optimizers is also included, taken 
from different places (gradient-based or gradient-free, and some stochastic 
methods). 

Scanning for 

## Getting Started:

### Prerequisites:
python >= 3.6
qiskit >= 1.0.0
(with qiskit-terra and qiskit-aer)
pyscf (and prerequisite packs) >= 1.5.4
nevergrad >= 0.2.0
 
### Installing:
Nothing too important besides having python and the corresponding modules. Need
to configure the Qconfig, and or load your own configuration with IBMQ provider
from qiskit if you want to run actual tests. 

Note, qiskit-aer should be installed for access to the C++ qasm simulator, and
ibmq-provider should be obtained for running results on the actualy quantum
computer. qiskit is inclusive of terra, aqua, and ibmq-provider, although the 
latter two are optional. 

## Running the tests

Tests are located in the test directory, which has some simple examples for a
variety of different calculations that can be done. 

Two types of runs are included. The 'sp' and 'scan' options. The latter is
designed to generate a potential surface over the optimizing parameters. Can be
used for diagnostic purposes. The 'sp' is a single point run and is used for
evaluating a single point on a general potential energy surface. 

# Authors
Scott Smart

# License
Project licensed under the MIT License. See LICENSE.txt for more details. 

# Acknowledgement

Acknowledgements to David Mazziotti, for supporting the work. 





