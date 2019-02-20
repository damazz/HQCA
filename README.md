# Hybrid Quantum Classical Algorithm for Quantum Chemistry Computation
Program which can perform a variety of quantum calculations for use in quantum
chemistry. Ideal for small molecules and small circuits which do not require
extensive fermionic mapping. 

Jordan-wigner mapping and some algorithms related to this are included. 

However, support for different types of single point energy runs with varying 
levels of entanglement and algorithms are availabl.e Single point energy is the 
default run type.

Also, some capabilities for confirming results with classical quantum
calculations are present as well in the natural orbital case. 

## Getting Started:

### Prerequisites:
python >= 3.5
QISKIT >= 0.7.1
IBMQuantumExperience >= 1.9.8
pyscf (and prerequisite packs) >= 1.5.6
numpy >= 1.14.1
 
### Installing:
Nothing too important besides having python and the corresponding modules. Need
to configure the Qconfig file if you want to run actual tests. 

Note, qiskit-aqua should be installed for access to the C++ qasm simulator. 
Simulations on the python simulator are much less efficient. 
## Running the tests


# Authors
Scott Smart

# License
Project licensed under the MIT License. See LICENSE.txt for more details. 

# Acknowledgement

Acknowledgements to David Mazziotti, for supporting the work. 





