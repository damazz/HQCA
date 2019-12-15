# Hybrid Quantum Classical Algorithm for Quantum Chemistry Computation
# v 2.0 
# hqca

In summary, this python module program is a compilation of relatively simple programs 
developed along the course of my doctoral studies under Prof. David A. Mazziotti at the 
University of Chicago to run different types of quantum chemistry calculations 
on a quantum computer.

Primarily, the focus is on approaches which incorporate reduced density matrix
theory, either in the construction or error correction, and which only need to
measure or utilize reduced system. Additionally, while there is the potential
for moderate simulations, say of 6-, 8- or 10- qubit simulations, the code is
not well optimized for large calculations, and is instead optimized around
practical calculations of smaller systems at a high accuracy, and for method
development. As such, many features which worked at one time, either have not
been maintained or simply have morphed and changed into newer features.

The generic program includes the following structures, included in the core
module:
- QuantumRun
- Hamiltonian
- Instructions
- Circuit
- Storage
- Tomography

A typical method might be constructed in this manner, with dependencies
indicated by ->, and storage indicated by -+->. I.e., y -+-> x implies that the
information in y is contained in x, an agglomerate variable. 

-- -- -- -- -- -- -- --
Hamiltonian -> Storage -+-> x
Instructions -+-> x
Tomography -+-> x

A necessary tool, though not in core, is the quantum storage.
QuantumStorage -+-> x

Then, all of x can be used in some run:
x -> QuantumRun
-- -- -- -- -- -- -- --
Note, other tools are included by importing hqca.tools. A summary of the above
core entities is given below: 

Hamiltonian:
Contains two or three attributes, chiefly:
1) ham.matrix - for evaluation, necessary
2) ham.qubit_operator - for propagation, recommended
3) ham.fermi_operator - for propagation, recommended

Storage:
Handles properties of a run, storing variables, and has the important functions:
1) evaluate
2) analysis

Instructions:
Dictates the method used to processes output of a run into quantum gates to be
used on the quantum computer. GenericPauli provides a simple scheme, but for
actual runs better compiled circuits with shorter lengths should be used. 

Tomography:
Handles the tomography, execution, constructions of a quantum system. Has
important functions:
1) set - 'sets' the problem, generating elements needed for tomography
2) generate - generates actual tomography circutis based on set
3) simulate - executes the circuits
4) construct - constructs the RDMs 



Instructions 


## Getting Started:

### Prerequisites:
python >= 3.7
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


# Authors
Scott Smart

# License
Project licensed under the MIT License. See LICENSE.txt for more details. 

# Acknowledgement

Acknowledgements to David Mazziotti, for supporting the work. 





