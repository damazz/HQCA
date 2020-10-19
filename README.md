# Hybrid Quantum Classical Algorithms for Quantum Chemistry Computation
# v 0.2.1
# updated 10-05-2020
# hqca

This python module program is a compilation of relatively simple programs 
developed along the course of my doctoral studies under Prof. David A. Mazziotti at the 
University of Chicago to run different types of quantum chemistry calculations 
on a quantum computer.

Primarily, the focus is on approaches which incorporate reduced density matrix
theory, either in the construction or error mitigation, and which only need to
measure or utilize reduced system. Additionally, while there is the potential
for moderate simulations, say of 6-, 8- or 10- qubit simulations, the code is
not well optimized for large calculations, and is instead optimized around
practical calculations of smaller systems at a high accuracy, and for method
development. The programs are intended for use with the IBMQ systems. 

Additionally, the following theoretical ideas are included at some level:
- Implementation of the quantum-ACSE method
- Implentation of basic variational quantum eigensovlers (VQE)
- Diffferent tomography schemes with options for grouping by cliques
- Tapering qubit Hamiltonians and locating different symmetries
- Construction of parity check matrices
- Handling and construction of RDMs 
- A couple of error mitigation techniques, mostly based in post processing RDMs 

The test examples generally provide the suitable range of applications. 


The generic program includes the following classes, included in the core
module:
- QuantumRun
- Hamiltonian
- Instructions
- Circuit
- Storage
- Tomography

Additionally, for interfacing with quantum devices, QuantumStorage in hqca/tools 
is needed. 

A typical method might be constructed in this manner with the help of the 

-- -- -- -- -- -- -- --
Hamiltonian
Storage(Hamiltonian)
QuantumStorage(Storage)
Instructions
Tomography(QuantumStorage,Instructions)
QuantumRun(Tomography)
-- -- -- -- -- -- -- --


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

Custom circuits can also be deisgned, although generally there will be a mapping function
which will take suitable input and then map it to a potential circuit. 

Tomography:
Handles the tomography, execution, constructions of a quantum system. Has
important functions:
1) set - 'sets' the problem, generating elements needed for tomography
2) generate - generates actual tomography circutis based on set
3) simulate - executes the circuits
4) construct - constructs the RDMs 


QuantumStorage: 
Contains information relevant to running things on a quantum computer. Typically, 
one sets the algorithm and then can set backend and additional parameters. Types of 
error mitigation are passed in through here. 
-- -- -- -- -- -- -- --

Operators and Transforms:
Operators are composite objects, composed of different strings. These strings are typically
either PauliStrings, QubitString, or FermiStrings. Qubit and Fermi Strings are denoted in
second quantized notation, and PauliStrings are denoted in the basis of quantum operators.

To go from a fermionic operator to a qubit operator, one uses the Transforms folder, which
has fermionic transformations and others. Transforms can function recursively as 
well which is desirable for qubit reduction techniques where we taper off qubits iteratively. 

To use a transform T, the operator.transform(T) functions is used, which will return a 
new operator in the requisite string basis. These can then be used for other work as well.


Operator:


## Getting Started:

### Prerequisites:
python >= 3.7
qiskit >= 0.15.1
pyscf (and prerequisite packs) >= 1.7.4


Optionally:
optss (simple optimization program for different ACSE or VQE optimizations)
graph_tool >= 2.35
maple, with QuantumChemistry module for SDP purification
 
### Installing:
Nothing too important besides having python and the corresponding modules. Using the quantum
computer should be set up by yourself. 

Note, qiskit-aer should be installed for access to the C++ qasm simulator, and
ibmq-provider should be obtained for running results on the actual quantum
computer. qiskit is inclusive of terra, aqua, and ibmq-provider, although the 
latter two are optional. 

## Running the tests

Tests are located in the test directory, which has some simple examples for a
variety of different calculations that can be done. 

# Authors
Scott Smart

# License
Project licensed under the MIT License. See LICENSE.txt for more details. 

# Acknowledgements

A very big thank you to David Mazziotti, for supporting the work and myself through
graduate school. Additionally to IBMQ for the support and development of open-access 
quantum computers, without which the extent of the current work would be impossible. 





