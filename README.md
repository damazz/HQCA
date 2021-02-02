# Hybrid Quantum Classical Algorithms for Quantum Chemistry
### v0.2.x

## Introduction

This python module is a compilation of tools developed by Scott Smart and David Mazziotti focusing on performing quantum chemistry simulations on near term quantum computers, and mainly focuses on approaches which based in reduced density matrix (RDM) theories. These include simply variationally modifying the 2-RDM, utilizing properties of RDMs (like the N-representability conditions) or in specific RDM methods, such as the quantum anti-Hermitian Contracted Schroedinger Equation method (qACSE). 

While there is the potential for moderate simulations, say of 6-, 8- or 10- qubit simulations, the module is not suitable for things much larger than that (with the exception of generating relevant tomographies) and is instead optimized around practical calculations of smaller molecular systems at a higher accuracy, and as a tool for method development. The module utilizes [Qiskit](https://qiskit.org) for interacting with, constructing, and running quantum circuits through the IBMQ backends, which can be accessed at the [IBM Quantum Experience page](https://quantum-computing.ibm.com/). The views or content expressed here are solely of the authors and do not reflect on any policy or position of IBM or the IBM Q team.

## Features and Overview

The following features are included:
- Implementation of the quantum-ACSE as a quantum eigensolver, with classical and quantum solutions of the ACSE condition
- Implentation of basic variational quantum eigensovlers (VQE)
- Allowal of programmable ansatz 
- Different tomography schemes of reduced density matrices with options for traditional or clique based grouping options 
- Symmetry projection of measurement operators for local qubit measurements
- Tapering of transformations to allow for qubit reduction schemes 
- A couple of error mitigation techniques, mostly based in post processing RDMs 
- General tools for dealing with quantum operators, fermionic operators, transformations, and matrix representations 


## Getting Started 

### Prerequisites 
python >= 3.7
qiskit >= 0.15.1
pyscf (and prerequisite packs) >= 1.7.4

Optionally:
graph_tool >= 2.35
Maple 202x, with QuantumChemistry module for SDP purification (not yet implemented)
 
### Installing:
Nothing too important besides having python3 and the corresponding modules. Using the quantum
computer should be set up by yourself through the IBM Quantum Experience  

qiskit-aer should be installed for access to the C++ qasm simulator, and
ibmq-provider should be obtained for running results on the actual quantum
computer. qiskit is inclusive of terra, aqua, and ibmq-provider, although the 
latter two are optional. 


### Operators and QuantumStrings 

The hqca module contains many useful tools for analyzing and handling basic quantum operations, which in general are centered on the Operator class (/hqca/tools/). An Operator can be initialized as an empty class, and then can hold certain types of strings, including QubitStrings, PauliStrings, or FermiStrings (creation and annihilation operators). Each string has a string.s and string.c attribute, indicating the string representation and coefficient.

```
>>> from hqca.tools import *
>>> A = Operator()
>>> A+= PauliString('XX',0.5j)
>>> A+= PauliString('YY',+0.5j)
```

The Operator class handles multiplication and addition as expected, and will return an Operator object. FermiStrings are slightly more complicated, and instead of forcing a normal ordered representation, while produce a string representation, using the anticommmutation relations. Note `p` and `h` represent the particle and hole operators. 

```
>>> Af = Operator()
>>> Af+= FermiString(coeff=1,indices=[0,3,2,0],ops='++--',N=4)
>>> print(Af)
pi-+: -1
```

From /hqca/transforms/ one can find common transformations between these operators, including the Jordan-Wigner transformation, the Bravyi-Kitaev tranformation, the Parity mapping, and some Qubit mappings as well. /hqca/quantum_tools/ serves as a tool for exploring common aspects of quantum computation and quantum chemistry. 

```
>>> from hqca.transforms import *
>>> print(Af.transform(JordanWigner))
IIYX: +0.12500000j
ZIYX: -0.12500000j
IIXX: 0.12500000
ZIXX: -0.12500000
IIYY: 0.12500000
ZIYY: -0.12500000
IIXY: -0.12500000j
ZIXY: +0.12500000j
```


### Molecular Simulation

To perform a molecualr simulation, a few objects are first required. /hqca/tests/_generic.py contains some basic objects which can be used as a guideline. 

Simulations do not require a molecule, though it is often used, but instead require a Hamiltonian object (/hqca/hamiltonian). These generate a matrix and operator form of the Hamiltonian, which is the same dimension as the appropriate RDM (either 1- or 2-RDM), or can be a qubit Hamiltonian as well.  

The Storage class, either StorageACSE or StorageVQE, builds off the Hamiltonian and stores and records certain aspects of the calculation. It handles energy evaluation, molecluar proerties and other parameters not related to the quantum computer. 

The QuantumStorage class on the other hand, relates properties of the quantum algorithm, and contains details related to performing an actual quantum simulation. Error mitigation methods are included here, as well as device specifications and options, the number of qubits, and other details pertaining to the quantum computer. 

Two seperate classes, Instructions and Process, can be selected, and provide a why for the algorithm to communicate how the ansatz should be interpreted, and then once results are obtained, how to processes them. 

With all of these, one can construct a Tomography object, which will ascertain the scale of the problem, the type of tomography (1-/2-/3-RDM, real or imaginary, etc.), and the circuits required for the quantum computer. The tomography class uses the Instructions and Process to generate an RDM, which then is fed into the algorithm in question to 

All of these culminate in the QuantumRun class, of which there currently are the RunACSE and RunVQE are built upon. RunACSE takes these inputs and will perform an ACSE calculation. 


Summary and Attributes:

1. Hamiltonian, H
    - H.matrix, H.qubit_operator, H.fermi_operator (optional)
    - may need a mol object from pyscf 
2. Storage, S
    - S.evaluate, S.analysis, S.update
    - Contains molecular information and stores information on the run 
3. QuantumStorage, qs
    - qs.set_backend, qs.set_algorithm, qs.set_noise_model, qs.set_error_mitigation
    - Contains information relevant to the quantum computation 
4. Instructions, I
    - Should not be instantiated, can be passed along to other objects
    - Ansatz or operator is fed into this, and then parsed into the language of quantum operations
    - All circuit simplifications or modifications are implemented here
5. Process
    - Default processor is usually okay, but if post correction or projection techniques are used, they would be included here. 
6. Tomography, T
    - T.set, T.generate, T.simulate, T.construct
    - Outputs an RDM object, actually interfaces with the quantum computer with the QuantumStorage object
7. QuantumRun
    - Object for running a molecular simulation in order to find ground state energies. 

### Examples and Tests 

Examples are included in the /examples/ directory. Tests are included in the /tests/ directory and can be run with the pytest module. From the main directory:

```
pytest tests
```


## References 

The software was utilized in various forms to obtain results listed in the publications below. In particular, some of the methods covered here are referenced and explained further in these articles, which cover varying aspects quantum simulation for quantum chemistry. While the earlier works might not be exactly replicated with this code, as this project, qiskit, and the quantum devices themselves have all changed significantly, the ideas present in them are accesible, and could be replicated with this module in a straightforward manner. When possible, these are provided as examples. 

Smart, S. E., Schuster, D. I., & Mazziotti, D. A. (2019). Experimental data from a quantum computer verifies the generalized Pauli exclusion principle. Communications Physics, 2(1). https://doi.org/10.1038/s42005-019-0110-3 

Smart, S. E., & Mazziotti, D. A. (2019). Quantum-classical hybrid algorithm using an error-mitigating <math> <mi>N</mi> </math> -representability condition to compute the Mott metal-insulator transition. Physical Review A, 100(2), 022517. https://doi.org/10.1103/PhysRevA.100.022517

Smart, S. E., & Mazziotti, D. A. (2020). Efficient two-electron ansatz for benchmarking quantum chemistry on a quantum computer, 023048, 1–8. https://doi.org/10.1103/PhysRevResearch.2.023048

Smart, S. E., & Mazziotti, D. A. (2020). Quantum Solver of Contracted Eigenvalue Equations for Scalable Molecular Simulations on Quantum Computing Devices, 60637(1), 1–6. Retrieved from http://arxiv.org/abs/2004.11416

Smart, S. E., & Mazziotti, D. A. (2020). Lowering Tomography Costs in Quantum Simulation with a Symmetry Projected Operator Basis, 1–15. Retrieved from http://arxiv.org/abs/2008.06027


# Authors

Scott E. Smart
David A. Mazziotti (advisor)

# License

Project licensed under the Apache 2.0 License. See LICENSE.txt for more details. 

# Acknowledgements

A very big thank you to David Mazziotti, for supporting the work and myself through graduate school. Also would like to thank the IBMQ team for the support and development of open-access quantum computers, without which much of my graduate work from 2018 and onward would be very different in nature. 





