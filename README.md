# Hybrid Quantum Classical Algorithms for Quantum Chemistry Computation
# v 0.2.x

## Introduction

This python module is a compilation of relatively simple programs developed along the course of my doctoral studies with Prof. David A. Mazziotti at the 
University of Chicago to run quantum chemistry calculations on a quantum computer. The focus is on approaches which incorporate reduced density matrix (RDM) theories, either simply in variationally modifying the 2-RDM, utilziing properties of RDMs (like the N-representability conditions) or in specific RDM methods, such as the quantum anti-Hermitian contracted Schroedinger Equation method (qACSE). 

While there is the potential for moderate simulations, say of 6-, 8- or 10- qubit simulations, the code is not suitable for large calculations (with the exception of generating relevant circuit sizes) and is instead optimized around practical calculations of smaller systems at a higher accuracy, and method development. The module utilizes qiskit for interacting and constructing quantum circuits, and interacting with the IBM backends, which can be accessed at the [IBM Quantum Experience page](https://quantum-computing.ibm.com/). Information on Qiskit can be found on [their web site](https://qiskit.org/).

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
opts (simple optimization program for different ACSE or VQE protocols involving classical optimization)
graph_tool >= 2.35
Maple 202x, with QuantumChemistry module for SDP purification
 
### Installing:
Nothing too important besides having python3 and the corresponding modules. Using the quantum
computer should be set up by yourself through the IBM Quantum Experience  

Note, qiskit-aer should be installed for access to the C++ qasm simulator, and
ibmq-provider should be obtained for running results on the actual quantum
computer. qiskit is inclusive of terra, aqua, and ibmq-provider, although the 
latter two are optional. 


### Operators and QuantumStrings 

The hqca module contains many useful tools for analyzing and handling basic quantum operations, which in general are centered on the Operator class (/hqca/tools/). An Operator can be initialized as an empty class, and then can hold certain types of strings, including QubitStrings, PauliStrings, or FermiStrings (creation and annihilation operators). Each string has a string.s and string.c attribute, indicating the string representation and coefficient.

'''
>>>from hqca.tools import *
>>>A = Operator()
>>>A+= PauliString('XX',0.5j)
>>>A+= PauliString('YY',+0.5j)
'''

The Operator class handles multiplication and addition as expected, and will return an Operator object. FermiStrings are slightly more complicated, and instead of forcing a normal ordered representation, while produce a string representation, using the anticommmutation relations. Note `p` and `h` represent the particle and hole operators. 

'''
A = Operator()
a+= FermiString(coeff=1,indices=[0,3,2,0],ops='++--',N=4)
print(a)
pi-+: -1
''' 


/hqca/tools holds the Operator class, which gives a way to describe quantum operators, and holds a number of strings. 

###

### Molecular Simulation

The varied examples in the /examples/ directory cover many different applications. In general, there are the 


### Examples and Tests 

Examples are included in the /examples/ directory. Tests are included in the /tests/ directory and can be run with the pytest module. From the main directory:

```
pytest tests
```



## References 

The software was utilized in various forms to obtain results listed in the publications below. In particular, some of the methods covered here are referenced and explained further in these articles, which cover varying aspects quantum simulation for quantum chemistry. While the earlier works could not be directly replicated, the ideas present in them are manifest in the current iteration, and should be replicated more easily with this design. 

Smart, S. E., Schuster, D. I., & Mazziotti, D. A. (2019). Experimental data from a quantum computer verifies the generalized Pauli exclusion principle. Communications Physics, 2(1). https://doi.org/10.1038/s42005-019-0110-3 

Smart, S. E., & Mazziotti, D. A. (2019). Quantum-classical hybrid algorithm using an error-mitigating <math> <mi>N</mi> </math> -representability condition to compute the Mott metal-insulator transition. Physical Review A, 100(2), 022517. https://doi.org/10.1103/PhysRevA.100.022517

Smart, S. E., & Mazziotti, D. A. (2020). Efficient two-electron ansatz for benchmarking quantum chemistry on a quantum computer, 023048, 1–8. https://doi.org/10.1103/PhysRevResearch.2.023048

Smart, S. E., & Mazziotti, D. A. (2020). Quantum Solver of Contracted Eigenvalue Equations for Scalable Molecular Simulations on Quantum Computing Devices, 60637(1), 1–6. Retrieved from http://arxiv.org/abs/2004.11416

Smart, S. E., & Mazziotti, D. A. (2020). Lowering Tomography Costs in Quantum Simulation with a Symmetry Projected Operator Basis, 1–15. Retrieved from http://arxiv.org/abs/2008.06027


# Authors

Scott E. Smart
David A. Mazziotti (advisor)

# License

Project licensed under the MIT License. See LICENSE.txt for more details. 

# Acknowledgements

A very big thank you to David Mazziotti, for supporting the work and myself through graduate school. Also would like to thank the IBMQ team for the support and development of open-access quantum computers, without which much of my graduate work from 2018 and onward would be very different in nature. 





