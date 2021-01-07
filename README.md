# Hybrid Quantum Classical Algorithms for Quantum Chemistry Computation
# v 0.2.x

## Introduction

This python module is a compilation of relatively simple programs developed along the course of my doctoral studies with Prof. David A. Mazziotti at the 
University of Chicago to run quantum chemistry calculations on a quantum computer. The focus is on approaches which incorporate reduced density matrix (RDM) theories, either simply in variationally modifying the 2-RDM, or in specific RDM methods, such as the quantum anti-Hermitian contracted Schroedinger Equation method (qACSE). 

Additionally, while there is the potential for moderate simulations, say of 6-, 8- or 10- qubit simulations, the code is not suitable for large calculations (with the exception of generating lists of required tomographies) and is instead optimized around practical calculations of smaller systems at a high accuracy, and for method development. The module utilizes qiskit for interacting and constructing quantum circuits, and interacting with the IBM backends, which can be accessed at the [IBM Quantum Experience page](https://quantum-computing.ibm.com/). Information on Qiskit can be found on [their web site](https://qiskit.org/).

## Features  

The following features are included:
- Implementation of the quantum-ACSE as a quantum eigensolver, with classical and quantum solutions of the ACSE condition
- Implentation of basic variational quantum eigensovlers (VQE)
- Allowal of programmable ansatz 
- Different tomography schemes of reduced density matrices with options for traditional or clique based grouping options 
- Symmetry projection of measurement operators for local qubit measurements
- Tapering of transformations to allow for qubit reduction schemes 
- A couple of error mitigation techniques, mostly based in post processing RDMs 
- General tools for dealing with quantum operators, fermionic operators, transformations, and matrix representations 

## Examples and Tests 

Examples are included in the /examples/ directory. Tests are included in the /tests/ directory and can be run with the pytest module. From the main directory:

```
pytest tests
```


## Getting Started 

### Prerequisites 
python >= 3.7
qiskit >= 0.15.1
pyscf (and prerequisite packs) >= 1.7.4

Optionally:
optss (simple optimization program for different ACSE or VQE optimizations)
graph_tool >= 2.35
Maple 20xx, with QuantumChemistry module for SDP purification
 
### Installing:
Nothing too important besides having python3 and the corresponding modules. Using the quantum
computer should be set up by yourself through the IBM Quantum Experience  

Note, qiskit-aer should be installed for access to the C++ qasm simulator, and
ibmq-provider should be obtained for running results on the actual quantum
computer. qiskit is inclusive of terra, aqua, and ibmq-provider, although the 
latter two are optional. 

### Running tests

Tests are located in the test directory, which has some simple examples for a
variety of different calculations that can be done. 


## References 

Smart, S. E., Schuster, D. I., & Mazziotti, D. A. (2019). Experimental data from a quantum computer verifies the generalized Pauli exclusion principle. Communications Physics, 2(1). https://doi.org/10.1038/s42005-019-0110-3 

Smart, S. E., & Mazziotti, D. A. (2019). Quantum-classical hybrid algorithm using an error-mitigating <math> <mi>N</mi> </math> -representability condition to compute the Mott metal-insulator transition. Physical Review A, 100(2), 022517. https://doi.org/10.1103/PhysRevA.100.022517

Smart, S. E., & Mazziotti, D. A. (2020). Efficient two-electron ansatz for benchmarking quantum chemistry on a quantum computer, 023048, 1–8. https://doi.org/10.1103/PhysRevResearch.2.023048

Smart, S. E., & Mazziotti, D. A. (2020). Quantum Solver of Contracted Eigenvalue Equations for Scalable Molecular Simulations on Quantum Computing Devices, 60637(1), 1–6. Retrieved from http://arxiv.org/abs/2004.11416

Smart, S. E., & Mazziotti, D. A. (2020). Lowering Tomography Costs in Quantum Simulation with a Symmetry Projected Operator Basis, 1–15. Retrieved from http://arxiv.org/abs/2008.06027


# Authors

Scott E. Smart
David A. Mazziotti

# License

Project licensed under the MIT License. See LICENSE.txt for more details. 

# Acknowledgements

A very big thank you to David Mazziotti, for supporting the work and myself through
graduate school. Additionally to IBMQ for the support and development of open-access 
quantum computers, without which the extent of the current work would be impossible. 





