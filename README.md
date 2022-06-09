# Hybrid Quantum Computing Algorithms for Quantum Chemistry

## Introduction

This python module is a compilation of tools for performing quantum chemistry simulations on near term quantum computers, with a focus on approaches based in reduced density matrix (RDM) theories, namely the contracted quantum eigensolvers. The methods are aimed at circuit based implementations, and can be readily translated to work on a variety of real device architectures. While there is the potential for moderate simulations, say 8-10 qubit simulations on local devices (with the exception of generating relevant tomographies), much larger simulations will require more memory and the program is only loosely optimized. Instead HQCA is centered around practical calculations of smaller molecular systems at a higher accuracy, and as a tool for method development. For instance, the code is written entirely in Python, and does not have many common quantum chemistry practices, including the use of spatial orbitals, universal adoption of symmetry adaptation, frozen core approximations, etc.  The module utilizes [Qiskit](https://qiskit.org) for interacting with, constructing, and running quantum circuits through the IBMQ backends, which can be accessed at the [IBM Quantum Experience page](https://quantum-computing.ibm.com/). 

## Features and Overview

The following features are included:
- Implementation of the quantum-ACSE as a contracted quantum eigensolver (CQE), with classical and quantum solutions of the ACSE condition
- Implementation of basic variational quantum eigensolvers (VQE), incldunig the ADAPT-VQE 
- Different tomography schemes of reduced density matrices with options for traditional or clique based grouping options 
- Symmetry projection of measurement operators for local qubit measurements
- Tapering of qubit symmetries to allow for qubit reduction schemes 
- Error mitigation techniques, mostly based in post processing RDMs 
- Simple toolkit for dealing with quantum operators, fermionic operators, transformations, and matrix representations 

## Getting Started 

### Installation

Package requires qiskit, pyscf, and networkx. For default installation, preferred method is to create a test environment, and run:

    pip install . 

For installation that allows local updating, run:

    pip install -e .


Additional modules include:

pytest, delayed_assert (for running test suite) 
graph_tool (conda installation recommended, needed for tomography grouping schemes)
Maple 2021 or greater, with QuantumChemistry module

### Operators and QuantumStrings 

The hqca module contains many useful tools for analyzing and handling basic quantum operations, which in general are centered on the Operator class (/hqca/operators/). An Operator can be initialized as an empty class, and then can hold certain types of strings, including QubitStrings, PauliStrings, or FermiStrings (creation and annihilation operators). Each string has a string.s and string.c attribute, indicating the string representation and coefficient.

```
>>> from hqca.operators import *
>>> A = Operator()
>>> A+= PauliString('XX',0.5j)
>>> A+= PauliString('YY',+0.5j)
```

The Operator class handles multiplication and addition as expected, and will return an Operator object. FermiStrings are slightly more complicated, and instead of forcing a normal ordered representation, it will produce a string representation, using the anticommmutation relations. Note `p` and `h` represent the particle and hole operators. 

```
>>> Af = Operator()
>>> Af+= FermiString(coeff=1,indices=[0,3,2,0],ops='++--',N=4)
>>> print(Af)
pi-+: -1
```

From /hqca/transforms/ one can find common transformations between these operators, including the Jordan-Wigner transformation, the Bravyi-Kitaev tranformation, the Parity mapping, and some Qubit mappings as well. 

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

### Conventions used 

Whenever possible, physics notation is used for RDM and integral ordering, as well as in higher RDMS. That is, the integral:

![Image](images/2k.png)
is stored as: $K[i,k,j,l]$. The integrals generated from pyscf are in chemist notation. 

A similar notation is present with reduced density matrices. The 3-RDM element defined by:

![Image](images/3rdm.png)
is stored as $D3[i,k,m,j,l,n]$.

The standard RDM class (which simply needs an alpha and beta set of indices to be specified), stores a k x k x k x k array (k^4 elements). 

The CompactRDM stores the RDM as a minimal vector, with excitations and their hermitian conjugates having unique indices. That is, we have all the elements of a complex RDM. For instance, the 2-RDM for a [2,2] system has:

(0 1 0 1) , (0 2 0 2), (0 3 0 3), (1 2 1 2), (1 3 1 3), (2 3 2 3), (0 2 1 2), (1 2 0 2), (0 3 1 3), (1 3 0 3), (0 2 0 3), (0 3 0 2), (1 2 1 3), (1 3 1 2), (0 2 1 3), (1 3 0 2) , (0 3 1 2) , (1 2 0 3)

The UniqueRDM class includes only the lower ordered pair, and not the hermitian conjugate. 

Norms in the CQE are by the Frobenius norm of the compact RDM, which is related to the norm of the standard RDM by a factor of 2. Note, operator norms (in the Pauli or fermionic basis) are not used, as these generally have to be renormalized by a factor depending on the size of the operator space. 


### Molecular Simulation

To perform a molecular simulation, a few objects are first required. /hqca/tests/_generic.py contains some basic objects which can be used as a guideline. 

Simulations do not require a molecule, though it is often used, but instead require a Hamiltonian object (/hqca/hamiltonian). These generate a matrix and operator form of the Hamiltonian, which is the same dimension as the appropriate RDM (either 1- or 2-RDM), or can be a qubit Hamiltonian as well.  

The Storage class, either StorageACSE or StorageVQE, builds off the Hamiltonian and stores and records certain aspects of the calculation. It handles energy evaluation, molecular properties and other parameters not related to the quantum computer. 

The QuantumStorage class on the other hand, relates properties of the quantum algorithm, and contains details related to performing an actual quantum simulation. Error mitigation methods are included here, as well as device specifications and options, the number of qubits, and other details pertaining to the quantum computer. 

Two separate classes, Instructions and Process, can be selected, and provide a way for the algorithm to communicate how the ansatz should be interpreted, and then once results are obtained, how to processes them. 

With all of these, one can construct a Tomography object, which will ascertain the scale of the problem, the type of tomography (1-/2-/3-RDM, real or imaginary, etc.), and the circuits required for the quantum computer. The tomography class uses the Instructions and Process to generate an RDM, which then is fed into the algorithm in question. 

All of these culminate in the QuantumRun class, which the RunACSE and RunVQE are built upon. RunACSE takes these inputs and will perform an ACSE calculation. 


Summary and Attributes:

1. Hamiltonian
    - matrix, qubit_operator, fermi_operator (optional)
    - may need a mol object from pyscf 
2. Storage (likely to be phase out)
    - evaluate, analysis, update
    - Contains molecular information and stores information on the run 
3. QuantumStorage
    - qs.set_backend, qs.set_algorithm, qs.set_noise_model, qs.set_error_mitigation
    - Contains information relevant to the quantum computation 
4. Instructions
    - Should not be instantiated, can be passed along to other objects
    - Ansatz or operator is fed into this, and then parsed into the language of quantum operations
    - All circuit simplifications or modifications are implemented here
5. Process
    - Default processor is usually okay, but if post correction or projection techniques are used, they would be included here. 
6. Tomography
    - set, generate, simulate, construct
    - Outputs an RDM object, actually interfaces with the quantum computer through qiskit QuantumStorage object
7. QuantumRun
    - Object for running a molecular simulation in order to find ground state energies. 

### Notes


### Examples and Tests 

Examples are included in the /examples/ directory. Tests are included in the /tests/ directory and can be run with the pytest module. From the main directory:

```
pytest tests
```


## References 

The software was utilized in various forms to obtain results in the below papers. In particular, some of the methods covered here are referenced and explained further in these articles, which cover varying aspects of quantum simulation for quantum chemistry. While the earlier works might not be exactly replicated with this code, as this project, qiskit, and the quantum devices themselves have all changed significantly, the ideas presented in them are accessible, and could be replicated with this module in a straightforward manner. When possible, problems tackled in these papers are provided as examples. 

### Quantum ACSE

Mazziotti, D. A. (2006). Anti-Hermitian Contracted Schrödinger Equation: Direct Determination of the Two-Electron Reduced Density Matrices of Many-Electron Molecules. Physical Review Letters 97, 143002. http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.143002 

Mazziotti, D. A. (2007). Anti-Hermitian part of the contracted Schrödinger equation for the direct calculation of two-electron reduced density matrices. Physical Review A - Atomic, Molecular, and Optical Physics, 75(2), 1–12. https://doi.org/10.1103/PhysRevA.75.022505

Smart, S. E., & Mazziotti, D. A. (2021). Quantum Solver of Contracted Eigenvalue Equations for Scalable Molecular Simulations on Quantum Computing Devices. Physical Review Letters, 126(7), 070504. https://doi.org/10.1103/PhysRevLett.126.070504

### Variational Quantum Eigensolvers

Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J. M., & Gambetta, J. M. (2017). Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. Nature, 549(7671), 242–246. https://doi.org/10.1038/nature23879

### Tomography

#### Symmetry Projected Tomography

Smart, S. E., & Mazziotti, D. A. (2020). Lowering Tomography Costs in Quantum Simulation with a Symmetry Projected Operator Basis, 1–15. Retrieved from http://arxiv.org/abs/2008.06027

#### Qubitwise-Commuting Tomography Grouping

Verteletskyi, V., Yen, T.-C., & Izmaylov, A. F. (2019). Measurement Optimization in the Variational Quantum Eigensolver Using a Minimum Clique Cover, 1–6. Retrieved from http://arxiv.org/abs/1907.03358

### Quantum Simulation & Tapering
 
Jordan, P., & Wigner, E. (1928). �ber das Paulische �quivalenzverbot. Zeitschrift F�r Physik, 47(9–10), 631–651. https://doi.org/10.1007/BF01331938

Bravyi, S., Gambetta, J. M., Mezzacapo, A., & Temme, K. (2017). Tapering off qubits to simulate fermionic Hamiltonians, (1), 1–15. Retrieved from http://arxiv.org/abs/1701.08213

Seeley, J. T., Richard, M. J., & Love, P. J. (2012). The Bravyi-Kitaev transformation for quantum computation of electronic structure. The Journal of Chemical Physics, 137(22), 224109. https://doi.org/10.1063/1.4768229

Bravyi, S., & Kitaev, A. (2000). Fermionic quantum computation. Journal of Chemical Physics, 137(22). https://doi.org/10.1006/aphy.2002.6254

### N-Representability, Purification

Foley, J. J., & Mazziotti, D. A. (2012). Measurement-driven reconstruction of many-particle quantum processes by semidefinite programming with application to photosynthetic light harvesting. Physical Review A, 86(1), 012512. https://doi.org/10.1103/PhysRevA.86.012512

Mazziotti, D. A. (2002). Purification of correlated reduced density matrices. Physical Review E, 65(2), 026704. https://doi.org/10.1103/PhysRevE.65.026704

Smart, S. E., & Mazziotti, D. A. (2019). Quantum-classical hybrid algorithm using an error-mitigating <math> <mi>N</mi> </math> -representability condition to compute the Mott metal-insulator transition. Physical Review A, 100(2), 022517. https://doi.org/10.1103/PhysRevA.100.022517

### Optimization Algorithms

Robinson, S. M. (2006). Numerical Optimization. Numerical Optimization. https://doi.org/10.1007/978-0-387-40065-5

Rakshit, P., & Konar, A. (2018). Principles in Noisy Optimization. Singapore: Springer Singapore. https://doi.org/10.1007/978-981-10-8642-7

### Published Literature 

Smart, S. E., Boyn, J.-N., & Mazziotti, D. A. (2021). Resolution of the Relative Energies of the Benzyne Isomers on a Quantum Computer Using a Contracted Schr\"odinger Equation. Retrieved from http://arxiv.org/abs/2103.06876

Smart, S. E., & Mazziotti, D. A. (2020). Quantum Solver of Contracted Eigenvalue Equations for Scalable Molecular Simulations on Quantum Computing Devices, 60637(1), 1–6. Retrieved from http://arxiv.org/abs/2004.11416

Smart, S. E., & Mazziotti, D. A. (2020). Lowering Tomography Costs in Quantum Simulation with a Symmetry Projected Operator Basis, 1–15. Retrieved from http://arxiv.org/abs/2008.06027

Smart, S. E., & Mazziotti, D. A. (2020). Efficient two-electron ansatz for benchmarking quantum chemistry on a quantum computer, 023048, 1–8. https://doi.org/10.1103/PhysRevResearch.2.023048 (no longer supported) 

Smart, S. E., & Mazziotti, D. A. (2019). Quantum-classical hybrid algorithm using an error-mitigating <math> <mi>N</mi> </math> -representability condition to compute the Mott metal-insulator transition. Physical Review A, 100(2), 022517. https://doi.org/10.1103/PhysRevA.100.022517 (no longer supported)

## Authors

Scott E. Smart, 
David A. Mazziotti

The views or content expressed here are solely of the authors and do not reflect on any policy or position of IBM or the IBM Q team.



# License

Project licensed under the Apache License. See LICENSE.txt for more details. 





