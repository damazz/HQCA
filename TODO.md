 TODO/Changelog List

### 12/11/2019 
(note that the 1-qubit cases actually work :) )
* Clean up code and output
* Put together some test cases with different documentation
* Documentation
    * Start on qasm simulations, noise models, etc.
* Configure tomorgraphy inputs for different cases
    * allow for differnt tomography 


### 12/5/2019

* Implementing 2-qubit case
* Clean up output, 
* Allow for different initial states
    * ~~~Understanding paths, plotting trajectories~~~
* ~~~Find some model Hamiltonians~~~
* Clean up Ansatz treatment so that it generates Pauli operators
* Start to look for simple instructions...
* Also, error mitigation, look at doing variaitonal methods on IBM devices

### 11/29/2019 

* Where does init functions go? 

### 11/25/2019

* Adjust hamiltonian qubit operators so it is consistent
* Update ACSE ansatz
* Update ACSE method
* Start checking how things work 

### 11/23/2019 


* v 0.2.0
* ACSE works *whew!* 
* Implementing qubit ACSE 
* Implement Ansatz abstractclass
* Implement Instrutions abstractclass
* Work on linking everythnig together 
    * Update old ACSE method
    * Update VQE? 
* Clean up code
* Loose ends: 
    * Hamiltonian propagation


### 10/3/2019

* Still adjusting ACSE
    * Implement different depth ansatz 
    * Troubleshooting 3/6 case
    * ~~Get sequoia to work~~
    * Implement clique search
    * ~~Monitor convergence for 2/4 case, make sure everything is okay, then 2/6~~
        * ~~Figure out a way to plot~~
        * ~~Check convergence with respect to shots/counts~~
        * ~~Maybe implement unitarie2s~~



### 9/23/2019 

* Adjusting ACSE, almost doing an overhaul of sorts. Lots of equations and small
  details which don't work very well and are easily subject to variation. 
    * Need to figure out an implementation of the Hamiltonian in an efficient
      way
    * So, from pyscf we can get a reduction of Hamiltonian - we can even feed it
      the Fock matrix to get perturbed Hamiltonian
    * Need to rebuild the build function
    * Once we get the Hamiltonian, we need to deconstruct it. So.....hrm. 
    * What does ACSE storage need to have? ACSE quantum storage? 
    * Okay so we need to know what fermionic terms, and then to find at what
      level to implement them, i.e. if they can be simplifeid, etc. 
      So this is like the Entangler type. Defaults should be minimal, but that
      is experimental side. 
      Still, need to generate the pairs first, then simplify, and then implement
      Finding pairs - functions
      Simpliying - ansatz generation
      Dont forget, quantum storage deals with backend, error mitigation, and
      qubit transformations

### 9/9/2019

* Got a working system for the ACSE. Almost done with energy evaluation. 
* Need to make sure the RDM element ordering is consistent. I.e., using the
  RDM-< qubit key a little more often, make sure it works. Then, evaluate RDM,
  and perform the ACSE manuever. 

### 6/13/2019

* Need to write down test cases. What is the best way to do the 1-RDM tomography
  with low error (in a consistent way) 
    * Need to complete test cases
    * Currently we have:
        * full 1-RDM on off-diagonal measurements
        * full sign measurements (can we simplify?)
        * parametrized method with signs
        * test 
* how does exhibit quantum algorithm?
    * doesn't really, but pushes model to limit 
* Configure experiemtns for IBM device? 

### 5/31/2019

* ~~Add possible input circuit configuration~~
* ~~Determine what product of pauli matrices gives....when ancilla'ed~~
* ~~See if we can specify qubit/classical registers, i.e. swaps  ~~

### 5/30/2019

* ~~Configure post correction~~
* Configure correctly for IBM device
* Error Mitigation
    * Document the error correction types a little better
* ~~Add in ancilla for sign? ~~
* Add statistics to processing of counts

### 5/24/2019

* Backend/Transpiler
    * ~~Create file which takes in QASM and performs transpilation~~
    * ~~Figure out which passes are supported by PassManager for parameters~~
        * ~~Read up on transpile/passmanager functionality~~
    * ~~Figure out a way to get a consistent transpilation (maybe slack)~~
* Error Mitigation
    * ~~Development of well-suited error mitigation in the algorithm~~
    * Continue to think of other ways to approach error mitigation, 
    * ~~Do some preliminary testing, configure so that it is realtively easy to do~~
* Revisions
    * ~~Try and fix todo file,~~
    * Edit and clean up print statements throughout program, edit output
