 TODO/Changelog List

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
