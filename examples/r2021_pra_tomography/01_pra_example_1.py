'''
Example calculation used in:

    Smart, S. E., & Mazziotti, D. A. (2021). Lowering tomography costs in quantum simulation
    with a symmetry projected operator basis. Physical Review A, 103(1), 012420.
    https://doi.org/10.1103/PhysRevA.103.012420

In particular, we are comparing different mappings, the effect of different symmetry projections for different fermionic system sizes. Once we obtain these, we combine them as a graph problem and try to find smaller groupings of simultanesously measurable results. 

'''

from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.transforms import *
from hqca.acse import *
from pyscf import gto


# select the number of hydrogens (equal to number of spin orbitals)
Nh = [2,3,4,5,6,7,8,9,10]
for N in Nh:
    # generate the mol object and prepare the quantum storage object for the tomography class
    print('N Qubits Calculation: {}'.format(2*N))
    atoms = [['H',(2*x,0,0)] for x in range(N)]
    mol = gto.Mole()
    mol.atom=atoms
    mol.basis='sto-3g'
    mol.spin=N%2
    mol.verbose=0
    mol.build()
    ham = MolecularHamiltonian(
                    mol,
                    verbose=False,
                    generate_operators=False) #skips transformation of H, not required for solely generating the tomography
    st = StorageACSE(ham)
    qs = QuantumStorage(verbose=False)
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=N*2,
            provider='Aer')
    count = 0
    for T,m in zip([JordanWigner,Parity,BravyiKitaev],['jw','par','bk']):

        # for each qubit size and for each mapping, we look at using no symmetries,
        # N, and N+Sz symmetries 
        # verbose=True will give more detailed statistics on the coloring procedures

        Na = mol.nelec[0]%2
        Nab  = (mol.nelec[0]+mol.nelec[1])%2
        kwargs = {
                'real':True,
                'imag':False,
                'verbose':True,
                'strategies':['lf'], #largest first 
                'methods':['gt'],  # graph-tools 
                'transform':T,
                'simplify':True,
                        }
        print('--- --- --- --- --- --- --- ---')
        print('Default Tomography for {} Qubits, Mapping: {}'.format(2*N,m))
        tomoRe = StandardTomography(qs)
        tomoRe.generate(**kwargs)
        print('- --- --- --- --- --- --- --- -')
        print('Reduced Tomography for {} Qubits, Mapping: {}'.format(2*N,m))
        print('- constant N symmetry - ')
        tomoRe = ReducedTomography(qs)
        tomoRe.generate(**kwargs,skip_sz=True)
        print('- --- --- --- --- --- --- --- -')
        print('Reduced Tomography for {} Qubits, Mapping: {}'.format(2*N,m))
        print('- N and Sz symmetries ')
        tomoRe = ReducedTomography(qs)
        tomoRe.generate(**kwargs,skip_sz=False)

