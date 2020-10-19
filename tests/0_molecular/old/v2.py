'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''


from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
mol = gto.Mole()

d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
#mol.atom=[['H',(0,0,0)],['H',(d,0,0)],['H',(-d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol)
Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        #backend='qasm_simulator',
        Nq=4,
        provider='Aer')
tomoRe = StandardTomography(qs)
tomoIm = StandardTomography(qs)
tomoRe.generate(real=True,imag=False)
tomoIm.generate(real=False,imag=True)

acse = RunACSE(
        st,qs,Ins,
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.01,
        max_iter=100,
        initial_trust_region=0.1,
        newton_step=-1,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        restrict_S_size=1.0,
        verbose=False,
        commutative_ansatz=True,
        )
acse.build()
acse.run()
#print(acse.log_rdm)

