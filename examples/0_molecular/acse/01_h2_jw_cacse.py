'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
import sys
sys.path.append('/home/scott/Documents/research/software/HQCA/')

from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
mol = gto.Mole()

d = 0.5
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
#
T = JordanWigner
ham = MolecularHamiltonian(mol,transform=T)
Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=4,
        provider='Aer')
tomoRe = StandardTomography(qs)
tomoIm = StandardTomography(qs)
tomoRe.generate(real=True,imag=False,simplify=False,transform=T,)
tomoIm.generate(real=False,imag=True,simplify=False,transform=T,)

#
acse = RunACSE(
        st,qs,Ins,
        method='newton',
        update='classical',
        opt_thresh=1e-10,
        trotter=1,
        ansatz_depth=1,
        quantS_thresh_rel=1e-6,
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.01,
        max_iter=5,
        initial_trust_region=0.1,
        newton_step=-1,
        restrict_S_size=0.5,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        verbose=False,
        )
acse.build()
acse.run()
#print(acse.log_rdm)

