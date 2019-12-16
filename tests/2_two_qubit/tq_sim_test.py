from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
import numpy as np
import sys
mol = gto.Mole()
d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol)

maps  = {0:0,1:0,2:1,3:1}
qubit = {0:0,1:1,2:0,3:1}

ham = TwoQubitHamiltonian(sq=True,
        fermi=True,
        en_c = ham._en_c,
        imag=False,
        real=True,
        ferOp=ham.fermi_operator,
        mapOrb=maps,
        mapQub=qubit,
        )

Ins = RestrictiveSet
#Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        #backend='statevector_simulator',
        #backend='ibmq_burlington',
        backend='qasm_simulator',
        num_shots=8192,
        Nq=2,
        provider='Aer',
        #provider='IBMQ',
        backend_initial_layout=[0,1],
        )

tomoRe = StandardTomography(qs)
tomoIm = StandardTomography(qs)
tomoRe.generate(real=True,imag=False)
tomoIm.generate(real=False,imag=True)

tomoIm.op.remove('YZ')
tomoIm.op.remove('ZY')
tomoRe.op.remove('XZ')
tomoRe.op.remove('ZX')
tomoRe.mapping['IZ']='ZZ'
tomoRe.mapping['ZI']='ZZ'
tomoRe.mapping['XI']='XZ'
tomoRe.mapping['IX']='ZX'
tomoRe.mapping['IY']='ZY'
tomoRe.mapping['YI']='YZ'
#print(tomoRe.mapping)
#print(tomoIm.mapping)
#print(tomoIm.op)
print(ham.qubit_operator)
acse = RunACSE(
        st,qs,Ins,
        #method='euler',
        method='newton',
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.25,
        commutative_ansatz=True,
        quantS_thresh_rel=0.5,
        max_iter=15,
        initial_trust_region=0.25,
        newton_step=-1,
        restrict_S_size=0.5,
        opt_thresh=1e-6,
        tr_taylor_criteria=1e-6,
        verbose=True,
        tr_objective_criteria=1e-6,
        tomo_S = tomoIm,
        tomo_Psi=tomoRe,
        )
acse.build()
acse.run()
print(acse.log_E)
print(acse.log_S)



