from pyscf import gto,scf,mcscf
import sys
import numpy as np
from functools import reduce, partial
from hqca.hamiltonian import *
from hqca.tools import *
from hqca.acse import *
from hqca.instructions import *
from hqca.transforms import *

#

mol = gto.Mole()
d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)],
        ]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
tr = JordanWigner
ham = MolecularHamiltonian(mol,
        transform=tr,
        )
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=4,
        provider='Aer')
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,
        simplify=False,method='gt',
        strategy='lf',transform=tr
        )
op = Operator()
for p in tomoRe.op:
    op+= PauliString(p,1)

par = ParityCheckMatrix(op,verbose=True)
#U,Ut = par.get_transformation(qubits=[1,3])

U1 = Operator()
x3 = 'IIIX'
x1 = 'IXI'
z3 = 'IIZZ'
z1 = 'ZZI'
c = 1/np.sqrt(2)
U1+= PauliString(x3,c)
U1+= PauliString(z3,c)
U2 = Operator()
U2+= PauliString(x1,c)
U2+= PauliString(z1,c)

# # 
# 
# now, transforming H to get new ones 
# 
# #

def modify(ops,
        fermi,
        U,Ut,
        qubits,
        paulis,
        eigvals):
    new = fermi(ops)
    new = change_basis(new,U,Ut)
    new = trim_operator(new,
            qubits=qubits,
            paulis=paulis,
            eigvals=eigvals)
    return new



tr = partial(
        modify,
        fermi=JordanWigner,
        U=U1,Ut=U1,
        qubits=[3],
        paulis=['X'],
        eigvals=[-1],
        )
tr1 = partial(
        modify,
        fermi=tr,
        U=U2,Ut=U2,
        qubits=[1],
        paulis=['X'],
        eigvals=[-1],
        )
print('Modifed Hamiltonian')
ham = MolecularHamiltonian(mol,
        transform=tr1,
        )

par2 = ParityCheckMatrix(ham._qubOp,verbose=True)
V,Vt = par2.get_transformation()


tr2 = partial(modify,
        fermi=tr1,
        U=V,Ut=Vt,
        qubits=[0],
        paulis=['X'],
        eigvals=[+1],
        )

ham = MolecularHamiltonian(mol,
        transform=tr2,
        )

#Ins = SingleQubitExponential
Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=1,
        provider='Aer')

tomoRe = ReducedTomography(qs)
tomoIm = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,simplify=True,method='gt',strategy='lf',
        transform=tr2,
        )
tomoIm.generate(real=False,imag=True,simplify=True,method='gt',strategy='lf',
        transform=tr2
        )
acse = RunACSE(
        st,qs,Ins,
        method='newton',
        update='quantum',
        opt_thresh=1e-10,
        trotter=1,
        ansatz_depth=1,
        quantS_thresh_rel=1e-6,
        propagation='trotter',
        use_trust_region=True,
        convergence_type='trust',
        commutative_ansatz=False,
        hamiltonian_step_size=0.01,
        max_iter=100,
        initial_trust_region=0.1,
        newton_step=-1,
        restrict_S_size=0.5,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        verbose=True,
        )
acse.build()
acse.run()
