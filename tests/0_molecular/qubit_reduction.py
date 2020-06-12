from pyscf import gto,scf,mcscf
import sys
import numpy as np
from functools import reduce
from hqca.hamiltonian import *
from hqca.tools.fermions import *
from hqca.tools._stabilizer import *
from hqca.tools import *
from hqca.acse import *

#

mol = gto.Mole()
d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)],
        ['H',(d,d,0)],
        ['H',(0,d,0)],
        ]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol,
        #mapping='parity',kw_mapping={'MapSet':MapSet}
        )
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        #backend='qasm_simulator',
        Nq=8,
        provider='Aer')
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,simplify=False,method='gt',strategy='lf',
        #mapping='parity',MapSet=MapSet,
        )
print(tomoRe.op)
op = Operator()

for p in tomoRe.op:
    op+= PauliOperator(p,1)

par = ParityCheckMatrix(op,verbose=True)
U,Ut = par.get_transformation(qubits=[3,7])

Hp = (U*ham._qubOp)*Ut
print('Transformed Hamiltonian')
print(Hp)
Hr = Operator()
for op in Hp.op:
    p = op.p[0:3]+op.p[4:7]
    z1 = op.p[3]=='X'
    z2 = op.p[7]=='X'
    ph = (-1)**(z1+z2)
    Hr += PauliOperator(p,op.c*ph)
print('Trimmed Hamiltonian')
print(Hr)

print('Finding generators of new Hamiltonian...')
par2 = ParityCheckMatrix(Hr,verbose=True)
V,Vt = par2.get_transformation()

W =Operator()
Wt = Operator()
def expand(pOp,q):
    p =pOp.p[:q]+'I'+pOp.p[q:]
    return PauliOperator(p,pOp.c)

for v in V.op:
    for q in [0,2]:
        v = expand(v,q)
    W+= v
for v in Vt.op:
    for q in [0,2]:
        v = expand(v,q)
    Wt+= v

U = W*U
Ut = Ut*Wt
print(U)
print(Ut)


