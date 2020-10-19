from pyscf import gto,scf,mcscf
import sys
import numpy as np
from functools import reduce
from hqca.hamiltonian import *
from hqca.tools.fermions import *
from hqca.instructions import *
from hqca.tools._stabilizer import *
from hqca.tools import *
from hqca.acse import *
from qubit_reduction import U,Ut

#

mol = gto.Mole()
d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)],
        #['H',(d,d,0)],
        #['H',(0,d,0)],
        ]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
MapSet = JordanWignerSet(4,
        qubits=[2,1,0],
        paulis=['X','X','X'],
        eigvals=[-1,+1,-1],
        reduced=True,
        U=U,
        Ut=Ut,
        )
ham = MolecularHamiltonian(mol,
        mapping='jw',kw_mapping={'MapSet':MapSet}
        )


Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=1,
        provider='Aer')
tomoRe = StandardTomography(qs)
tomoIm = StandardTomography(qs)
tomoRe.generate(real=True,imag=False,simplify=False,method='gt',strategy='lf',
        mapping='jw',MapSet=MapSet,
        )
tomoIm.generate(real=False,imag=True,simplify=False,method='gt',strategy='lf',
        mapping='jw',MapSet=MapSet,
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
