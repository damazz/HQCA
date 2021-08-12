from copy import deepcopy as copy
import sys
from functools import partial
import numpy as np
from hqca.operators import *



def trim_operator(ops,
        qubits,
        paulis,
        eigvals,
        null=0,
        ):
    new = Operator()
    if not qubits==sorted(qubits)[::-1]:
        sys.exit('Reorder your trimming operations to ensure qubit ordering.')
    for op in ops:
        s,c = op.s,op.c
        for q,p,e in zip(qubits,paulis,eigvals):
            if s[q]==p:
                c*= e
            elif s[q]=='I':
                pass
            else:
                c*=null
            s = s[:q]+s[q+1:]
        new+= PauliString(s,c,symbolic=op.sym)
    return new

def change_basis(op,
        U,
        Ut=None,
        **kw):
    if type(Ut)==type(None):
        Ut = copy(U)
    return (U*op)*Ut

def modify(ops,
        fermi,
        U,Ut,
        qubits,
        paulis,
        eigvals,
        initial=False):
    T = fermi(ops,initial=initial)
    # initialize
    T = change_basis(T,U,Ut)
    # apply Pauli change of basis
    T = trim_operator(T,
            qubits=qubits,
            paulis=paulis,
            null=int(initial),
            eigvals=eigvals)
    # perform trimming
    return T

def clifford(ops,
        fermi,
        U,
        **kw
        ):
    new = fermi(ops,**kw)
    return new.clifford(U)

def get_transform_from_symmetries(
        Transform,
        symmetries,
        qubits,
        eigvals,
        ):
    cTr = copy(Transform)
    for i in range(len(symmetries)):
        x = 'I'*len(symmetries[i])
        ind = qubits[i]
        x = x[:ind] + 'X' + x[ind+1:]
        op = Operator([
            PauliString(symmetries[i],1/np.sqrt(2)),
            PauliString(x,1/np.sqrt(2))
                ])
        cTr = partial(
                modify,
                fermi=copy(cTr),
                U=op,Ut=op,
                qubits=[qubits[i]],
                eigvals=[eigvals[i]],
                paulis=['X'],
                )
    ciTr = partial(cTr,initial=True)
    return cTr, ciTr

def parity_free(Na,Nb,paritya,parityb,transform):
    Z1 = 'Z'*(Na+Nb)
    Z2 = 'Z'*Na + 'I'*(Nb-1)
    Tr,iTr = get_transform_from_symmetries(
            transform,
            [Z1,Z2],
            [Na+Nb-1,Na-1],
            [paritya,parityb])
    return Tr,iTr

'''
def find_initial_symmetries(fermi):
    # what is the quickest way......hrm. 
    rho  = Op([fermi])

def tapered_transform(Transform,hamiltonian,
        initial_state,
        verbose=False,
        ):

    print(stab)
    rho  = initial_state
    cH = hamiltonian.transform(Transform)
    def _remaining_symmetries(stab):
        return len(cstab.null_basis)

    cstab = Stabilizer(cH,verbose=False)
    cstab.gaussian_elmination()
    cstab.find_symmetry_generators()
    cTr = copy(Transform)
    appended_symmetries = []
    while _remaining_symmetries(cstab)>0:
        for S in cstab.null_basis:
            # check compatibility with previous
            use = False
            if len(appended_symmetries)==0:
                appended_symmetries.append(S)
            else:
                for a in appended_symmetries:
                    pass
            if use:
                cTr = partial(
                        modify,
                        fermi=copy(cTr),
                        U=U,
                        qubits=[]
                        paulis=[]
                        eigvals=[]
        cH = hamiltonian.tranform(cTr)
        cstab = Stabilizer(cH,verbose=False)
        cstab.gaussian_elmination()
        cstab.find_symmetry_generators()




        pass

    nTr= None
    iTr= None
    return nTr,iTr
'''
