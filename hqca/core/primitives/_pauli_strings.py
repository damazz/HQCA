from math import pi
import sys
'''
takes Pauli strings from qiskit aqua package, and actually adds on Hamiltonian
circuit
'''

def apply_clifford_operation(Q,U):
    def V(n):
        Q.qc.s(n)
        Q.qc.h(n)
        Q.qc.sdg(n)
    def S(n):
        Q.qc.s(n)
    cliff = {
            'H':Q.qc.h,
            'S':S,
            'V':V,
            }
    for n,u in enumerate(U):
        if u in ['I','i']:
            continue
        op = cliff[u]
        op(n)

def pauliOp(Q,loc,sigma='x',inv=False):
    if sigma in ['Z','z']:
        pass
    elif sigma in ['X','x']:
        #Q.qc.rz(pi/2,Q.q[loc])
        #Q.qc.sx(Q.q[loc])
        #Q.qc.rz(pi/2,Q.q[loc])
         
        #
        Q.qc.h(Q.q[loc])
    elif sigma in ['I','i']:
        pass
    elif sigma in ['Y','y']:
        if inv: # S gate
            Q.qc.h(Q.q[loc])
            Q.qc.s(Q.q[loc])
            #
            #Q.qc.rz(pi/2,Q.q[loc])
            #Q.qc.sx(Q.q[loc])
            #Q.qc.rz(pi,Q.q[loc])
            # #
            #Q.qc.rx(-pi/2,Q.q[loc])
        else:
            Q.qc.sdg(Q.q[loc])
            Q.qc.h(Q.q[loc])
            #Q.qc.rx(pi/2,Q.q[loc])
            # #
            #Q.qc.sx(Q.q[loc])
            #Q.qc.rz(pi/2,Q.q[loc])

def apply_pauli_string(Q,pauli):
    if not abs(abs(pauli.c)-1)<1e-4:
        print('Pauli operator:')
        print(pauli)
        sys.exit('Can not implement partial Pauli operator in line.')
    for q,i in enumerate(pauli.s):
        if i=='X':
            Q.qc.x(Q.q[q])
        elif i=='Y':
            Q.qc.rz(pi/2,Q.q[q])
            Q.qc.x(Q.q[q])
            Q.qc.rz(pi/2,Q.q[q])
        elif i=='Z':
            Q.qc.rz(pi,Q.q[q])

def generic_Pauli_term(Q,val,pauli,scaling=1.0):
    s,c = pauli,val
    val*= -1
    # the Rz is actually exp -i theta/2 Z; we are correcting here
    if len(s)==1:
        if s=='I':
            pass
        elif s=='X':
            Q.qc.rx(val*scaling,Q.q[0])
        elif s=='Y':
            Q.qc.ry(val*scaling,Q.q[0])
        elif s=='Z':
            Q.qc.rz(val*scaling,Q.q[0])
    else:
        pauliTerms=0
        ind = []
        terms = []
        for n,i in enumerate(s):
            if not i in ['I']:
                pauliTerms+=1
                ind.append(n)
                terms.append(i)
        if pauliTerms==0:
            Q.qc.u1(val,Q.q[0])
            Q.qc.x(Q.q[0])
            Q.qc.u1(val,Q.q[0])
            Q.qc.x(Q.q[0])
        else:
            # basis
            for n,p in zip(ind,terms):
                pauliOp(Q,n,p)
            # exp cnot
            for n in range(0,pauliTerms-1):
                Q.qc.cx(Q.q[ind[n]],Q.q[ind[n+1]])
            # parameter
            Q.qc.rz(val*scaling,Q.q[ind[-1]])
            # exp cnot
            for n in reversed(range(pauliTerms-1)):
                Q.qc.cx(Q.q[ind[n]],Q.q[ind[n+1]])
            # inv. basis
            for n,p in zip(ind,terms):
                pauliOp(Q,n,p,inv=True)

