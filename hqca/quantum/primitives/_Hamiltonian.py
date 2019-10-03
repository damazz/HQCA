from math import pi
'''
takes Pauli strings from qiskit aqua package, and actually adds on Hamiltonian
circuit
'''

def __pauliOp(Q,loc,sigma='x',inv=False):
    if sigma in ['Z','z']:
        pass
    elif sigma in ['X','x']:
        Q.qc.h(Q.q[loc])
    elif sigma in ['I','i']:
        pass
    elif sigma in ['Y','y']:
        if not inv:
            Q.qc.rx(-pi/2,Q.q[loc])
        else:
            Q.qc.rx(pi/2,Q.q[loc])


def _generic_Pauli_term(Q,val,pauli):
    pauliTerms=0
    ind = [] 
    terms = []
    for n,i in enumerate(pauli):
        if not i in ['I','i']:
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
            __pauliOp(Q,n,p)
        # exp cnot
        for n in range(0,pauliTerms-1):
            Q.qc.cx(Q.q[ind[n]],Q.q[ind[n+1]])
        # parameter
        Q.qc.rz(val,Q.q[ind[-1]])
        # exp cnot
        for n in reversed(range(pauliTerms-1)):
            Q.qc.cx(Q.q[ind[n]],Q.q[ind[n+1]])
        # inv. basis
        for n,p in zip(ind,terms):
            __pauliOp(Q,n,p,inv=True)
    

def _generic_Pauli_term_qiskit(Q,term,scaling=1):
    '''

    note input should be from qiskit
    entry 1 is value, entry to is a Pauli object
    '''
    val = term[0]
    pauliStr = term[1].to_label()
    pauliTerms=0
    ind = [] 
    terms = []
    for n,i in enumerate(pauliStr):
        if not i in ['I','i']:
            pauliTerms+=1
            ind.append(n)
            terms.append(i)
    if pauliTerms==0:
        #Q.qc.ph(val*scaling,Q.q[0])
        Q.qc.u1(val,Q.q[0])
        Q.qc.x(Q.q[0])
        Q.qc.u1(val,Q.q[0])
        Q.qc.x(Q.q[0])
    else:
        # basis
        for n,p in zip(ind,terms):
            __pauliOp(Q,n,p)
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
            __pauliOp(Q,n,p,inv=True)

def _add_Hamiltonian(Q,qubOp,scaling):
    for term in qubOp:
        _generic_Pauli_term_qiskit(Q,term,scaling=scaling)
