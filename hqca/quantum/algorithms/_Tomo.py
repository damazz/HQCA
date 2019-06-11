from math import pi

def _apply_pauli_op(qgdc,loc,sigma='x',inv=False):
    if sigma=='z':
        pass
    elif sigma=='x':
        qgdc.qc.h(qgdc.q[loc])
    elif sigma=='i':
        pass
    elif sigma=='y':
        if not inv:
            qgdc.qc.rx(-pi/2,qgdc.q[loc])
        else:
            qgdc.qc.rx(+pi/2,qgdc.q[loc])

def _pauli_2rdme_inline_symm(qc,i,j,k,l,anc,pauli='zzzz',swap=False):
    _apply_pauli_op(qc,i,pauli[0])
    _apply_pauli_op(qc,j,pauli[1])
    _apply_pauli_op(qc,k,pauli[2])
    _apply_pauli_op(qc,l,pauli[3])
    if not swap:
        for q in [i,j,k,l]:
            qc.qc.cx(qc.q[q],qc.q[anc])
    else:
        qc.qc.cx(qc.q[i],qc.q[anc])
    _apply_pauli_op(qc,i,pauli[0],inv=True)
    _apply_pauli_op(qc,j,pauli[1],inv=True)
    _apply_pauli_op(qc,k,pauli[2],inv=True)
    _apply_pauli_op(qc,l,pauli[3],inv=True)

def _pauli_2rdm(qgdc,i,j,k,l,pauli='zzzz'):
    '''
    applies operators on i,j,k,l, assuming that they are ordered
    '''
    for a in range(i+1,j):
        qgdc.qc.z(qgdc.q[a])
    for b in range(k+1,l):
        qgdc.qc.z(qgdc.q[b])
    _apply_pauli_op(qgdc,i,pauli[0])
    _apply_pauli_op(qgdc,j,pauli[1])
    _apply_pauli_op(qgdc,k,pauli[2])
    _apply_pauli_op(qgdc,l,pauli[3])

def _ses_tomo_1rdm_(Q,i,k,imag=False):
    '''
    generic 1rdm circuit for ses method
    '''
    # apply cz phase
    for l in range(i+1,k):
        Q.qc.cz(Q.q[i],Q.q[l])
    # apply cnot1
    Q.qc.cx(Q.q[k],Q.q[i])
    Q.qc.x(Q.q[k])
    if imag:
        Q.qc.s(Q.q[k])
    # ch gate
    Q.qc.ry(pi/4,Q.q[k])
    Q.qc.cx(Q.q[i],Q.q[k])
    Q.qc.ry(-pi/4,Q.q[k])
    if imag:
        Q.qc.s(Q.q[k])
        Q.qc.x(Q.q[i])
        Q.qc.cz(Q.q[i],Q.q[k])
        Q.qc.x(Q.q[i])
    # apply cnot2
    Q.qc.x(Q.q[k])
    Q.qc.cx(Q.q[k],Q.q[i])
    return Q

