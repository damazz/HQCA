'''
hqca/quantum/algorithms/_ECC.py

has some error correction techniques

'''
from math import pi
import sys

def _ec_ucc2_parity_single(dc,i,j,k,l,an):
    dc.qc.cx(dc.q[i],dc.q[an])
    dc.qc.cx(dc.q[j],dc.q[an])
    dc.qc.cx(dc.q[k],dc.q[an])
    dc.qc.cx(dc.q[l],dc.q[an])

def _ec_spin_parity(dc,i,j,k,l,an1,an2):
    dc.qc.cx(dc.q[i],dc.q[an1])
    dc.qc.cx(dc.q[j],dc.q[an1])
    dc.qc.cx(dc.q[k],dc.q[an2])
    dc.qc.cx(dc.q[l],dc.q[an2])

def _ec_ancilla_sign(qc,i,j,k,l,anc):
    '''
    takes a measurement in the xxxx basis
    '''
    an = anc[0]
    qc.qc.h(qc.q[i])
    qc.qc.h(qc.q[j])
    qc.qc.h(qc.q[k])
    qc.qc.h(qc.q[l])
    qc.qc.cx(qc.q[i],qc.q[an])
    qc.qc.cx(qc.q[j],qc.q[an])
    qc.qc.cx(qc.q[k],qc.q[an])
    qc.qc.cx(qc.q[l],qc.q[an])
    qc.qc.h(qc.q[i])
    qc.qc.h(qc.q[j])
    qc.qc.h(qc.q[k])
    qc.qc.h(qc.q[l])

def _ec_ancilla_UCC2_test_1s(
        qc,
        phi,i,j,k,l,
        anc=[],
        seq='default',
        pauli='default',
        **kw):
    an = anc[0]
    if phi%(2*pi)>-0.01 and phi%(2*pi)<=0:
        phi= -0.01+2*pi*(phi//(2*pi))
    elif phi%(2*pi)>0 and phi%(2*pi)<0.01:
        phi= 0.01 +2*pi*(pi//(2*pi))
    if seq=='default':
        sequence = 'xxxy'
    else:
        sequence=seq
    if pauli=='default':
        pauli = 'xxxx'
    var =  [[+1]]
    index = [i,j,k,l]
    ind=0
    for item in sequence:
        if item=='x':
            qc.qc.h(qc.q[index[ind]])
        elif item=='y':
            #qc.qc.rx(-pi/2,qc.q[index[ind]])
            qc.qc.z(qc.q[index[ind]])
            qc.qc.s(qc.q[index[ind]])
            qc.qc.h(qc.q[index[ind]])
        ind+=1
    for control in range(i,l):
        target = control+1
        qc.qc.cx(qc.q[control],qc.q[target])
    qc.qc.rz(phi,qc.q[l])
    qc.qc.cx(qc.q[l],qc.q[an])
    for control in reversed(range(i,l)):
        target = control+1
        qc.qc.cx(qc.q[control],qc.q[target])
    ind = 0
    for item in sequence:
        if item=='x':
            qc.qc.h(qc.q[index[ind]])
        elif item=='y':
            #qc.qc.rx(pi/2,qc.q[index[ind]])
            qc.qc.h(qc.q[index[ind]])
            qc.qc.s(qc.q[index[ind]])
        ind+=1
    phase= 0
    for s in range(len(sequence)):
        if not pauli[s]==sequence[s]:
            if pauli[s] in ['z','h']:
                sys.exit('Wrong pauli error correction. Not configured.')
            elif pauli[s]=='x' and sequence[s]=='y':
                qc.qc.cx(qc.q[index[s]],qc.q[an])
                phase+=1 
            elif pauli[s]=='y' and sequence[s]=='x':
                qc.qc.cx(qc.q[index[s]],qc.q[an])
                phase-=1
    if phase==1:
        qc.qc.h(qc.q[an])
        qc.qc.s(qc.q[an])
        qc.qc.h(qc.q[an])
    elif phase==-1:
        qc.qc.h(qc.q[an])
        qc.qc.z(qc.q[an])
        qc.qc.s(qc.q[an])
        qc.qc.h(qc.q[an])


