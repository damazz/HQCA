import sys
from copy import deepcopy as copy
from hqca.tools.quantum_strings import *
from hqca.tools._operator import *

def ParityTransform(op):
    Nq = len(op.s)
    pauli = ['I'*Nq]
    new = Operator()+PauliString('I'*Nq,op.c)
    for qi,o in enumerate(op.s):
        # q index, op
        if o=='i':
            continue
        if qi==0:
            if o in ['+','-']:
                s1 = 'X'+(Nq-qi-1)*'X'
                s2 = 'Y'+(Nq-qi-1)*'X'
                c1,c2 = 0.5,((o=='-')-0.5)*1j
            elif o in ['p','h']:
                s1 = 'I'+(Nq-qi-1)*'I'
                s2 = 'Z'+(Nq-qi-1)*'I'
                c1,c2 = 0.5,(o=='h')-0.5
        else:
            if o in ['+','-']:
                s1 = 'I'*(qi-1)+'ZX'+(Nq-qi-1)*'X'
                s2 = 'I'*(qi-1)+'IY'+(Nq-qi-1)*'X'
                c1,c2 = 0.5,((o=='-')-0.5)*1j
            elif o in ['p','h']:
                s1 = 'I'*(qi-1)+'II'+(Nq-qi-1)*'I'
                s2 = 'I'*(qi-1)+'ZZ'+(Nq-qi-1)*'I'
                c1,c2 = 0.5,(o=='h')-0.5
        tem = Operator()
        tem+= PauliString(s1,c1)
        tem+= PauliString(s2,c2)
        new = new*tem
    return new

def Parity(operator,
        **kw,
        ):
    if isinstance(operator,type(QuantumString())):
        return ParityTransform(operator,**kw)
    else:
        new = Operator()
        for op in operator:
            new+= ParityTransform(op,**kw)
        return new


'''
    for q,o in zip(op.qInd[::-1],op.qOp[::-1]):
        p1s,c1s,p2s,c2s = [],[],[],[]
        for p,c in zip(pauli,coeff):
            c1,c2 = [],[]
            if o=='+':
                if q>0:
                    tc0,tp0 = _commutator_relations(
                            'Z',p[q-1])
                tc1,tp1 = _commutator_relations(
                        'X',p[q])
                tc2,tp2 = _commutator_relations(
                        'Y',p[q])
                if q==0:
                    p1 = p[:q]+tp1+p[q+1:]
                else:
                    p1 = p[:q-1]+tp0+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(-1j*c*0.5*tc2)
            elif o=='-':
                if q>0:
                    tc0,tp0 = _commutator_relations(
                            'Z',p[q-1])
                tc1,tp1 = _commutator_relations(
                        'X',p[q])
                tc2,tp2 = _commutator_relations(
                        'Y',p[q])
                if q==0:
                    p1 = p[:q]+tp1+p[q+1:]
                else:
                    p1 = p[:q-1]+tp0+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(1j*c*0.5*tc2)
            if o in ['+','-']:
                for i in range(q+1,MapSet.Nq):
                    nc1,np1 = _commutator_relations(
                            'X',p1[i])
                    nc2,np2 = _commutator_relations(
                            'X',p2[i])
                    p1 = p1[:i]+np1+p1[i+1:]
                    p2 = p2[:i]+np2+p2[i+1:]
                    c1[0]*=nc1
                    c2[0]*=nc2
            elif o in ['1','p']:
                if q>0:
                    tc1,tp1 = _commutator_relations(
                            'Z',p[q-1])
                else:
                    tc1,tp1 = 1,p
                tc2,tp2 = _commutator_relations(
                        'Z',p[q])
                if q==0:
                    p1 = p[:q]+tp2+p[q+1:]
                else:
                    p1 = p[:q-1]+tp1+tp2+p[q+1:]
                p2 = p[:]
                c1.append(-c*0.5*tc1*tc2)
                c2.append(c*0.5)
            elif o in ['0','h']:
                if q>0:
                    tc1,tp1 = _commutator_relations(
                            'Z',p[q-1])
                else:
                    tc1,tp1 = 1,p
                tc2,tp2 = _commutator_relations(
                        'Z',p[q])
                if q==0:
                    p1 = tp2+p[q+1:]
                else:
                    p1 = p[:q-1]+tp1+tp2+p[q+1:]
                p2 = p[:]
                c1.append(c*0.5*tc1*tc2)
                c2.append(c*0.5)
            p1s.append(p1)
            p2s.append(p2)
            c1s+= c1
            c2s+= c2
        pauli = p1s+p2s
        coeff = c1s+c2s
    if MapSet.reduced:
        q1,q2 = MapSet._reduced_set[0],MapSet._reduced_set[1]
        c1,c2 = MapSet._reduced_coeff[q1],MapSet._reduced_coeff[q2]
        npauli = []
        ncoeff = []
        for n in range(len(pauli)):
            tc = copy(coeff[n])
            if pauli[n][q1]=='Z':
                tc = tc*c1
            if pauli[n][q2]=='Z':
                tc = tc*c2
            #if pauli[n][q1] in ['Y','X'] or pauli[n][q2] in ['Y','X']:
            #    print(pauli[n])
            npauli.append(pauli[n][:q1]+pauli[n][(q1+1):q2])
            ncoeff.append(tc)
        pauli = npauli[:]
        coeff = ncoeff[:]
        #print(pauli)
    return pauli,coeff
'''




