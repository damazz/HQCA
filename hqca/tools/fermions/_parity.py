import sys


def _commutator_relations(lp,rp):
    if rp=='I':
        return 1,lp
    elif rp==lp:
        return 1,'I'
    elif rp=='Z':
        if lp=='X':
            return -1j,'Y'
        elif lp=='Y':
            return 1j,'X'
    elif rp=='Y':
        if lp=='X':
            return 1j,'Z'
        elif lp=='Z':
            return -1j,'X'
    elif rp=='X':
        if lp=='Y':
            return -1j,'Z'
        elif lp=='Z':
            return 1j,'Y'
    elif rp=='h':
        if lp=='Z':
            return 1,'h'
        elif lp=='X':
            return 1,'+'
        elif lp=='Y':
            return 1j,'+'
    elif rp=='p':
        if lp=='Z':
            return -1,'p'
        elif lp=='X':
            return 1, '-'
        elif lp=='Y':
            return -1j,'-'
    else:
        sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))


class ParitySet:
    def __init__(self,Nq,reduced):
        pass


def ParityTransform(op,Nq):
    coeff = [op.qCo]
    pauli = ['I'*Nq]
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
                for i in range(q+1,Nq):
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
                tc2,tp2 = _commutator_relations(
                        'Z',p[q])
                if q==0:
                    p1 = p[:q]+tp2+p[q+1:]
                else:
                    p1 = p[:q-1]+tp1+tp2+p[q+1:]
                p2 = p[:]
                c1.append(-c*0.5*tc1)
                c2.append(c*0.5*tc2)
            elif o in ['0','h']:
                if q>0:
                    tc1,tp1 = _commutator_relations(
                            'Z',p[q-1])
                tc2,tp2 = _commutator_relations(
                        'Z',p[q])
                if q==0:
                    p1 = p[:q]+tp2+p[q+1:]
                else:
                    p1 = p[:q-1]+tp1+tp2+p[q+1:]
                p2 = p[:]
                c1.append(c*0.5*tc1)
                c2.append(c*0.5*tc2)
            p1s.append(p1)
            p2s.append(p2)
            c1s+= c1
            c2s+= c2
        pauli = p1s+p2s
        coeff = c1s+c2s
    return pauli,coeff
        
