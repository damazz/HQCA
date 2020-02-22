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
    elif rp=='p':
        if lp=='Z':
            return -1,'p'
    else:
        sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))

def JordanWignerTransform(op,Nq,**kw):
    coeff = [op.qCo]
    pauli = ['I'*Nq]
    for q,o in zip(op.qInd[::-1],op.qOp[::-1]):
        p1s,c1s,p2s,c2s = [],[],[],[]
        for p,c in zip(pauli,coeff):
            c1,c2 = [],[]
            if o=='+':
                tc1,tp1 = _commutator_relations(
                        'X',p[q])
                tc2,tp2 = _commutator_relations(
                        'Y',p[q])
                p1 = p[:q]+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(-1j*c*0.5*tc2)
            elif o=='-':
                tc1,tp1 = _commutator_relations(
                        'X',p[q])
                tc2,tp2 = _commutator_relations(
                        'Y',p[q])
                p1 = p[:q]+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(1j*c*0.5*tc2)
            if o in ['+','-']:
                for i in range(q):
                    nc1,np1 = _commutator_relations(
                            'Z',p1[i])
                    nc2,np2 = _commutator_relations(
                            'Z',p2[i])
                    p1 = p1[:i]+np1+p1[i+1:]
                    p2 = p2[:i]+np2+p2[i+1:]
                    c1[0]*=nc1
                    c2[0]*=nc2
            elif o in ['1','p']:
                print(q)
                tc1,tp1 =1,p[q]
                tc2,tp2 = _commutator_relations(
                        'Z',p[q])
                p1 = p[:q]+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(-1*c*0.5*tc2)
            elif o in ['0','h']:
                tc1,tp1 =1,p[q]
                tc2,tp2 = _commutator_relations(
                        'Z',p[q])
                p1 = p[:q]+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(c*0.5*tc2)
            p1s.append(p1)
            p2s.append(p2)
            c1s+= c1
            c2s+= c2
        pauli = p1s+p2s
        coeff = c1s+c2s
    return pauli,coeff

def _project(self):
    pass

def InverseJordanWigner():
    pass
