from math import pi

def _UCC1(qgdc,phi,i,k,**kw):
    sequence = [['h','y'],
            ['y','h']]
    index = [i,k]
    for nt,term in enumerate(sequence):
        ind=0
        for item in term:
            if item=='h':
                qgdc.qc.h(qgdc.q[index[ind]])
            elif item=='y':
                qgdc.qc.rx(-pi/2,qgdc.q[index[ind]])
            ind+=1
        for control in range(i,k):
            target = control+1
            qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
        qgdc.qc.rz(phi/2,qgdc.q[k])
        for control in reversed(range(i,k)):
            target = control+1
            qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
        ind = 0
        for item in term:
            if item=='h':
                qgdc.qc.h(qgdc.q[index[ind]])
            elif item=='y':
                qgdc.qc.rx(pi/2,qgdc.q[index[ind]])
            ind+=1

def _UCC2_full(qgdc,phi1,phi2,phi3,i,j,k,l,operator='++--',**kw):
    '''
    three parameters for appling '++--','-+-+','+--+', etc.
    '''
    if phi1==0 and phi2==0 and phi3==0:
        pass
    else:
        sequence = [
                ['h','h','h','y'],
                ['h','h','y','h'],
                ['h','y','h','h'],
                ['h','y','y','y'],
                ['y','h','h','h'],
                ['y','h','y','y'],
                ['y','y','h','y'],
                ['y','y','y','h']
            ]
        var =  [
                [+1,+1,-1],[+1,-1,+1],
                [-1,+1,+1],[+1,+1,+1],
                [-1,-1,-1],[+1,-1,-1],
                [-1,+1,-1],[-1,-1,+1]]
        # seq 1
        index = [i,j,k,l]
        for nt,term in enumerate(sequence):
            theta = phi1*var[nt][0]+phi2*var[nt][1]+phi3*var[nt][2]
            ind=0
            for item in term:
                if item=='h':
                    qgdc.qc.h(qgdc.q[index[ind]])
                elif item=='y':
                    qgdc.qc.rx(-pi/2,qgdc.q[index[ind]])
                ind+=1
            for control in range(i,l):
                target = control+1
                qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
            qgdc.qc.rz(theta/8,qgdc.q[l])
            for control in reversed(range(i,l)):
                target = control+1
                qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
            ind =  0
            for item in term:
                if item=='h':
                    qgdc.qc.h(qgdc.q[index[ind]])
                elif item=='y':
                    qgdc.qc.rx(pi/2,qgdc.q[index[ind]])
                ind+=1


def _UCC2_1s(Q,phi,i,j,k,l,skip=False,seq='default',**kw):
    if phi%(2*pi)>-0.01 and phi%(2*pi)<=0:
        phi= -0.01+2*pi*(phi//(2*pi))
    elif phi%(2*pi)>0 and phi%(2*pi)<0.01:
        phi= 0.01 +2*pi*(pi//(2*pi))
    if seq=='default':
        sequence='xxxy'
    else:
        sequence = seq
    index = [i,j,k,l]
    ind=0
    for item in sequence:
        if item=='x':
            Q.qc.h(Q.q[index[ind]])
        elif item=='y':
            Q.qc.sdg(Q.q[index[ind]])
            Q.qc.h(Q.q[index[ind]])
        ind+=1
    for control in range(i,l):
        target = control+1
        Q.qc.cx(Q.q[control],Q.q[target])
    Q.qc.rz(phi,Q.q[l])
    for control in reversed(range(i,l)):
        target = control+1
        Q.qc.cx(Q.q[control],Q.q[target])
    ind = 0
    for item in sequence:
        if item=='x':
            Q.qc.h(Q.q[index[ind]])
        elif item=='y':
            Q.qc.h(Q.q[index[ind]])
            Q.qc.s(Q.q[index[ind]])
        ind+=1

def _UCC2_1s_custom(Q,phi,i,j,k,l,skip=False,seq='default',**kw):
    if phi%(2*pi)>-0.01 and phi%(2*pi)<=0:
        phi= -0.01+2*pi*(phi//(2*pi))
    elif phi%(2*pi)>0 and phi%(2*pi)<0.01:
        phi= 0.01 +2*pi*(pi//(2*pi))
    if seq=='default':
        sequence='xxxy'
    else:
        sequence = seq
    index = [i,j,k,l]
    for item in sequence:
        ind=0
        if item=='x':
            Q.qc.h(Q.q[index[ind]])
        elif item=='y':
            Q.qc.sdg(Q.q[index[ind]])
            Q.qc.h(Q.q[index[ind]])
        ind+=1
    Q.qc.cx(Q.q[l],Q.q[k])
    Q.qc.cx(Q.q[k],Q.q[j])
    Q.qc.cx(Q.q[j],Q.q[i])
    Q.qc.rz(phi,Q.q[i])
    Q.qc.cx(Q.q[j],Q.q[i])
    Q.qc.cx(Q.q[k],Q.q[j])
    Q.qc.cx(Q.q[l],Q.q[k])
    ind = 0
    for item in sequence:
        if item=='x':
            Q.qc.h(Q.q[index[ind]])
        elif item=='y':
            Q.qc.h(Q.q[index[ind]])
            Q.qc.s(Q.q[index[ind]])
        ind+=1

def _UCC2_2s_custom(Q,phi,i,j,k,l,operator='-+-+',spin='abab',start='xxxy',**kw):
    if not spin=='abab':
        print('Not configured for other spin excitations.')
        sys.exit()
    if start=='xxxy':
        Q.qc.h(Q.q[i])
        Q.qc.z(Q.q[i])
        Q.qc.h(Q.q[j])
        Q.qc.h(Q.q[k])
        Q.qc.sdg(Q.q[l])
        Q.qc.h(Q.q[l])
        Q.qc.cx(Q.q[l],Q.q[k])
        Q.qc.cx(Q.q[k],Q.q[j])
        Q.qc.cx(Q.q[i],Q.q[j])

        Q.qc.rz(phi/2,Q.q[j])
        Q.qc.z(Q.q[k])
        Q.qc.h(Q.q[j])
        Q.qc.cx(Q.q[j],Q.q[i])
        Q.qc.cx(Q.q[j],Q.q[k])
        Q.qc.sdg(Q.q[i])
        Q.qc.z(Q.q[j])
        Q.qc.h(Q.q[j])
        Q.qc.h(Q.q[i])

        Q.qc.sdg(Q.q[k])
        Q.qc.h(Q.q[k])
        Q.qc.s(Q.q[k])
        Q.qc.s(Q.q[i])

        Q.qc.rz(phi/2,Q.q[j])

        Q.qc.cx(Q.q[i],Q.q[j])
        Q.qc.cx(Q.q[k],Q.q[j])
        Q.qc.cx(Q.q[l],Q.q[k])

        Q.qc.h(Q.q[i])
        Q.qc.s(Q.q[i])
        Q.qc.h(Q.q[j])
        Q.qc.h(Q.q[k])
        Q.qc.s(Q.q[k])
        Q.qc.h(Q.q[l])
        Q.qc.s(Q.q[l])
    elif start=='yxyy':
        Q.qc.sdg(Q.q[i])
        Q.qc.sdg(Q.q[k])
        Q.qc.sdg(Q.q[l])
        Q.qc.h(Q.q[i])
        Q.qc.h(Q.q[j])
        Q.qc.h(Q.q[k])
        Q.qc.h(Q.q[l])
        Q.qc.cx(Q.q[l],Q.q[k])
        Q.qc.cx(Q.q[k],Q.q[j])
        Q.qc.cx(Q.q[i],Q.q[j])

        Q.qc.rz(phi/2,Q.q[j])

        Q.qc.h(Q.q[j])
        Q.qc.sdg(Q.q[i])
        Q.qc.h(Q.q[i])
        Q.qc.sdg(Q.q[i])

        Q.qc.cx(Q.q[j],Q.q[i])
        Q.qc.cx(Q.q[j],Q.q[k])
        Q.qc.z(Q.q[j])
        Q.qc.h(Q.q[j])
        Q.qc.rz(phi/2,Q.q[j])
        Q.qc.sdg(Q.q[j])
        Q.qc.h(Q.q[j])
        Q.qc.sdg(Q.q[j])

        Q.qc.cx(Q.q[i],Q.q[j])
        Q.qc.cx(Q.q[k],Q.q[j])
        Q.qc.cx(Q.q[l],Q.q[k])

        Q.qc.h(Q.q[i])
        Q.qc.h(Q.q[j])
        Q.qc.h(Q.q[k])
        Q.qc.h(Q.q[l])
        Q.qc.s(Q.q[l])


def _UCC2_2s(qgdc,phi,i,j,k,l,skip=False,operator='-+-+',spin='aabb'):
    # set phi
    if phi%(2*pi)>-0.01 and phi%(2*pi)<=0:
        phi= -0.01+2*pi*(phi//(2*pi))
    elif phi%(2*pi)>0 and phi%(2*pi)<0.01:
        phi= 0.01 +2*pi*(pi//(2*pi))
    # set operator
    var =  [[+1],[+1]]
    if spin in ['aabb','bbaa']:
        sequence = [['h','h','h','y'],['y','y','h','y']]
    elif spin in ['abab','baba']:
        sequence = [['h','h','h','y'],['y','h','y','y']]
        var =  [[-1],[-1]]
    elif spin in ['abba','baab']:
        sequence = [['h','h','h','y'],['h','y','y','y']]
    index = [i,j,k,l]
    for nt,term in enumerate(sequence):
        ind=0
        for item in term:
            if item=='h':
                qgdc.qc.h(qgdc.q[index[ind]])
            elif item=='y':
                qgdc.qc.rx(-pi/2,qgdc.q[index[ind]])
            ind+=1
        for control in range(i,l):
            target = control+1
            qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
        qgdc.qc.rz(phi*var[nt][0]/2,qgdc.q[l])
        for control in reversed(range(i,l)):
            target = control+1
            qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
        ind = 0
        for item in term:
            if item=='h':
                qgdc.qc.h(qgdc.q[index[ind]])
            elif item=='y':
                qgdc.qc.rx(pi/2,qgdc.q[index[ind]])
            ind+=1

def _UCC2_4s(qgdc,phi,i,j,k,l,skip=False,operator='-+-+',spin='aabb'):
    # set phi
    if phi%(2*pi)>-0.01 and phi%(2*pi)<=0:
        phi= -0.01+2*pi*(phi//(2*pi))
    elif phi%(2*pi)>0 and phi%(2*pi)<0.01:
        phi= 0.01 +2*pi*(pi//(2*pi))
    # set operator
    if spin in ['aabb','bbaa']:
        var =  [[+1],[-1],[-1],[+1]]
        sequence = [
                ['h','y','h','h'],['y','h','y','y'],
                ['y','h','h','h'],['h','y','y','y']]
    elif spin in ['abab','baba']:
        var =  [[+1],[-1],[-1],[+1]]
        sequence = [
                ['h','h','y','h'],['y','y','h','y'],
                ['y','h','h','h'],['h','y','y','y']]
    elif spin in ['abba','baab']:
        var =  [[+1],[-1],[-1],[+1]]
        sequence = [
                ['h','h','h','y'],['y','y','y','h'],
                ['y','h','h','h'],['h','y','y','y']]
    index = [i,j,k,l]
    for nt,term in enumerate(sequence):
        ind=0
        for item in term:
            if item=='h':
                qgdc.qc.h(qgdc.q[index[ind]])
            elif item=='y':
                qgdc.qc.rx(-pi/2,qgdc.q[index[ind]])
            ind+=1
        for control in range(i,l):
            target = control+1
            qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
        qgdc.qc.rz(phi*var[nt][0]/4,qgdc.q[l])
        for control in reversed(range(i,l)):
            target = control+1
            qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
        ind = 0
        for item in term:
            if item=='h':
                qgdc.qc.h(qgdc.q[index[ind]])
            elif item=='y':
                qgdc.qc.rx(pi/2,qgdc.q[index[ind]])
            ind+=1

