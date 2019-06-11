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


def _UCC2_1s(qgdc,phi,i,j,k,l,skip=False,**kw):
    if phi%(2*pi)>-0.01 and phi%(2*pi)<=0:
        phi= -0.01+2*pi*(phi//(2*pi))
    elif phi%(2*pi)>0 and phi%(2*pi)<0.01:
        phi= 0.01 +2*pi*(pi//(2*pi))
    sequence = [['h','h','h','y']] #,['y','y','y','h']]
    var =  [[+1]]#,[-1]]
    index = [i,j,k,l]
    for nt,term in enumerate(sequence):
        ind=0
        for item in term:
            if item=='h':
                qgdc.qc.h(qgdc.q[index[ind]])
            elif item=='y':
                qgdc.qc.z(qgdc.q[index[ind]])
                qgdc.qc.s(qgdc.q[index[ind]])
                qgdc.qc.h(qgdc.q[index[ind]])
            ind+=1
        for control in range(i,l):
            target = control+1
            qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
        #qgdc.qc.rz(phi*var[nt][0],qgdc.q[l])
        qgdc.qc.rz(phi,qgdc.q[l])
        for control in reversed(range(i,l)):
            target = control+1
            qgdc.qc.cx(qgdc.q[control],qgdc.q[target])
        ind = 0
        for item in term:
            if item=='h':
                qgdc.qc.h(qgdc.q[index[ind]])
            elif item=='y':
                qgdc.qc.h(qgdc.q[index[ind]])
                qgdc.qc.s(qgdc.q[index[ind]])
            ind+=1

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

