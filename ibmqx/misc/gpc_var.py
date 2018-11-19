import sympy as sy
from sympy import pprint
import sys
import numpy as np
import gates as g
import canonical as canon
from sympy.physics.quantum import TensorProduct as tp
from mpmath import nprint
np.set_printoptions(precision=5)
#
# Initialize system
#
c1,c2,c3,s1,s2,s3 = sy.symbols('c1,c2,c3,s1,s2,s3')
t1,t2,t3= sy.symbols('t1,t2,t3')

#r1 = sy.Matrix([[c1,-s1],[s1,c1]])
#r2 = sy.Matrix([[c2,-s2],[s2,c2]])
#r3 = sy.Matrix([[c3,-s3],[s3,c3]])
r1 = sy.Matrix([[sy.cos(t1),-sy.sin(t1)],[sy.sin(t1),sy.cos(t1)]])
r2 = sy.Matrix([[sy.cos(t2),-sy.sin(t2)],[sy.sin(t2),sy.cos(t2)]])
r3 = sy.Matrix([[sy.cos(t3),-sy.sin(t3)],[sy.sin(t3),sy.cos(t3)]])
i4 = sy.eye(4)
i2 = sy.eye(2)


r1_1 = tp(r1,i4)
r1_2 = tp(i2,tp(r1,i2))
r1_3 = tp(i4,r1)
r2_1 = tp(r2,i4)
r2_2 = tp(i2,tp(r2,i2))
r2_3 = tp(i4,r2)
r3_1 = tp(r3,i4)
r3_2 = tp(i2,tp(r3,i2))
r3_3 = tp(i4,r3)
wf = np.matrix([[1],[0],[0],[0],[0],[0],[0],[0]])
wf_1 =np.copy(wf)


#
# Perform Transformations on states
#
wf_1 = g.g3_CNOT_32 *g.g3_CNOT_13*r2_1*r1_3*wf_1
#wf_1 = g.g3_CNOT_13*r1_1*wf_1 #02
#wf_1 = g.g3_CNOT_12*r2_3*wf_1 #01 
#wf_1 = g.g3_CNOT_32*r3_3*wf_1 #21
for a in sy.preorder_traversal(wf_1):
    if isinstance(a,sy.Float):
        wf_1 = wf_1.subs(a,round(a,2))
pprint(wf_1)
wf = {}
wf['111000']=wf_1[0]
wf['011001']=wf_1[1]
wf['101010']=wf_1[2]
wf['001011']=wf_1[3]
wf['110100']=wf_1[4]
wf['010101']=wf_1[5]
wf['100110']=wf_1[6]
wf['000111']=wf_1[7]

print(wf)


def build_2rdm(wavefunction,alpha=[0,1,2],beta=[3,4,5]):
    '''
    Given a wavefunction, and the alpha beta orbitals, will construct the 2RDM for a system. 
    Note, the output format is for a general two body operator, aT_i, aT_k, a_l, a_j
    i/j are electron 1, and k/l are electron 2
    '''
    def phase_2e_oper(I,K,L,J,det):
        # very general phase operator - works for ANY i,k,l,j
        # generates new occupation as well as the sign of the phase

        def new_det(i,k,l,j,det):
            # transform old determinant to new
            det=det[:j]+'0'+det[j+1:]
            det=det[:l]+'0'+det[l+1:]
            det=det[:k]+'1'+det[k+1:]
            det=det[:i]+'1'+det[i+1:]
            return det
        def delta(a,b):
            # delta function
            delta =0
            if a==b:
                delta = 1
            return delta
        def det_phase(det,place):
            # generate phase of a determinant based on number of occupied orbitals with index =1
            p = 1
            for i in range(0,place):
                if det[i]=='1':
                    p*=-1
            return p
        # conditionals for switching orbital orderings
        a1 = (L<=J)
        b1,b2 = (K<=L),(K<=J)
        c1,c2,c3 = (I<=K),(I<=L),(I<=J)
        eps1,eps2,eps3 = 1,1,1
        if a1%2==1: #i.e., if J<L
            eps1=-1
        if (b1+b2)%2==1 : # if K>L, K>J
            eps2=-1
        if (c1+c2+c3)%2==1: # if I>K,I>L,I>J
            eps3=-1
        t2 = 1-delta(L,J) # making sure L/J, I/K not same orbital
        t1 = 1-delta(I,K)
        t7 = eps1*eps2*eps3 
        d1 = delta(I,L) 
        d2 = delta(I,J)
        d3 = delta(K,L)
        d4 = delta(K,J)
        kI = int(det[I]) # occupation of orbitals
        kK = int(det[K])
        kL = int(det[L])
        kJ = int(det[J])
        pI = det_phase(det,I)
        pK = det_phase(det,K)
        pL = det_phase(det,L)
        pJ = det_phase(det,J)
        '''if fermi=='on':
            eps1,eps2,eps3=1,1,1
            if not c1:
                eps1=-1
            ps = 1
            pq = 1
            t7 = eps1
        '''
        t6 = pJ*pL*pK*pI
        t5 = kL*kJ # if 0, then we have a problem
        t3 = d1+d2+1-kI # IL or IJ are the same, and I is already occupied - if I is occupied and not I=J, or I=L, then 0
        t4 = d3+d4+1-kK # same as above, for K and K/L, K/J
        Phase = t1*t2*t3*t4*t5*t6*t7
        ndet = new_det(I,K,L,J,det)
        return Phase,ndet

    #
    # First, alpha alpha 2-RDM, by selecting combinations only within alpha
    #

    norb = len(alpha)+len(beta)

    #rdm2 = sy.zeros((norb,norb,norb,norb))
    rdm2 = np.reshape(tp(sy.zeros(norb),sy.zeros(norb)),(norb,norb,norb,norb))
    for i in alpha:
        for k in alpha:
            if (i<k):
                for l in alpha:
                    for j in alpha:
                        if (l<j):
                            for det1 in wavefunction:
                                ph,check = phase_2e_oper(i,k,l,j,det1)
                                #print(i,k,l,j,det1,ph,check)
                                if ph==0:
                                    continue
                                else:
                                    try:
                                        # here we have symmetry for 4 symmetric orbitals
                                        rdm2[i,k,l,j]+= +1*ph*wavefunction[det1]*wavefunction[check] 
                                        #print(+1*ph*wavefunction[det1]*wavefunction[check])
                                        rdm2[k,i,j,l]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[i,k,j,l]+= -1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[k,i,l,j]+= -1*ph*wavefunction[det1]*wavefunction[check]   
                                    except KeyError as e:
                                        #print(e)
                                        continue
    # Now, the beta beta block
    for i in beta:
        for k in beta:
            if (i<k):
                for l in beta:
                    for j in beta:
                        if (l<j):
                            for det1 in wavefunction:
                                ph,check = phase_2e_oper(i,k,l,j,det1)
                                if ph==0:
                                    continue
                                else:
                                    try:
                                        rdm2[i,k,l,j]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[k,i,j,l]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[i,k,j,l]+= -1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[k,i,l,j]+= -1*ph*wavefunction[det1]*wavefunction[check]
                                    except KeyError:
                                        continue

    # now, the non-covetted alpha beta block
    for i in alpha:
        for k in beta:
            if (i<k):
                for l in beta:
                    for j in alpha:
                        if (l<j or l>=j):
                            for det1 in wavefunction:
                                ph,check = phase_2e_oper(i,k,l,j,det1)
                                '''
                                print(i,k,l,j,det1,ph,check,
                                    map_lambda[i],
                                    map_lambda[j],
                                    map_lambda[k],
                                    map_lambda[l])
                                '''
                                if ph==0:
                                    continue
                                else:
                                    try:
                                        rdm2[i,k,l,j]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[k,i,j,l]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                    except KeyError:
                                        continue
    # now, the non-covetted beta alpha block 
    # (note, symmetric  w.r.t. to the conjugate alpha-beta block) 
    for i in beta:
        for k in alpha:
            if (i<k):
                for l in alpha:
                    for j in beta:
                        if (l<j or l>=j):
                            for det1 in wavefunction:
                                ph,check = phase_2e_oper(i,k,l,j,det1)
                                if ph==0:
                                    continue
                                else:
                                    try:
                                        rdm2[i,k,l,j]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[k,i,j,l]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                    except KeyError:
                                        continue
    # All set! 
    # Returning the 2 electron reduced density matrix! 
    return rdm2
     

def check_2rdm(rdm2):
    # given a 2rdm, generate the traced out 1-RDM
    def rdme1(i,j,rdm2,N):
        # given a 2rdm, generate a 1 rdm element (tracing out k==l)
        # really necessary? no....
        rdme=0
        for k in range(0,N):
            rdme+= rdm2[i,k,k,j]
        return rdme
    test_1rdm = sy.zeros(6)
    for i in range(0,6):
        for j in range(0,6):
            test_1rdm[i,j]=rdme1(i,j,rdm2,6)
    test_1rdm*=1/2
    return test_1rdm
test = build_2rdm(wf)
rdm1 = check_2rdm(test)
#pprint(rdm1[0,0])
pprint(sy.trigsimp(rdm1))
#pprint(rdm1)

