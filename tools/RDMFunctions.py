#
# rdmf.py 
# Deals with functions that involve or are pertaining to reduced density matrices 
#
import numpy as np
import numpy.linalg as LA

def project_gpc(rdm,s1=1,s2=1,s3=1):
    # given an 3 electron 1-RDM, projects it onto the GPC plane
    occnum,occorb = LA.eig(rdm)
    occnum.sort()
    on = occnum[:2:-1]
    t = np.sqrt(3)
    norm_vec = np.array([1/t,1/t,-1/t])
    w_vec = on-np.array([1,1,1])
    orthog = np.dot(norm_vec,w_vec)
    proj = on - orthog*norm_vec
    for i in range(0,3):
        proj[i]=min(proj[i],1)
    alpha = np.sqrt(min(1,proj[2]))*s1
    beta  = np.sqrt(max(0,proj[0]-alpha*np.conj(alpha)))*s2
    gamma  = np.sqrt(max(0,proj[1]-proj[2]))*s3
    return alpha,beta,gamma, np.sqrt(np.sum(np.square(orthog*norm_vec)))


def project_to_plane(vec):
    v1 = np.array([1,1,1])
    v2 = np.array([0.5,0.5,0.5])
    v3 = np.array([1,0.5,0.5])
    nv1 = v1-v2
    nv2 = v2-v3
    cross = np.cross(nv1,nv2)
    cross = cross/(np.sqrt(np.sum(np.square(cross))))
    w_vec = vec-np.array([1,1,1])
    orthog = np.dot(cross,w_vec)
    proj = vec - orthog*cross
    return proj


def wf_BD(alpha,beta,gamma):
    # Generate a Borland-Dennis wavefunction
    wf = {
        '111000':alpha,
        '100110':beta,
        '010101':gamma}
    return wf

def build_2rdm(
        wavefunction,
        alpha,beta,
        region='full',
        rdm2=None):
    '''
    Given a wavefunction, and the alpha beta orbitals, will construct the 2RDM for a system. 
    Note, the output format is for a general two body operator, aT_i, aT_k, a_l, a_j
    i/j are electron 1, and k/l are electron 2
    '''
    def phase_2e_oper(I,K,L,J,det,low=0):
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
        def det_phase(det,place,start):
            # generate phase of a determinant based on number of occupied orbitals with index =1
            p = 1
            for i in range(start,place):
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
        pI = det_phase(det,I,start=low)
        pK = det_phase(det,K,start=low)
        pL = det_phase(det,L,start=low)
        pJ = det_phase(det,J,start=low)
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
    if region=='full':
        norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
        norb+= len(beta['virtual']+beta['inactive']+beta['active'])
        alpha = alpha['inactive']+alpha['active']
        beta = beta['inactive']+beta['active']
        rdm2 = np.zeros((norb,norb,norb,norb))
    elif region=='active':
        ai = set(alpha['inactive'])
        bi = set(beta['inactive'])
        alpha = alpha['inactive']+alpha['active']
        beta = beta['inactive']+beta['active']
    for i in alpha:
        for k in alpha:
            if (i<k):
                for l in alpha:
                    for j in alpha:
                        if (l<j):
                            #low = min(i,l)
                            low = 0 
                            if region=='active':
                                if set((i,j,k,l)).issubset(ai):
                                    continue
                                rdm2[i,k,l,j]=0
                                rdm2[k,i,l,j]=0
                                rdm2[k,i,j,l]=0
                                rdm2[i,k,j,l]=0
                            for det1 in wavefunction:
                                ph,check = phase_2e_oper(i,k,l,j,det1,low)
                                if ph==0:
                                    continue
                                else:
                                    if check in wavefunction:
                                        rdm2[i,k,l,j]+= -1*ph*wavefunction[det1]*wavefunction[check] 
                                        rdm2[k,i,j,l]+= -1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[i,k,j,l]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[k,i,l,j]+= +1*ph*wavefunction[det1]*wavefunction[check]   

    for i in beta:
        for k in beta:
            if (i<k):
                for l in beta:
                    for j in beta:
                        if (l<j):
                            low = 0 
                            if region=='active':
                                if set((i,j,k,l)).issubset(bi):
                                    continue
                                rdm2[i,k,l,j]=0
                                rdm2[k,i,l,j]=0
                                rdm2[k,i,j,l]=0
                                rdm2[i,k,j,l]=0
                            for det1 in wavefunction:
                                ph,check = phase_2e_oper(i,k,l,j,det1,low)
                                if ph==0:
                                    continue
                                else:
                                    if check in wavefunction:
                                        rdm2[i,k,l,j]+= -1*ph*wavefunction[det1]*wavefunction[check] 
                                        rdm2[k,i,j,l]+= -1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[i,k,j,l]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[k,i,l,j]+= +1*ph*wavefunction[det1]*wavefunction[check]   

    for i in alpha:
        for j in alpha:
            for k in beta:
                for l in beta:
                    low = 0
                    if region=='active':
                        if set((i,j,k,l)).issubset(ai.union(bi)):
                            continue
                        rdm2[i,k,j,l]=0
                        rdm2[i,k,l,j]=0
                        rdm2[k,i,l,j]=0
                        rdm2[k,i,j,l]=0
                    for det1 in wavefunction:
                        ph,check = phase_2e_oper(i,k,l,j,det1,low)
                        if ph==0:
                            continue
                        else:
                            if check in wavefunction:
                                rdm2[i,k,l,j]+= -1*ph*wavefunction[det1]*wavefunction[check] 
                                rdm2[k,i,j,l]+= -1*ph*wavefunction[det1]*wavefunction[check]
                                rdm2[i,k,j,l]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                rdm2[k,i,l,j]+= +1*ph*wavefunction[det1]*wavefunction[check]   
    return rdm2



def check_2rdm(rdm2,Ne):
    # given a 2rdm, generate the traced out 1-RDM
    def rdme1(i,j,rdm2,N):
        # given a 2rdm, generate a 1 rdm element (tracing out k==l)
        # really necessary? no....
        rdme=0
        for k in range(0,N):
            #if (i==3 and j==4):
            #    print(i,k,k,j,rdm2[i,k,k,j])
            rdme+= rdm2[i,k,j,k]
        return rdme

    No = rdm2.shape[0] #number of orbitals

    test_1rdm = np.zeros((No,No),dtype=np.complex_)
    for i in range(0,No):
        for j in range(0,No):
            test_1rdm[i,j]=rdme1(i,j,rdm2,No)
    test_1rdm*=1/(Ne-1)
    return test_1rdm

def trace_2rdm(rdm2):
    # gives proper trace (i.e, sum of identity elements with j>i
    N = rdm2.shape[0]
    trace = 0
    for i in range(0,N):
        for j in range(0,N):
           if j>i:
                trace+=rdm2[i,j,i,j]
    return trace

