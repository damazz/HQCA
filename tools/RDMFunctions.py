#
# rdmf.py 
# Deals with functions that involve or are pertaining to reduced density matrices 
#
import numpy as np
import numpy.linalg as LA
from functools import reduce
from hqca.tools.Functions import contract,expand


def build_2rdm(
        wavefunction,
        alpha,beta,
        region='full',
        rdm2=None):
    '''
    Given a wavefunction, and the alpha beta orbitals, will construct the 2RDM
    for a system.
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

def get_Sz_mat(alpha,beta,s2s):
    '''
    Make sure that the molecular 
    '''
    norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
    norb+= len(beta['virtual']+beta['inactive']+beta['active'])
    alp = alpha['active']
    bet = beta['active']
    alpha = alpha['inactive']+alpha['active']
    beta = beta['inactive']+beta['active']
    sz = np.zeros((norb,norb))
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for pa in alp:
        sz[pa,pa]=0.5
    for pb in bet:
        sz[pb,pb]=-0.5
    return sz

def get_Sz2_mat(
        alpha,
        beta,
        s2s
        ):
    norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
    norb+= len(beta['virtual']+beta['inactive']+beta['active'])
    alp = alpha['active']
    bet = beta['active']
    alpha = alpha['inactive']+alpha['active']
    beta = beta['inactive']+beta['active']
    sz2_2 = np.zeros((norb,norb,norb,norb))
    sz2_1 = np.zeros((norb,norb))
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for pa in alp:
        pb = a2b[pa]
        for qb in bet:
            qa = b2a[qb]
            sz2_2[pa,qa,pa,qa]+=0.25
            sz2_2[pb,qb,pb,qb]+=0.25
            sz2_2[pb,qa,pb,qa]-=0.25
            sz2_2[pa,qb,pa,qb]-=0.25
    for pa in alp:
        sz2_1[pa,pa]+=0.25
    for pb in bet:
        sz2_1[pb,pb]+=0.25
    return sz2_1,sz2_2

def get_SpSm_mat(
        alpha,
        beta,
        s2s
        ):
    norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
    norb+= len(beta['virtual']+beta['inactive']+beta['active'])
    alp = alpha['active']
    bet = beta['active']

    alpha = alpha['inactive']+alpha['active']
    beta = beta['inactive']+beta['active']
    spsm_2 = np.zeros((norb,norb,norb,norb))
    spsm_1 = np.zeros((norb,norb))
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for pa in alp:
        pb = a2b[pa]
        for qb in bet:
            qa = b2a[qb]
            spsm_2[pa,qb,qa,pb]=-1
    for pa in alp:
        spsm_1[pa,pa]=1
    return spsm_1,spsm_2

def get_SmSp_mat(
        alpha,
        beta,
        s2s
        ):
    norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
    norb+= len(beta['virtual']+beta['inactive']+beta['active'])
    alp = alpha['active']
    bet = beta['active']
    alpha = alpha['inactive']+alpha['active']
    beta = beta['inactive']+beta['active']
    smsp = np.zeros((norb,norb,norb,norb))
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for pa in alp:
        pb = a2b[pa]
        for qb in bet:
            qa = b2a[qb]
            smsp[pb,qa,pa,qb]=1
    return smsp


def S2(
        rdm2,
        rdm1,
        alpha,
        beta,
        s2s
        ):
    rdm2 = contract(rdm2)
    spm_1,spm_2  = get_SpSm_mat(
            alpha,
            beta,
            s2s)
    spm_2 = contract(spm_2)
    sz2_1,sz2_2 = get_Sz2_mat(
            alpha,
            beta,
            s2s)
    sz2_2 = contract(sz2_2)
    sz = get_Sz_mat(
            alpha,
            beta,
            s2s)
    s2pm_2 = reduce(np.dot, (spm_2,rdm2)).trace()
    s2pm_1 = reduce(np.dot, (spm_1,rdm1)).trace()
    s2z2_2= reduce(np.dot, (sz2_2,rdm2)).trace()
    s2z2_1= reduce(np.dot, (sz2_1,rdm1)).trace()
    s2z1= reduce(np.dot, (sz,rdm1)).trace()
    return s2pm_2+s2pm_1+s2z2_2+s2z2_1-s2z1

def Sz(rdm1,alpha,beta,s2s):
    sz = reduce(np.dot,
            (
                get_Sz_mat(
                    alpha,
                    beta,
                    s2s
                    ),
                rdm1
                )
        ).trace()
    return sz




def spin_rdm_to_spatial_rdm(
        rdm2,
        alpha,
        beta,
        s2s
        ):
    alpha = alpha['inactive']+alpha['active']+alpha['virtual']
    beta  =  beta['inactive']+ beta['active']+ beta['virtual']
    Nso = rdm2.shape[0]
    No = Nso//2
    nrdm2 = np.zeros((No,No,No,No))
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    '''
    for i in alpha:
        for j in alpha:
            for k in alpha:
                for l in alpha:
                    p,q,r,s = s2s[i],s2s[j],s2s[k],s2s[l]
                    nrdm2[p,r,q,s]+=rdm2[i,k,j,l]
    for i in alpha:
        for j in alpha:
            for k in beta:
                for l in beta:
                    print('{} {} {} {}'.format(i,j,k,l))
                    p,q,r,s = s2s[i],s2s[j],s2s[k],s2s[l]
                    print('{} {} {} {}'.format(p,q,r,s))
                    nrdm2[p,r,q,s]+=rdm2[i,k,j,l]
                    nrdm2[p,r,q,s]+=rdm2[k,i,l,j]
    for i in beta:
        for j in beta:
            for k in beta:
                for l in beta:
                    p,q,r,s = s2s[i],s2s[j],s2s[k],s2s[l]
                    nrdm2[p,r,q,s]+=rdm2[i,k,j,l]
    '''
    for i in alpha:
        for j in alpha:
            for k in alpha:
                for l in alpha:
                    p,q,r,s = a2b[i],a2b[j],a2b[k],a2b[l]
                    #P,Q,R,S = s2s[p],s2s[q],s2s[r],s2s[s]
                    I,J,K,L = s2s[i],s2s[j],s2s[k],s2s[l]
                    nrdm2[I,K,J,L] =rdm2[i,k,j,l]
                    nrdm2[I,K,J,L]+=rdm2[p,r,q,s]
                    nrdm2[I,K,J,L]+=rdm2[i,r,j,s]
                    nrdm2[I,K,J,L]+=rdm2[p,k,q,l]
    return nrdm2


def switch_alpha_beta(
        rdm2,
        alpha,
        beta
        ):
    Nso = rdm2.shape[0]
    alpha = alpha['inactive']+alpha['active']+alpha['virtual']
    beta  =  beta['inactive']+ beta['active']+ beta['virtual']
    nrdm2 = np.zeros(rdm2.shape)
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for i in alpha:
        for j in alpha:
            for k in alpha:
                for l in alpha:
                    p,q,r,s = a2b[i],a2b[j],a2b[k],a2b[l]
                    nrdm2[p,q,r,s]=rdm2[i,j,k,l]

    for i in beta:
        for j in beta:
            for k in beta:
                for l in beta:
                    p,q,r,s = b2a[i],b2a[j],b2a[k],b2a[l]
                    nrdm2[p,q,r,s]=rdm2[i,j,k,l]
    for i in alpha:
        for j in alpha:
            for k in beta:
                for l in beta:
                    p,q,r,s = a2b[i],a2b[j],b2a[k],b2a[l]
                    nrdm2[p,r,q,s]=rdm2[i,k,j,l]
                    nrdm2[r,p,q,s]=rdm2[k,i,j,l]
                    nrdm2[r,p,s,q]=rdm2[k,i,l,j]
                    nrdm2[p,r,s,q]=rdm2[i,k,l,j]
    return nrdm2



def rotate_2rdm(
        rdm2,
        U_a,
        U_b,
        alpha,
        beta,
        spin2spac,
        region='active'
        ):
    '''
    Input is the standard electron integral matrices, ik format where i,k are
    spatial orbitals. 

    Output is a matrix with indices, i,k,l,j

    Note that U_a should be the left transformation matrix. 

    '''

    N = len(U_a)
    fa = alpha['inactive']+alpha['active']+alpha['virtual']
    fb = beta['inactive']+beta['active']+beta['virtual']
    if region=='full':
        alpha = alpha['inactive']+alpha['active']+alpha['virtual']
        beta = beta['inactive']+beta['active']+beta['virtual']
    elif region in ['active','as','active_space']:
        alpha=alpha['active']+alpha['inactive']
        beta =beta['active']+beta['inactive']
    n2rdm = np.zeros((N*2,N*2,N*2,N*2)) # size of spin 2rdm 
    temp1 = np.zeros((N*2,N*2,N*2,N*2)) # size of spatial 2rdm 
    temp2 = np.zeros((N*2,N*2,N*2,N*2))
    temp3 = np.zeros((N*2,N*2,N*2,N*2))
    ## alpha alpha portion
    for i in alpha: # i, P
        P = spin2spac[i]
        for a in fa:
            A = spin2spac[a]
            temp1[i,:,:,:] += U_a[P,A]*rdm2[a,:,:,:]
        for j in alpha: # j , Q
            Q = spin2spac[j]
            for b in fa:
                B = spin2spac[b]
                temp2[i,:,j,:] += U_a[Q,B]*temp1[i,:,b,:]
            for k in alpha: # k, R
                R = spin2spac[k]
                for c in fa:
                    C = spin2spac[c]
                    temp3[i,k,j,:] += U_a[R,C]*temp2[i,c,j,:]
                for l in alpha: # l , S
                    S = spin2spac[l]
                    for d in fa:
                        D = spin2spac[d]
                        # i,k,j,l -> P,R,Q,S
                        n2rdm[i,k,j,l]+= U_a[S,D]*temp3[i,k,j,d]
    temp1 = np.zeros((N*2,N*2,N*2,N*2)) 
    temp2 = np.zeros((N*2,N*2,N*2,N*2))
    temp3 = np.zeros((N*2,N*2,N*2,N*2))
    ## beta beta portion
    for i in beta: # i, P
        P = spin2spac[i]
        for a in fb:
            A = spin2spac[a]
            temp1[i,:,:,:] += U_b[P,A]*rdm2[a,:,:,:]
        for j in beta: # j , Q
            Q = spin2spac[j]
            for b in fb:
                B = spin2spac[b]
                temp2[i,:,j,:] += U_b[Q,B]*temp1[i,:,b,:]
            for k in beta: # k, R
                R = spin2spac[k]
                for c in fb:
                    C = spin2spac[c]
                    temp3[i,k,j,:] += U_b[R,C]*temp2[i,c,j,:]
                for l in beta: # l , S
                    S = spin2spac[l]
                    for d in fb:
                        D = spin2spac[d]
                        # i,k,j,l -> P,R,Q,S
                        n2rdm[i,k,j,l]+= U_b[S,D]*temp3[i,k,j,d]

    temp1 = np.zeros((N*2,N*2,N*2,N*2))
    temp2 = np.zeros((N*2,N*2,N*2,N*2))
    temp3 = np.zeros((N*2,N*2,N*2,N*2))
    ## alpha beta portion
    for i in alpha: # i, P
        P = spin2spac[i]
        for a in fa:
            A = spin2spac[a]
            temp1[i,:,:,:] += U_a[P,A]*rdm2[a,:,:,:]
        for j in alpha: # j , Q
            Q = spin2spac[j]
            for b in fa:
                B = spin2spac[b]
                temp2[i,:,j,:] += U_a[Q,B]*temp1[i,:,b,:]
            for k in beta: # k, R
                R = spin2spac[k]
                for c in fb:
                    C = spin2spac[c]
                    temp3[i,k,j,:] += U_b[R,C]*temp2[i,c,j,:]
                for l in beta: # l , S
                    S = spin2spac[l]
                    for d in fb:
                        D = spin2spac[d]
                        # i,k,j,l -> P,R,Q,S
                        n2rdm[i,k,j,l]+= U_b[S,D]*temp3[i,k,j,d]
                        n2rdm[k,i,l,j]+= U_b[S,D]*temp3[i,k,j,d]
                        n2rdm[i,k,l,j]-= U_b[S,D]*temp3[i,k,j,d]
                        n2rdm[k,i,j,l]-= U_b[S,D]*temp3[i,k,j,d]
    return n2rdm


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
