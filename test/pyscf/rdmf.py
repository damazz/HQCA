# file for constructing the hamiltonian matrix from the integrals 
import numpy as np
import numpy.linalg as LA


def project_gpc(rdm):
    # given an rdm, projects it onto the plane
    occnum,occorb = LA.eig(rdm)
    # project vector
    occnum.sort()
    on = occnum[:2:-1]
    t = np.sqrt(3)
    norm_vec = np.array([1/t,1/t,-1/t])
    w_vec = on-np.array([1,1,1])
    orthog = np.dot(norm_vec,w_vec)
    proj = on - orthog*norm_vec
    for i in range(0,3):
        proj[i]=min(proj[i],1)

    # now, fit to alpha, beta, gamma
    #print('Projection: {}'.format(proj))
    alpha = np.sqrt(min(1,proj[2]))
    beta  = np.sqrt(max(0,proj[0]-alpha*np.conj(alpha)))
    gamma  = np.sqrt(max(0,proj[1]-proj[2]))
    #print('Alpha: {}, Beta: {}, Gamma: {}'.format(alpha,beta,gamma)    
    return alpha,beta,gamma

def Unroll(M):
    '''
    Unroll a rank 4 tensor to a 2-D matrix.
    Input: Square rank 4 tensor of dimension r
    Output: Square matrix of dimension r^2

    tensor[i][j][k][l] -> matrix[i*r+k][j*r+l]

    Note that axes 0 and 2 are combined, as well as axes 1 and 3. This is in conjunction with the output of PySCF so that
    the 2-RDM has i and j on one axis, and k and l on the other.
    '''

    res = np.zeros([len(M)**2, len(M)**2])

    # Check for rank 4 tensor
    if M.ndim != 4:
        print "Unroll must act on a rank 4 tensor. Your matrix is rank", M.ndim, ". Returning a zero matrix."
        return res

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                for l in range(M.shape[3]):
                    res[i*len(M) + k][j*len(M) + l] = M[i][j][k][l]
    return res


#def build_1rdm_sf(mapping,alpha,beta,gamma):

def wf_BD(alpha,beta,gamma):
    wf = {
        '111000':alpha,
        '100110':beta,
        '010101':gamma}
    return wf

 

def build_2rdm(wavefunction,alpha=[0,1,2],beta=[3,4,5]):
    '''
    generates the 2rdm according to the following two body operator
    convention: aT_i, aT_k, a_l, a_j
     
    no mapping involved, all that is needed is alpha and beta spins
    ordering of orbitals is by beta and 

	'''
    def match_det(det1,det2):
        # generates list of possible operators
        op_list = []
        diff=0
        for i in range(0,len(det1)):
            if det1[i]==det2[j]:
                use=True
            else:
                use=False
                break
        return use

    def phase_2e_oper(I,K,L,J,det):
        # very general phase operator - works for ANY i,j,k,l
        # generates 0 occupation as well as sign phase
        def new_det(i,k,l,j,det):
            det=det[:j]+'0'+det[j+1:]
            det=det[:l]+'0'+det[l+1:]
            det=det[:k]+'1'+det[k+1:]
            det=det[:i]+'1'+det[i+1:]
            return det

        def delta(a,b):
            delta =0
            if a==b:
                delta = 1
            return delta
        def det_phase(det,place):
            p = 1
            for i in range(0,place):
                if det[i]=='1':
                    p*=-1
            return p
        a1 = (L<=J)
        b1,b2 = (K<=L),(K<=J)
        c1,c2,c3 = (I<=K),(I<=L),(I<=J)
        eps1,eps2,eps3 = 1,1,1
        if a1%2==1: #i.e., if J<L
            eps1=-1
        if (b1+b2)%2==1 : # if K>L, K>J
            eps2=-1
        if (c1+c2+c3)%2==1: 
            eps3=-1
        #print(eps1*eps2*eps3)
        t2 = 1-delta(L,J)
        t1 = 1-delta(I,K)
        t7 = eps1*eps2*eps3
        d1 = delta(I,L)
        d2 = delta(I,J)
        d3 = delta(K,L)
        d4 = delta(K,J)
        kI = int(det1[I])
        kK = int(det1[K])
        kL = int(det1[L])
        kJ = int(det1[J])
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
        t5 = kL*kJ
        t3 = d1+d2+1-kI
        t4 = d3+d4+1-kK
        Phase = t1*t2*t3*t4*t5*t6*t7
        ndet = new_det(I,K,L,J,det)
        return Phase,ndet

    #
    # First, alpha alpha 2-RDM
    #

    norb = len(alpha)+len(beta)
    rdm2 = np.zeros((norb,norb,norb,norb))
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
    for i in beta:
        for k in alpha:
            if (i<k):
                for l in alpha:
                    for j in beta:
                        if (l<j or l>=j):
                            for det1 in wavefunction:
                                ph,check = phase_2e_oper(i,k,l,j,det1)
                                #print(i,k,l,j,det1,ph,check)
                                if ph==0:
                                    continue
                                else:
                                    try:
                                        rdm2[i,k,l,j]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                        rdm2[k,i,j,l]+= +1*ph*wavefunction[det1]*wavefunction[check]
                                    except KeyError:
                                        continue
    #print(rdm2)
    # All set! 
    # Returning the 2 electron reduced density matrix! 
    #rdm2 = np.reshape(rdm2,(36,36))
    #print(rdm2.trace())
    return rdm2

def spin_free_rdm1(rdm1,mapping):
    spatial = int(len(rdm1)/2)
    sf_rdm1 = np.zeros((spatial,spatial),dtype=np.complex_)
    for i in range(0,len(rdm1)):
        for j in range(0,len(rdm1)):
            sf_rdm1[mapping[i],mapping[j]]+=rdm1[i,j]
    return sf_rdm1


def spin_free_rdm2(rdm2,mapping):
    spatial = int(len(rdm2)/2)
    sf_rdm2 = np.zeros((spatial,spatial,spatial,spatial))
    rdm1 = check_2rdm(rdm2)
    for i in range(0,len(rdm2)):
        for k in range(0,len(rdm2)):
            for l in range(0,len(rdm2)):
                for j in range(0,len(rdm2)):
                    #check1 = (mapping[i]==0 and mapping[j]==2 and mapping[k]==0 and mapping[l]==2)
                    #if check1:
                    #    print(rdm2[i,k,l,j], i,k,l,j)
                    sf_rdm2[mapping[i],mapping[j],mapping[k],mapping[l]]+=rdm2[i,k,l,j]
                    #if k==j:
                    #    sf_rdm2[mapping[i],mapping[j],mapping[k],mapping[l]]+=rdm1[i,l]
                    '''
                    if k==j:
                        for z in range(0,len(rdm2)):
                            sf_rdm2[mapping[i],mapping[j],mapping[k],mapping[l]]+=0.5*rdm2[i,z,z,l]
                    '''
    return sf_rdm2

def permute_2rdme(a,b,c,d,e,alpha1=True,alpha2=True):
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)
    if (alpha1 and alpha2):
        permute = np.array([
            [a,b,c,d,e],
            [b,a,d,c,e],
            [a,b,d,c,-e],
            [b,a,c,d,-e]])
    elif (not (alpha1 or alpha1)):
        permute = np.array([
            [a,b,c,d,e],
            [b,a,d,c,e],
            [a,b,d,c,-e],
            [b,a,c,d,-e]])
    return permute

def gen_2rdm(a,b,c):
    rdm2 = np.zeros((6,6,6,6),dtype=np.complex_)
    def C(a):
        return np.conj(a)

    values = np.array([
        [0,1,1,0,C(a)*a],
        [0,2,2,0,C(a)*a],
        [3,5,2,0,C(c)*a],
        [0,3,3,0,C(b)*b],
        [0,4,4,0,C(b)*b],
        [1,5,4,0,C(c)*b],
        [1,2,2,1,C(a)*a],
        [3,4,2,1,C(b)*a],
        [1,3,3,1,C(c)*c],
        [0,4,5,1,C(b)*c],
        [1,5,5,1,C(c)*c],
        [1,2,4,3,C(a)*b],
        [3,4,4,3,C(b)*b],
        [0,2,5,3,-C(a)*c],
        [3,5,5,3,C(c)*c]])
    for item in values:
        indices = permute_2rdme(item[0],item[1],item[2],item[3],item[4])
        #print(indices)
        for perm in indices:
            if (((perm[0] in alpha) and (perm[3] in alpha) and (perm[1] in alpha) and (perm[2] in alpha)) or ((perm[0] in beta) and (perm[3] in beta) and (perm[1] in beta) and (perm[2] in beta)) or ((perm[0] in beta) and (perm[3] in beta) and (perm[1] in alpha) and (perm[2] in alpha)) or ((perm[0] in alpha) and (perm[3] in alpha) and (perm[1] in beta) and (perm[2] in beta))):
                rdm2[int(perm[0]),int(perm[1]),int(perm[2]),int(perm[3])]=perm[4]
    return rdm2

def rdme1(i,j,rdm2,N):
    rdme=0
    for k in range(0,N):
        for l in range(0,N):
            if k==l:
                rdme+= rdm2[i,k,l,j]
    return rdme

def check_2rdm(rdm2):
    test_1rdm = np.zeros((6,6),dtype=np.complex_)
    for i in range(0,6):
        for j in range(0,6):
            test_1rdm[i,j]=rdme1(i,j,rdm2,6)
    test_1rdm*=1/2
    return test_1rdm

def build_1rdm(a,b,c):
    rdm = np.zeros((6,6))
    rdm[0,0]=np.conj(a)*a+np.conj(b)*b
    rdm[1,1]=np.conj(a)*a+np.conj(c)*c
    rdm[2,2]=np.conj(a)*a
    rdm[3,3]=np.conj(b)*b+np.conj(c)*c
    rdm[4,4]=np.conj(b)*b
    rdm[5,5]=np.conj(c)*c
    return rdm

def trace_2rdm(rdm2):
    trace = 0
    for i in range(0,6):
        for j in range(0,6):
           if j>i:
                trace+=rdm2[i,j,j,i]
    return trace

map_zeta = {
    0:0, 1:1, 3:2,
    2:3, 4:4, 5:5}
map_lambda = {
    0:0, 1:1, 3:2,
    2:3, 4:5, 5:4}
map_kappa = {
    0:1, 1:0, 3:2,
    2:3, 4:4, 5:5}
map_iota = {
    0:1, 1:0, 3:2,
    2:3, 4:5, 5:4}


map_spatial = {
    0:0, 1:1, 2:2,
    3:0, 4:1, 5:2}

def map_wf(wf,mapping):
    new_wf = {}
    for det, val in wf.items():
        new_det = ''
        for i in range(0,len(det)):
            new_det += '0'
        for i in range(0,len(det)):
            ind = mapping[i]
            new_det = new_det[:ind]+det[i]+new_det[ind+1:]
        new_wf[new_det]=val
    return new_wf

'''
np.set_printoptions(precision=3)

identity = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
lam = 1/np.sqrt(3)
wf = wf_BD(lam,lam,lam)
rdm2 = build_2rdm(wf,identity)
print('Trace of the 2-RDM: {:.3}'.format(float(np.real(trace_2rdm(rdm2)))))
check = check_2rdm(rdm2)
print(check)
print(build_1rdm(lam,lam,lam))
'''
