import numpy as np
'''
old, do not use
'''

def rotate_2rdm_unrestricted(
        rdm2,
        U,
        alpha,
        beta,
        region='active'
        ):
    '''
    Input is the standard electron integral matrices, ik format where i,k are
    spatial orbitals. 

    Output is a matrix with indices, i,k,l,j

    Note that U_a should be the left transformation matrix. 

    '''
    N = len(U)
    fa = alpha['inactive']+alpha['active']+alpha['virtual']
    fb = beta['inactive']+beta['active']+beta['virtual']
    if region=='full':
        sys.exit('Don\'t have rotations set up yet.')
    elif region in ['active','as','active_space']:
        alpha=alpha['active']#+alpha['inactive']
        beta =beta['active']#+beta['inactive']
    orbs = alpha+beta
    n2rdm = np.zeros((N,N,N,N),dtype=complex_) # size of spin 2rdm 
    temp1 = np.zeros((N,N,N,N),dtype=complex_) # size of spatial 2rdm 
    temp2 = np.zeros((N,N,N,N),dtype=complex_)
    temp3 = np.zeros((N,N,N,N),dtype=complex_)
    ## alpha alpha portion
    for i in orbs: # i, P
        for a in orbs:
            temp1[i,:,:,:] += U[i,a]*rdm2[a,:,:,:]
        for j in orbs: # j , Q
            for b in orbs:
                temp2[i,:,j,:] += con(U[j,b])*temp1[i,:,b,:]
            for k in orbs: # k, R
                for c in orbs:
                    temp3[i,k,j,:] += U[k,c]*temp2[i,c,j,:]
                for l in orbs: # l , S
                    for d in orbs:
                        # i,k,j,l -> P,R,Q,S
                        n2rdm[i,k,j,l]+= con(U[l,d])*temp3[i,k,j,d]
    return n2rdm

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
    n2rdm = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_) # size of spin 2rdm 
    temp1 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_) # size of spatial 2rdm 
    temp2 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_)
    temp3 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_)
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
                temp2[i,:,j,:] += con(U_a[Q,B])*temp1[i,:,b,:]
                #temp2[i,:,j,:] += U_a[Q,B]*temp1[i,:,b,:]
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
                        n2rdm[i,k,j,l]+= con(U_a[S,D])*temp3[i,k,j,d]
                        #n2rdm[i,k,j,l]+= U_a[S,D]*temp3[i,k,j,d]
    temp1 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_) 
    temp2 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_)
    temp3 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_)
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
                temp2[i,:,j,:] += con(U_b[Q,B])*temp1[i,:,b,:]
                #temp2[i,:,j,:] += U_b[Q,B]*temp1[i,:,b,:]
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
                        n2rdm[i,k,j,l]+= con(U_b[S,D])*temp3[i,k,j,d]
                        #n2rdm[i,k,j,l]+= U_b[S,D]*temp3[i,k,j,d]

    temp1 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_)
    temp2 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_)
    temp3 = np.zeros((N*2,N*2,N*2,N*2),dtype=complex_)
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
                temp2[i,:,j,:] += con(U_a[Q,B])*temp1[i,:,b,:]
                #temp2[i,:,j,:] += U_a[Q,B]*temp1[i,:,b,:]
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
                        n2rdm[i,k,j,l]+= con(U_b[S,D])*temp3[i,k,j,d]
                        n2rdm[k,i,l,j]+= con(U_b[S,D])*temp3[i,k,j,d]
                        n2rdm[i,k,l,j]-= con(U_b[S,D])*temp3[i,k,j,d]
                        n2rdm[k,i,j,l]-= con(U_b[S,D])*temp3[i,k,j,d]
    return n2rdm

