import numpy as np

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
