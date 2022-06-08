""" 
Implementation of the ACSE algorithm with a conjugate gradient search direction

For details on the line search, interpolation, 
BFGS, and l-BFGS algorithms, see Nocedal & Wright, chapters 6 and 7. 

"""

from copy import deepcopy as copy
import numpy as np
from hqca.operators import *
from hqca.acse._line_search_acse import LineSearchACSE
from hqca.acse._tools_acse import truncation, vector_to_operator
from hqca.core import *
from warnings import warn
import math

def _conjugate_gradient_step(acse):
    # okay, given gradient information
    # get inverse
    fermi = acse.acse_update in ['q','c']
    keys = acse.rdme#+acse.rdme+acse.rdme+acse.rdme
    symm = 4

    G_0 = copy(acse.A) #this already has a factor of epsilon from either build or previous run
    if len(G_0.shape)>1:
        G0 = np.asarray(G_0).T
    else:
        G0 = np.asarray([G_0]).T

    p_k = acse.p
    #
    if acse.psi.closed==1:
        prev = []
    else:
        prev = acse._log_psi[acse.psi.closed:]
    p_k = truncation(
            p_k,acse.S_thresh_rel,acse.S_min,
            method=acse.trunc_method,
            include=acse.trunc_include,
            gradient=G0,
            previous=prev,
            keys=keys,
            mapping=acse.rdme_mapping,
            )
            
    ovlp = p_k.T.dot(G0)[0,0]
    norm_p = np.linalg.norm(p_k)
    norm_g = np.linalg.norm(G0)
    print('<gT|p>**2 : {}'.format(ovlp/(norm_p*norm_g)))
    print('norm p, {}, norm g {}'.format(norm_p,norm_g))

    acse._opt_log = []
    acse._opt_en = []
    #alp,f_star,g_star,err = line_search(acse,p_k,G0,c2=0.5)

    ls = LineSearchACSE(acse)
    if acse.total.iter>0:
        dphi0 = symm*np.dot(acse.A,p_k)[0,0]
        dphi1 = symm*np.dot(acse.log_A[-1].T,acse.log_p[-1])[0,0]
        alp = acse.alp*(dphi1/dphi0)
        alp = 2*(acse.log_E[-1]-acse.log_E[-2])/dphi1
    else:
        alp = acse.epsilon
    #print('suggested alp...',alp)
    if math.isnan(alp) or alp<1e-2:
        alp = acse.epsilon
    alp = min(1,1.01*alp)
    alp = max(0.1,alp)
    acse.log_A.append(G0)
    acse.log_p.append(p_k)

    ls.run(p_k,g_0=G0,c2=0.25,alp_1=alp)
    alp = ls.alp_star
    acse.alp = alp
    f_star = ls.f_star
    g_star = ls.g_star

    if ls.err:
        acse.total.done = True
        return None
    op = vector_to_operator(p_k*alp,fermi=fermi,keys=keys,N=acse.qs.dim,
            S_min=acse.S_min,)
    #if acse.acse_update in ['u']:
    #    P = op
    #else:
    #    P = op.transform(acse.transform_psi)
    P = op
    acse.psi = acse.psi + P
    #print(P)

    acse.A = g_star #already has factor of epsilon
    acse.p = p_k # no factor of epsilon

    if acse.psi.closed==1:
        acse._log_psi = []
    else:
        P_k_bool = np.zeros(p_k.shape)
        for i in range(len(p_k)):
            if abs(p_k[i])>1e-10:
                print(i,keys[i])
                P_k_bool[i]=1
        temp = np.zeros(p_k.shape)
        for i in acse._log_psi[acse.psi.closed:]:
            temp+= i
        temp = np.multiply(P_k_bool,temp)
        P_k_bool = np.mod(P_k_bool+temp,2)
        if np.any(P_k_bool):
            acse._log_psi.append(P_k_bool)
        print('Length log_g: {}'.format(len(acse._log_psi)))

    G_1 = copy(acse.A)
    G1 = G_1.T
    #
    acse.norm = 2*np.linalg.norm(G1,ord=acse.A_norm)
    # if update is FR
    yk = G1 - G0
    #
    #beta_k = vecG_k1.T.dot(vecG_k1)[0,0] / (vecG_k0.T.dot(vecG_k0)[0,0]) #FR
    #beta_k = vecG_k1.T.dot(vecG_k1-vecG_k0)[0,0] / (vecG_k0.T.dot(vecG_k0)[0,0]) #FR
    try:
        if acse._cg_update=='FR': #Fletcher, Reeves, 1964
            beta = G1.T.dot(G1)[0,0]/G0.T.dot(G0)[0,0]
        elif acse._cg_update=='HS': # 
            beta = G1.T.dot(yk)[0,0]/p_k.T.dot(yk)[0,0]
        elif acse._cg_update in ['PR+','PR']:
            beta = G1.T.dot(yk)[0,0]/G0.T.dot(G0)[0,0]
            if acse._cg_update=='PR+':
                beta = max(beta,0)
        elif acse._cg_update=='HZ':
            b1 = (yk-(2*p_k*yk.T.dot(yk)[0,0])/(p_k.T.dot(yk)[0,0]))
            b2 = G1*(1/(y_k.T.dot(pk))[0,0])
            beta = b1.T.dot(b2)[0,0]
        elif acse._cg_update=='Yuan':
            pass
        else:
            raise OptimizationError('Incorrect conjugate-gradient update: {}'.format(acse._cg_update))
    except Exception as e:
        beta = 0
    print('Beta: {:.12f}'.format(beta))
    num = abs(G1.T.dot(G0)[0,0])
    den = (G1.T.dot(G1)[0,0])
    print('Overlap: {:.12f}'.format(num/den))
    if num/den>0.5:
        print('Too much overlap, setting beta = 0.')
        beta = 0
    acse.p = -G1 + beta*copy(acse.p)
    #
    #
    #y_k = vecG_k1 - vecG_k0
    #s_k = alp*p_k  
    #
    #if acse.total.iter==0 and not acse._limited:
    #    normalize = (y_k.T.dot(s_k)[0,0])/(y_k.T.dot(y_k)[0,0])
    #    print('B inv normalization: {}'.format(normalize))
    #    acse.B_k0 *= 1/normalize
    #    acse.Bi_k0*= normalize


