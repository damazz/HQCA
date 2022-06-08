"""
BFGS implementation of the ACSE algorithm

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


def _bfgs_step(acse,limited=False):
    # okay, given gradient information
    #  solve B_k p_k = -grad f(x_k)
    fermi = acse.acse_update in ['q','c']
    keys = acse.rdme#+acse.rdme+acse.rdme+acse.rdme
    symm = 4

    G_k0 = copy(acse.A) #this already has a factor of epsilon from either build or previous run
    if len(G_k0.shape)>1:
        vecG_k0 = np.asarray(G_k0).T
    else:
        vecG_k0 = np.asarray([G_k0]).T
    if limited:
        if len(acse._lbfgs_sk)==0:
            p_k = -vecG_k0
        else:
            p_k = -acse._lbfgs_r
    else:
        p_k = -acse.Bi_k0.dot(vecG_k0)

    if acse.psi.closed==1:
        prev = []
    else:
        #prev = acse.log_p[acse.psi.closed:]
        prev = acse._log_psi[acse.psi.closed:]

    if acse.S_thresh_rel>0:
        p_k = truncation(
                p_k,acse.S_thresh_rel,acse.S_min,
                method=acse.trunc_method,
                include=acse.trunc_include,
                gradient=vecG_k0,
                previous=prev,
                keys=keys,
                mapping=acse.rdme_mapping,
                )
    nz = np.nonzero(p_k)
    print('Nonzero elements: {}'.format(len(nz[0])))
    #
    # acse option to include previous terms
    #
    # need to check the angle between p_k and g_k for the line search
    #
    ovlp = p_k.T.dot(vecG_k0)[0,0]
    norm_p = np.linalg.norm(p_k)
    norm_g = np.linalg.norm(vecG_k0)
    print('<gT|p>**2 : {}'.format(ovlp/(norm_p*norm_g)))
    print('norm p, {}, norm g {}'.format(norm_p,norm_g))
    acse._opt_log = []
    acse._opt_en = []

    #
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
    alp = max(0.5,alp)

    ls = LineSearchACSE(acse)
    acse.log_A.append(vecG_k0)
    acse.log_p.append(p_k)
    ls.run(p_k,vecG_k0,alp_1=alp)
    alp = ls.alp_star
    f_star = ls.f_star
    g_star = ls.g_star
    if ls.err:
        acse.total.done = True
        return None
    op = vector_to_operator(p_k*alp,fermi=fermi,keys=keys,N=acse.qs.dim,
            S_min=acse.S_min)
    #if acse.acse_update in ['u']:
    #    P = op
    #else:
    #    P = op.transform(acse.transform_psi)
    P = op
    acse.psi = acse.psi + P
    #
    #
    acse.A = g_star #already has factor of epsilon
    acse.p = p_k*alp # no factor of epsilon
    acse.alp = alp
    #
    #
    # here we relate vectors to elements of psi for the truncation scheme
    if acse.psi.closed==1:
        acse._log_psi = []
    else:
        P_k_bool = np.zeros(p_k.shape)
        for i in range(len(p_k)):
            if abs(p_k[i])>acse.S_min:
                #print(i,keys[i])
                P_k_bool[i]=1
        temp = np.zeros(p_k.shape)
        for i in acse._log_psi[acse.psi.closed:]:
            temp+= i
        # temp will have elements of previouis one
        temp = np.multiply(P_k_bool,temp)
        P_k_bool = np.mod(P_k_bool+temp,2)
        if np.any(P_k_bool):
            acse._log_psi.append(P_k_bool)
        nz = np.nonzero(P_k_bool)
        #print('bool storage')
        #for i in nz[0]:
        #    if P_k_bool[i]>acse.S_min:
        #        print(i,keys[i])


        print('Length log_g: {}'.format(len(acse._log_psi)))

    #acse._get_S()
    G_k1 = copy(acse.A)
    acse.norm = np.linalg.norm(G_k1,ord=acse.A_norm)

    vecG_k1 = G_k1.T
    y_k = vecG_k1 - vecG_k0
    s_k = alp*p_k  # 
    I = np.identity(len(y_k))
    if acse.total.iter==0 and not limited:
        normalize = (1/1)*(y_k.T.dot(s_k)[0,0])/(y_k.T.dot(y_k)[0,0])
        print('B inv normalization: {}'.format(normalize))
        #acse.B_k0 *= 1/normalize*I
        acse.Bi_k0 = normalize*I
    rho_k = (1/1)/(y_k.T.dot(s_k)[0,0])
    alphaT = (1/1)*(1/s_k.T.dot(y_k)[0,0])
    alpha = (1/1)*(1/y_k.T.dot(s_k)[0,0])
    if not limited:
        S = s_k.dot(s_k.T)
        #beta = -1/s_k.T.dot(acse.B_k0.dot(s_k))[0,0]
        #v = acse.B_k0.dot(s_k)
        print('alpha_k: {}'.format(alphaT))
        #print('beta_k: {}'.format(beta))
        #acse.B_k0 = acse.B_k0 + alpha*y_k.dot(y_k.T) + beta*v.dot(v.T)
        #print('Norm B_k: {}'.format(np.linalg.norm(acse.B_k0)))
        #a1 = s_k.T.dot(y_k)[0,0]
        L = I-rho_k*s_k.dot(y_k.T)
        R = I-rho_k*y_k.dot(s_k.T)
        #a1 = 1*y_k.T.dot(acse.Bi_k0.dot(y_k))[0,0] + 1*s_k.T.dot(y_k)[0,0]
        #b1 = 1*acse.Bi_k0.dot(y_k.dot(s_k.T)) + 1*s_k.dot(y_k.T.dot(acse.Bi_k0))
        acse.Bi_k0 = L.dot(acse.Bi_k0.dot(R)) + rho_k * S
        ##
        #temp = S*a1*(alphaT**2) - b1*alphaT
        #acse.Bi_k0+= temp
        #acse.Bi_k0-= b1*alpha
        #acse.Bi_k0+= S*alpha
        #acse.Bi_k0+= S*a1*alpha**2
        #print('Norm Bi_k: {}'.format(np.linalg.norm(acse.Bi_k0)))
    else:
        q = copy(vecG_k1)
        alp_is = []
        for i in range(1,min(acse._limited+1,len(acse._lbfgs_sk)+1)):
            s,y,rho = acse._lbfgs_sk[-i],acse._lbfgs_yk[-1],acse._lbfgs_rk[-i]
            alp_i = np.real(rho*s.T.dot(q)[0,0])
            alp_is.append(alp_i)
            q-= alp_i*y
        gamma = 1/(alphaT * y_k.T.dot(y_k))
        #gamma = 1/(acse._lbfgs_rk[-1] *(acse._lbfgs_yk[-1].T.dot(acse._lbfgs_yk[-1])))
        r = copy(q)*gamma
        print('Gamma_k', gamma)
        for i in range(len(acse._lbfgs_rk)):
            s,y,rho = acse._lbfgs_sk[i],acse._lbfgs_yk[i],acse._lbfgs_rk[i]
            beta = rho*y.T.dot(r)[0,0]
            r+= s*(alp_is[-(i+1)]-beta)
        acse._lbfgs_r = r
        acse._lbfgs_rk.append(alpha)
        acse._lbfgs_sk.append(s_k)
        acse._lbfgs_yk.append(y_k)
        acse._lbfgs_rk = acse._lbfgs_rk[-acse._limited:]
        print(acse._lbfgs_rk)
        acse._lbfgs_sk = acse._lbfgs_sk[-acse._limited:]
        acse._lbfgs_yk = acse._lbfgs_yk[-acse._limited:]


