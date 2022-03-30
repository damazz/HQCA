"""
BFGS implementation of the ACSE algorithm

For details on the line search, interpolation, 
BFGS, and l-BFGS algorithms, see Nocedal & Wright, chapters 6 and 7. 

"""

from copy import deepcopy as copy
import numpy as np
from hqca.operators import *
from hqca.acse._quant_S_acse import solveqACSE
from hqca.acse._class_S_acse import solvecACSE
from hqca.acse._qubit_A import findQubitAQuantum
from hqca.acse._user_A import findUserA
from hqca.acse._line_search_acse import vector_to_operator,truncation,LineSearch
from hqca.core import *
from warnings import warn


def _bfgs_step(acse):
    # okay, given gradient information
    #  solve B_k p_k = -grad f(x_k)
    # get inverse
    fermi = acse.acse_update in ['q','c']
    user = acse.acse_update=='u'
    if user:
        keys = acse.tomo_S.rdme
    else:
        keys = acse.rdme

    G_k0 = copy(acse.A) #this already has a factor of epsilon from either build or previous run
    if len(G_k0.shape)>1:
        vecG_k0 = np.asarray(G_k0).T
    else:
        vecG_k0 = np.asarray([G_k0]).T
    if acse._limited:
        if len(acse._lbfgs_sk)==0:
            p_k = -vecG_k0
        else:
            p_k = -acse._lbfgs_r
    else:
        p_k = -acse.Bi_k0.dot(vecG_k0)
    if acse._output==2:
        temp = vector_to_operator(vecG_k0,fermi=fermi,user=user,keys=keys,N=acse.QuantStore.dim,
                S_min=acse.S_min
                )
        #print(vecG_k0.T)
        for i,k in zip(vecG_k0.flatten(),acse.rdme):
            if abs(i)>1e-6:
                print(k,i)
    #for i,k in zip(p_k.flatten(),keys):
    #    if abs(i)>1e-2:
    #        print(k,i)
    #
    #
    #
    #print('Old: ')
    #print(p_k.T)
    p_k = truncation(
            p_k,acse.S_thresh_rel,acse.S_min,
            method=acse.S_trunc,
            gradient=vecG_k0,
            previous=acse._log_p,
            )
    #print('New: ')
    #print(p_k.T)
    #
    # acse option to include previous terms
    #
    # need to check the angle between p_k and g_k for the line search
    #
    ovlp = p_k.T.dot(vecG_k0)[0,0]**2
    norm_p = np.linalg.norm(p_k)
    norm_g = np.linalg.norm(vecG_k0)
    print('<gT|p>**2 : {}'.format(ovlp))
    print('norm p, {}, norm g {}'.format(norm_p,norm_g))
    acse._opt_log = []
    acse._opt_en = []
    #alp,f_star,g_star,err = line_search(acse,p_k,vecG_k0)
    ls = LineSearch(acse)
    ls.run(p_k,vecG_k0)
    alp = ls.alp_star
    f_star = ls.f_star
    g_star = ls.g_star
    if ls.err:
        acse.total.done = True
        return None
    op = vector_to_operator(p_k*alp,fermi=fermi,user=user,keys=keys,N=acse.QuantStore.dim,
            S_min=acse.S_min,)
    if acse.acse_update in ['c', 'q']:
        P = op.transform(acse.QuantStore.transform)
    elif acse.acse_update in ['p']:
        P = op.transform(acse.QuantStore.qubit_transform)
    elif acse.acse_update in ['u']:
        P = op
    acse.S = acse.S + P
    #
    #
    #
    acse.A = g_star #already has factor of epsilon
    acse.p = p_k*alp # no factor of epsilon
    acse._log_p.append(p_k*alp)
    if acse.S.closed==1:
        acse._log_p = []
    elif acse.S.closed==0:
        pass
    else:
        acse._log_p = acse._log_p[acse.S.closed:]

    #acse._get_S()
    G_k1 = copy(acse.A)
    acse.norm = np.linalg.norm(G_k1,ord=acse.A_norm)/acse.epsilon

    vecG_k1 = G_k1.T
    y_k = vecG_k1 - vecG_k0
    s_k = alp*p_k  # 
    if acse.total.iter==0 and not acse._limited:
        normalize = (y_k.T.dot(s_k)[0,0])/(y_k.T.dot(y_k)[0,0])
        print('B inv normalization: {}'.format(normalize))
        acse.B_k0 *= 1/normalize
        acse.Bi_k0*= normalize

    alpha = (1 / y_k.T.dot(s_k)[0,0])
    if not acse._limited:
        alphaT = 1/s_k.T.dot(y_k)[0,0]
        S = s_k.dot(s_k.T)
        beta = -1/s_k.T.dot(acse.B_k0.dot(s_k))[0,0]
        v = acse.B_k0.dot(s_k)
        print('alpha_k: {}'.format(alphaT))
        print('beta_k: {}'.format(beta))
        acse.B_k0 = acse.B_k0 + alpha*y_k.dot(y_k.T) + beta*v.dot(v.T)
        print('Norm B_k: {}'.format(np.linalg.norm(acse.B_k0)))
        a1 = s_k.T.dot(y_k)[0,0]+y_k.T.dot(acse.Bi_k0.dot(y_k))[0,0]
        b1 = acse.Bi_k0.dot(y_k.dot(s_k.T)) + s_k.dot(y_k.T.dot(acse.Bi_k0))
        temp = S*a1*(alphaT**2) - b1*alphaT
        acse.Bi_k0+= temp
        #print('Norm Bi_k: {}'.format(np.linalg.norm(acse.Bi_k0)))
    elif acse._limited:
        q = copy(vecG_k1)
        alp_is = []
        for i in range(1,min(acse._limited+1,len(acse._lbfgs_sk)+1)):
            s,y,rho = acse._lbfgs_sk[-i],acse._lbfgs_yk[-1],acse._lbfgs_rk[-i]
            alp_i = np.real(rho*s.T.dot(q)[0,0])
            alp_is.append(alp_i)
            q-= alp_i*y
        gamma = 1/(alpha * y_k.T.dot(y_k))
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


