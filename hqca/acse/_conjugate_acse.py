""" 
Implementation of the ACSE algorithm with a conjugate gradient search direction

For details on the line search, interpolation, 
BFGS, and l-BFGS algorithms, see Nocedal & Wright, chapters 6 and 7. 

"""

from copy import deepcopy as copy
import numpy as np
from hqca.operators import *
from hqca.acse._quant_S_acse import solveqACSE
from hqca.acse._class_S_acse import solvecACSE
from hqca.acse._qubit_A import solvepACSE
from hqca.acse._user_A import findUserA
from hqca.acse._line_search_acse import vector_to_operator,LineSearch,truncation
from hqca.core import *
from warnings import warn


def _conjugate_gradient_step(acse):
    # okay, given gradient information
    # get inverse
    fermi = acse.acse_update in ['q','c']
    user = acse.acse_update=='u'
    if user:
        keys = acse.tomo_S.rdme
    else:
        keys = acse.rdme

    G_0 = copy(acse.A) #this already has a factor of epsilon from either build or previous run
    if len(G_0.shape)>1:
        G0 = np.asarray(G_0).T
    else:
        G0 = np.asarray([G_0]).T

    p_k = acse.p
    if acse._output==2:
        temp = vector_to_operator(
                G0,fermi=fermi,user=user,
                keys=keys,N=acse.QuantStore.dim,
                S_min=acse.S_min
                )
        print(temp)
        for i,k in zip(G0.flatten(),acse.rdme):
            if abs(i)>1e-6:
                print(k,i)
    #
    p_k = truncation(
            p_k,acse.S_thresh_rel,acse.S_min,
            method=acse.S_trunc,
            gradient=G0,
            previous=acse._log_p,
            )
    acse._opt_log = []
    acse._opt_en = []
    #alp,f_star,g_star,err = line_search(acse,p_k,G0,c2=0.5)
    ls = LineSearch(acse)
    ls.run(p_k,G0,c2=0.5)
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
    acse.S = acse.S + P*acse.epsilon
    #print(acse.S)
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
    G_1 = copy(acse.A)
    G1 = G_1.T
    #
    acse.norm = np.linalg.norm(G1,ord=acse.A_norm)/acse.epsilon
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
        elif acse._cg_update=='PRP':
            beta = G1.T.dot(yk)[0,0]/G0.T.dot(G0)[0,0]
        elif acse._cg_update=='HZ':
            b1 = (yk-(2*p_k*yk.T.dot(yk)[0,0])/(p_k.T.dot(yk)[0,0]))
            b2 = G1*(1/(p_k.T.dot(yk)))
            beta = b1.T.dot(b2)[0,0]
        else:
            raise OptimizationError('Incorrect conjugate-gradient update: {}'.format(acse._cg_update))
    except Exception as e:
        beta = 0
    print('Beta: {:.12f}'.format(beta))
    num = abs(G1.T.dot(G0)[0,0])**2
    den = G1.T.dot(G1)[0,0] * G0.T.dot(G0)[0,0]
    print('Overlap: {:.12f}'.format(num/den))
    if num/den>0.25:
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


