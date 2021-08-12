import numpy as np
from sympy import symbols,numbered_symbols
from hqca.tools import *
from hqca.operators import *
from hqca.core import *
from hqca.vqe._ansatz import *

# need a function which generates an operators 



def getUCCAnsatz(
        quantstore,
        singles=True,
        doubles=True,
        triples=False,
        quadruples=False,
        verbose=False,
        **kw):
    alp_occ = []
    alp_vir = []
    bet_occ = []
    bet_vir = []
    for a in quantstore.groups[0]:
        if a in quantstore.initial:
            alp_occ.append(a)
        else:
            alp_vir.append(a)
    for b in quantstore.groups[1]:
        if b in quantstore.initial:
            bet_occ.append(b)
        else:
            bet_vir.append(b)
    # single excitations    
    ucc = Operator()
    parameters = []
    if verbose:
        print('Preparing unitary coupled cluster operator...')
    if doubles:
        T = numbered_symbols('t')
        indices = []
        for i in alp_occ:
            for k in alp_occ:
                if k<=i:
                    continue
                for l in alp_vir:
                    for j in alp_vir:
                        if j<=l:
                            continue
                        indices.append([i,k,l,j])
        for i in bet_occ:
            for k in bet_occ:
                if k<=i:
                    continue
                for l in bet_vir:
                    for j in bet_vir:
                        if j<=l:
                            continue
                        indices.append([i,k,l,j])
        for i in alp_occ:
            for k in bet_occ:
                for l in bet_vir:
                    for j in alp_vir:
                        if j<=i:
                            continue
                        indices.append([i,k,l,j])
        for (i,k,l,j) in indices:
            t = next(T)
            ucc+= FermiString(
                    coeff=t,
                    indices=[i,k,l,j],
                    ops='++--',
                    symbolic=True,
                    N=quantstore.dim,
                    )
            ucc+= FermiString(
                    coeff=-t,
                    indices=[i,k,l,j],
                    ops='--++',
                    symbolic=True,
                    N=quantstore.dim,

                    )
            parameters.append(t)
    if singles:
        S = numbered_symbols('s')
        indices = []
        for i in alp_occ:
            for j in alp_vir:
                indices.append([i,j])
        for i in bet_occ:
            for j in bet_vir:
                indices.append([i,j])
        for (i,j) in indices:
            s = next(S)
            ucc+= FermiString(
                    coeff=s,
                    indices=[i,j],
                    ops='+-',
                    symbolic=True,
                    N=quantstore.dim,
                    )
            ucc+= FermiString(
                    coeff=s,
                    indices=[i,j],
                    ops='-+',
                    symbolic=True,
                    N=quantstore.dim,
                    )
            parameters.append(s)
    elif triples or quadruples:
        raise AnsatzError
    if verbose:
        print('Unitary coupled cluster operator: ')
        print(ucc)
        print('Transforming...')
    qub = ucc.transform(quantstore.transform)
    return VariationalAnsatz(qub,parameters)


