import numpy as np
from sympy import symbols,numbered_symbols
from hqca.tools import *


# need a function which generates an operators 



def getUCCAnsatz(
        QuantStore,
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
    for a in QuantStore.groups[0]:
        if a in QuantStore.initial:
            alp_occ.append(a)
        else:
            alp_vir.append(a)
    for b in QuantStore.groups[1]:
        if b in QuantStore.initial:
            bet_occ.append(b)
        else:
            bet_vir.append(b)
    # single excitations    
    qubOp = Operator()
    ferOp = Operator()
    parameters = []
    if verbose:
        print('Preparing unitary coupled cluster operator...')
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
            newOp = FermionicOperator(
                    coeff=s,
                    indices=[i,j],
                    sqOp='+-',
                    symbolic=True,
                    )
            newOp.generateOperators(
                    Nq=QuantStore.Nq,
                    mapping=QuantStore.mapping,
                    **QuantStore._kw_mapping)
            ferOp+=newOp
            qubOp+=newOp.formOperator()
            newOp = FermionicOperator(
                    coeff=s,
                    indices=[i,j],
                    sqOp='-+',
                    symbolic=True,
                    )
            newOp.generateOperators(
                    Nq=QuantStore.Nq,
                    mapping=QuantStore.mapping,
                    **QuantStore._kw_mapping)
            ferOp+=newOp
            qubOp+=newOp.formOperator()
            parameters.append(s)
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
                        indices.append([i,k,l,j])
        for (i,k,l,j) in indices:
            t = next(T)
            newOp = FermionicOperator(
                    coeff=t,
                    indices=[i,k,l,j],
                    sqOp='++--',
                    symbolic=True,
                    )
            newOp.generateOperators(
                    Nq=QuantStore.Nq,
                    mapping=QuantStore.mapping,
                    **QuantStore._kw_mapping)
            ferOp+=newOp
            qubOp+=newOp.formOperator()
            newOp = FermionicOperator(
                    coeff=-t,
                    indices=[i,k,l,j],
                    sqOp='--++',
                    symbolic=True,
                    )
            newOp.generateOperators(
                    Nq=QuantStore.Nq,
                    mapping=QuantStore.mapping,
                    **QuantStore._kw_mapping)
            ferOp+=newOp
            qubOp+=newOp.formOperator()
            parameters.append(t)
    elif triples or quadruples:
        sys.exit('Have not implemented triples or quadruples in UCC.')
    qubOp.clean()
    print('Unitary coupled cluster operator: ')
    print(ferOp)
    print(qubOp)
    return qubOp,parameters

