import numpy as np
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
    indices = []
    if singles:
        for i in alp_occ:
            for j in alp_vir:
                indices.append([i,j])
        for i in bet_occ:
            for j in bet_vir:
                indices.append([i,j])
    elif triples or quadruples:
        raise AnsatzError
    if doubles:
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
    return VariationalAnsatz(indices=indices)


