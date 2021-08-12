from hqca.tools import *


'''
Making some fermionic operator functions which can bring a quick representation
'''



def spin_projected(alpha,beta,N=1):
    Sz = Operator()
    for a in alpha:
        Sz+= FermiString(
                coeff=0.5,
                indices=[a,a],
                ops='+-',
                N=N)
    for b in beta:
        Sz+= FermiString(
                coeff=-0.5,
                indices=[b,b],
                ops='+-',
                N=N)
    return Sz

def number(orbitals=None,N=8):
    n = Operator()
    if type(orbitals)==type(None):
        for i in range(N):
            n+= FermiString(
                    coeff=1,
                    indices=[i,i],
                    ops='+-',
                    N=N)
    else:
        for i in orbitals:
            n+= FermiString(
                    coeff=1,
                    indices=[i,i],
                    ops='+-',
                    N=N)
    return n

def spin_plus(spatial,alpha,beta,N=8):
    Sp = Operator()
    for s in spatial:
        a,b  =alpha[s],beta[s]
        Sp+= FermiString(
                coeff=1,
                indices=[a,b],
                N=N,
                ops='+-'
                )
    return Sp

def spin_minus(spatial,alpha,beta,N=8):
    Sm = Operator()
    for s in spatial:
        a,b  =alpha[s],beta[s]
        Sm+= FermiString(
                coeff=1,
                indices=[b,a],
                N=N,
                ops='+-'
                )
    return Sm

def partial_projected_spin(spatial,alpha,beta,N=8):
    Sz = Operator()
    for s in spatial:
        a,b  =alpha[s],beta[s]
        Sz+= FermiString(
                coeff=0.5,
                indices=[a,a],
                N=N,
                ops='+-'
                )
        Sz+= FermiString(
                coeff=-0.5,
                indices=[b,b],
                N=N,
                ops='+-'
                )
    return Sz

def total_spin_squared(spatial,alpha,beta,N=8):
    Sz = partial_projected_spin(spatial,alpha,beta,N=N)
    Sp = spin_plus(spatial,alpha,beta,N=N)
    Sm = spin_minus(spatial,alpha,beta,N=N)
    return Sp*Sm + Sz*Sz - Sz

