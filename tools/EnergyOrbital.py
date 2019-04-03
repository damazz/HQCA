import subprocess
from math import pi
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import time
import timeit
from functools import reduce
import sys
np.set_printoptions(precision=8,suppress=False)
try:
    from hqca.tools import Functions as fx
    from hqca.tools import Chem as chem
    from hqca.tools import RDMFunctions as rdmf
    from hqca.tools import EnergyFunctions as enf
except ImportError:
    import Functions as fx
    import Chem as chem
    import RDMFunctions as rdmf
    import EnergyFunctions as enf


def f_givens(n_orb,theta,i,j):
    temp = np.identity(n_orb)
    c = np.cos(theta)
    s = np.sin(theta)
    temp[i,i]=c
    temp[j,j]=c
    temp[i,j]=-s
    temp[j,i]=s
    return temp

def g_givens(n_orb,theta,i,j):
    temp = np.zeros((n_orb,n_orb))
    c = np.cos(theta)
    s = np.sin(theta)
    temp[i,i]=-s
    temp[j,j]=-s
    temp[i,j]=-c
    temp[j,i]=c
    return temp

def energy_eval_orbitals(
        para,
        Store,
        QuantStore,
        diag=False
        ):
    '''
    Energy function for constructing a rotated electron integral. 
    '''
    pr_s = Store.pr_s
    spin_mapping = QuantStore.spin_mapping
    N = len(Store.ints_1e_ao)
    Np = len(para)
    T_a = reduce( np.dot, (
            Store.T_alpha,
            enf.rotation_parameter_generation(
                spin_mo=Store.alpha_mo,
                region='active',
                output='matrix',
                para=para[:Np//2]
                ).T
            )
            )
    if spin_mapping=='restricted':
        T_b = T_a.copy()
    elif spin_mapping in ['default','unrestricted']:
        T_b = reduce(np.dot, (
                Store.T_beta,
                enf.rotation_parameter_generation(
                    spin_mo=Store.beta_mo,
                    region='active',
                    output='matrix',
                    para=para[Np//2:]
                    ).T
                )
                )
    tic = timeit.default_timer()
    ints_1e_n = chem.gen_spin_1ei(
            Store.ints_1e_ao,
            T_a.T,
            T_b.T,
            alpha=Store.alpha_mo,
            beta=Store.beta_mo,
            region='active',
            spin2spac=Store.s2s,
            new_ei = Store.ints_1e.copy()
            )
    if diag:
        return T_a
    ints_2e_n = chem.gen_spin_2ei(
            Store.ints_2e_ao,
            T_a.T,
            T_b.T,
            alpha=Store.alpha_mo,
            beta=Store.beta_mo,
            region='active',
            spin2spac=Store.s2s,
            new_ei = np.reshape(Store.ints_2e.copy(),(2*N,2*N,2*N,2*N)),
            )
    toc = timeit.default_timer()
    if pr_s>3:
        print('Time to rotate electron integrals: {}'.format(toc-tic))
    ints_2e_n = fx.contract(ints_2e_n)
    E_h1 = np.dot(ints_1e_n,Store.rdm1).trace()
    E_h2 = 0.5*np.dot(ints_2e_n,Store.rdm2).trace()
    E_t = np.real(E_h1+E_h2+Store.E_ne)
    if pr_s>2:
        print('One Electron Energy: {}'.format(E_h1))
        print('Two Electron Energy: {}'.format(np.real(E_h2)))
        print('Nuclear Repulsion Energy: {}'.format(Store.E_ne))
        print('Total Energy: {} Hartrees'.format(E_t))
        print('----------')
    Store.opt_update_int(para,E_t,T_a,T_b)
    return E_t


def orbital_en_grad_numerical(
        para,
        Store,
        QuantStore):
    Np = len(para)
    dE = np.zeros(Np)
    dt = 0.00001
    for i in range(0,Np):
        temp = np.zeros(Np)
        temp[i] = dt
        plus = energy_eval_orbitals(
                para+temp,
                Store,
                QuantStore)
        minus = energy_eval_orbitals(
                para-temp,
                Store,
                QuantStore)
        dE[i]= (plus-minus)/(2*dt)
    return dE




def orbital_energy_gradient_givens(
        para,
        Store,
        QuantStore
        ):
    def f_theta(n_orb,theta,i,j):
        temp = np.identity(n_orb)
        c = np.cos(theta)
        s = np.sin(theta)
        temp[i,i]=c
        temp[j,j]=c
        temp[i,j]=-s
        temp[j,i]=s
        return temp

    def g_theta(n_orb,theta,i,j):
        temp = np.zeros((n_orb,n_orb))
        c = np.cos(theta)
        s = np.sin(theta)
        temp[i,i]=-s
        temp[j,j]=-s
        temp[i,j]=-c
        temp[j,i]=c
        return temp

    def ddx_rot(n_orb,parameters,a,b):
        rot_mat = np.identity(n_orb)
        count = 0
        for i in range(0,n_orb):
            for j in range(i+1,n_orb):
                if (i==a and j==b):
                    rot_mat = np.dot(
                        rot_mat,
                        g_theta(
                            n_orb,
                            parameters[count],
                            i,j
                            )
                        )
                else:
                    rot_mat = np.dot(
                        rot_mat,
                        f_theta(
                            n_orb,
                            parameters[count],
                            i,j
                            )
                        )
                count+=1 
        return rot_mat

    def rot(n_orb,parameters):
        rot_mat = np.identity(n_orb)
        count = 0
        for i in range(0,n_orb):
            for j in range(i+1,n_orb):
                temp = np.identity(n_orb)
                c = np.cos(((parameters[count])))
                s = np.sin(((parameters[count])))
                temp[i,i] = c
                temp[j,j] = c
                temp[i,j] = -s
                temp[j,i] = s
                rot_mat = np.dot(rot_mat,temp)
                count+=1
        return rot_mat
    N = len(Store.ints_1e_ao)
    Np = len(para)
    ddE = np.zeros(Np)
    count= 0
    mapping={}
    for m in range(0,N):
        for n in range(0,N):
            if m<n:
                mapping['{}{}'.format(str(m),str(n))]=count
                count+=1
    rev_map = {v:k for k,v in mapping.items()}

    Ta_f = np.dot(Store.T_alpha,rot(N,para[0:Np//2]).T)
    Tb_f = np.dot(Store.T_beta,rot(N,para[Np//2:]).T)
    for i in range(0,Np):
        a = int(rev_map[i%(Np//2)][0])
        b = int(rev_map[i%(Np//2)][1])
        if i<(Np//2):
            Ta_g = np.dot(Store.T_alpha,ddx_rot(N,para[0:Np//2],a,b).T)
            Tb_g = np.zeros((N,N))
        else:
            Ta_g = np.zeros((N,N))
            Tb_g = np.dot(Store.T_beta,ddx_rot(N,para[Np//2: ],a,b).T)
        ints_1e_n = chem.gen_spin_1ei_lr(
                Store.ints_1e_ao,
                Ta_f.T,Ta_g,
                Tb_f.T,Tb_g,
                alpha=Store.alpha_mo,
                beta=Store.beta_mo,
                spin2spac=Store.s2s
                )
        '''
        ints_1e_n += chem.gen_spin_1ei_lr(
                Store.ints_1e_ao,
                Ta_g.T,Ta_f,
                Tb_g.T,Tb_f,
                alpha=Store.alpha_mo,
                beta=Store.beta_mo,
                spin2spac=Store.s2s
                )
        '''
        ints_2e_n =  chem.gen_spin_2ei_lr(
                Store.ints_2e_ao,
                Ta_g.T,Ta_f.T,Ta_f.T,Ta_f.T,
                Tb_g.T,Tb_f.T,Tb_f.T,Tb_f.T,
                alpha=Store.alpha_mo,
                beta=Store.beta_mo,
                spin2spac=Store.s2s
                )
        '''
        ints_2e_n= ints_2e_n + chem.gen_spin_2ei_lr(
                Store.ints_2e_ao,
                Ta_f.T,Ta_g.T,Ta_f.T,Ta_f.T,
                Tb_f.T,Tb_g.T,Tb_f.T,Tb_f.T,
                alpha=Store.alpha_mo,
                beta=Store.beta_mo,
                spin2spac=Store.s2s
                )
        ints_2e_n= ints_2e_n + chem.gen_spin_2ei_lr(
                Store.ints_2e_ao,
                Ta_f.T,Ta_f.T,Ta_g.T,Ta_f.T,
                Tb_f.T,Tb_f.T,Tb_g.T,Tb_f.T,
                alpha=Store.alpha_mo,
                beta=Store.beta_mo,
                spin2spac=Store.s2s
                )
        ints_2e_n= ints_2e_n + chem.gen_spin_2ei_lr(
                Store.ints_2e_ao,
                Ta_f.T,Ta_f.T,Ta_f.T,Ta_g.T,
                Tb_f.T,Tb_f.T,Tb_f.T,Tb_g.T,
                alpha=Store.alpha_mo,
                beta=Store.beta_mo,
                spin2spac=Store.s2s
                )
        '''
        ints_2e_n = fx.contract(ints_2e_n)
        E_h1 = 2*(np.dot(ints_1e_n,Store.rdm1).trace())
        E_h2 = 2*(np.dot(ints_2e_n,Store.rdm2).trace())
        ddE[i] = np.real(E_h1+E_h2)
    return ddE

