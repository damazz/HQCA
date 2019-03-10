import subprocess
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import time
import timeit
from functools import reduce
import sys
np.set_printoptions(precision=6,suppress=True)
from hqca.tools.QuantumFramework import build_circuits,run_circuits,construct
from hqca.tools.QuantumFramework import wait_for_machine
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

def energy_eval_orbitals(
        para,
        Store,
        QuantStore,
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
    ints_2e_n = np.reshape(ints_2e_n,((N*2)**2,(N*2)**2))
    rdm2 = Store.rdm2
    rdm1 = rdmf.check_2rdm(Store.rdm2,Store.Nels_tot)
    rdm2 = np.reshape(rdm2,((2*N)**2,(2*N)**2))
    E_h1 = np.dot(ints_1e_n,rdm1).trace()
    E_h2 = 0.5*np.dot(ints_2e_n,rdm2.T).trace()
    E_t = np.real(E_h1+E_h2+Store.E_ne)
    if pr_s>2:
        print('One Electron Energy: {}'.format(E_h1))
        print('Two Electron Energy: {}'.format(np.real(E_h2)))
        print('Nuclear Repulsion Energy: {}'.format(Store.E_ne))
        print('Total Energy: {} Hartrees'.format(E_t))
        print('----------')
    Store.opt_update_int(E_t,T_a,T_b)
    return E_t
'''
def orbital_energy_gradient_givens(
        para,
        wf, #needs to be in fully mapped out form, i.e. match wf form
        ints_1e_ao,
        ints_2e_ao,
        E_ne,
        mo_coeff_a,
        mo_coeff_b,
        print_run=False,
        store='default',
        **kwargs
        ):

    def f_theta(n_orb,theta,i,j):
        temp = np.identity(n_orb)
        c = np.cos(np.radians(theta))
        s = np.sin(np.radians(theta))
        temp[i,i]=c
        temp[j,j]=c
        temp[i,j]=-s
        temp[j,i]=s
        return temp

    def g_theta(n_orb,theta,i,j):
        temp = np.zeros((n_orb,n_orb))
        c = np.cos(np.radians(theta))
        s = np.sin(np.radians(theta))
        temp[i,i]=-s
        temp[j,j]=-s
        temp[i,j]=-c
        temp[j,i]=c
        return temp

    def ddx_rot(n_orb,parameters,a,b):
        if a<b:
            pass
        else:
            a,b= b,a
        rot_mat = np.identity(n_orb)
        count = 0
        for i in range(0,n_orb):
            for j in range(0,n_orb):
                if i<j:
                    if (i==a and j==b):
                        rot_mat = np.dot(
                            g_theta(
                                n_orb,
                                parameters[count],
                                i,j
                                ),
                            rot_mat
                            )
                    else:
                        rot_mat = np.dot(
                            f_theta(
                                n_orb,
                                parameters[count],
                                i,j
                                ),
                            rot_mat
                            )
                    count+=1 
        return rot_mat

    def rot(n_orb,parameters):
        rot_mat = np.identity(n_orb)
        count = 0
        for i in range(0,n_orb):
            for j in range(0,n_orb):
                if i<j:
                    temp = np.identity(n_orb)
                    c = np.cos((np.radians(parameters[count])))
                    s = np.sin((np.radians(parameters[count])))
                    temp[i,i] = c
                    temp[j,j] = c
                    temp[i,j] = -s
                    temp[j,i] = s
                    rot_mat = np.dot(temp,rot_mat)
                    count+=1 
        return rot_mat


    N = len(ints_1e_ao)
    sys.exit()
    Np = len(para)
    ddE = np.zeros(Np)
    mapping = {}
    count= 0
    for m in range(0,N):
        for n in range(0,N):
            if m<n:
                mapping['{}{}'.format(str(m),str(n))]=count
                count+=1
    rev_map = {v:k for k,v in mapping.items()}
    Ta_f = np.dot(mo_coeff_a,rot(N,para[0:Np//2]))
    Tb_f = np.dot(mo_coeff_b,rot(N,para[Np//2:]))
    for i in range(0,Np):
        a = int(rev_map[i%N][0])
        b = int(rev_map[i%N][1])
        if i<N:
            Ta_g = np.dot(mo_coeff_a,ddx_rot(N,para[0:Np//2],a,b))
            Tb_g = np.zeros((N,N))
        else:
            Ta_g = np.zeros((N,N))
            Tb_g = np.dot(mo_coeff_b,ddx_rot(N,para[Np//2: ],a,b))
        ints_1e_n = chem.gen_spin_1ei_lr(
                ints_1e_ao,
                Ta_f.T,Ta_g.T,
                Tb_f.T,Tb_g.T,
                alpha=store.alpha,beta=store.beta,
                spin2spac=store.s2s
                )
        ints_1e_n+= chem.gen_spin_1ei_lr(
                ints_1e_ao,
                Ta_g.T,Ta_f.T,
                Tb_g.T,Tb_f.T,
                alpha=store.alpha,beta=store.beta,
                spin2spac=store.s2s
                )

        ints_2e_n = chem.gen_spin_2ei_lr(
                ints_2e_ao,
                Ta_g.T,Ta_f.T,Ta_f.T,Ta_f.T,
                Tb_g.T,Tb_f.T,Tb_f.T,Tb_f.T,
                alpha=store.alpha,beta=store.beta,
                spin2spac=store.s2s
                )
        ints_2e_n+= chem.gen_spin_2ei_lr(
                ints_2e_ao,
                Ta_f.T,Ta_g.T,Ta_f.T,Ta_f.T,
                Tb_f.T,Tb_g.T,Tb_f.T,Tb_f.T,
                alpha=store.alpha,beta=store.beta,
                spin2spac=store.s2s
                )
        ints_2e_n+= chem.gen_spin_2ei_lr(
                ints_2e_ao,
                Ta_f.T,Ta_f.T,Ta_g.T,Ta_f.T,
                Tb_f.T,Tb_f.T,Tb_g.T,Tb_f.T,
                alpha=store.alpha,beta=store.beta,
                spin2spac=store.s2s
                )
        ints_2e_n+= chem.gen_spin_2ei_lr(
                ints_2e_ao,
                Ta_f.T,Ta_f.T,Ta_f.T,Ta_g.T,
                Tb_f.T,Tb_f.T,Tb_f.T,Tb_g.T,
                alpha=store.alpha,beta=store.beta,
                spin2spac=store.s2s
                )
        ints_2e_n = np.reshape(ints_2e_n,((N*2)**2,(N*2)**2))
        rdm2=store.rdm2
        rdm2 = np.reshape(rdm2,((2*norb)**2,(2*norb)**2)) 
        rdm1 = rdmf.check_2rdm(rdm2,store.Nels_tot) # from that, build the spin 1RDM
        rdm2 = np.reshape(rdm2,((N*2)**2,(N*2)**2)) # reshape to ik form
        E_h1 = np.dot(ints_1e_n,rdm1).trace()
        E_h2 = 0.5*np.dot(ints_2e_n,rdm2.T).trace()
        ddE[i] = np.real(E_h1+E_h2)*(np.pi/180)
    return ddE
'''
