import subprocess
from functools import reduce
import pickle
import os
import numpy as np
from numpy import conj as con
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import time
import timeit
import sys
from math import pi
np.set_printoptions(precision=6,suppress=True)
from hqca.tools.QuantumFramework import build_circuits,run_circuits,Construct
from hqca.tools.QuantumFramework import wait_for_machine
try:
    from hqca.tools import Functions as fx
    from hqca.tools import Chem as chem
    from hqca.tools import RDMFunctions as rdmf
except ImportError:
    import Functions as fx
    import Chem as chem
    import RDMFunctions as rdmf

def build_2e_2rdm(
        Store,
        nocc,
        idx,
        signs=None,
        pr_m=True,
        ):
    '''
    builds and returns 2rdm for a wavefunction in the NO basis
    '''
    if type(signs)==type(None):
        signs = [1]*N//2
    N = nocc.shape[0]
    Ne_tot = np.sum(nocc)
    wf = {}
    for i in range(0,N//2):
        term = '0'*(i)+'1'+'0'*(N//2-i-1)
        term+= term
        val = np.sqrt(max(0,(nocc[idx[2*i]]+nocc[idx[2*i+1]])/2))
        wf[term]=(signs[i])*val
    wf = fx.extend_wf(wf,
            Store.Norb_tot,
            Store.Nels_tot,
            Store.alpha_mo,
            Store.beta_mo)
    rdm2 = rdmf.build_2rdm(
            wf,
            Store.alpha_mo,
            Store.beta_mo)
    return wf,rdm2

def energy_eval_rdm(
        para,
        Store,
        QuantStore
        ):
    '''
    Energy evaluation for single shot quantum computer where we measure the full
    1-RDM. Phase cna be assigned with some 2-RDM values. 
    '''

    if Store.pr_m>1:
        print('Parameters, degrees: ')
        print(para)
    spin_mapping = QuantStore.spin_mapping
    unrestrict=False
    if spin_mapping=='spatial':
        para = para.tolist()
        para = para + para
    if QuantStore.use_radians==False:
        para = [i*pi/180 for i in para]
    else:
        para = [i*pi for i in para]
    QuantStore.parameters = para
    q_circ,qc_list = build_circuits(QuantStore)
    qc_obj = run_circuits(
            q_circ,
            qc_list,
            QuantStore
            )
    proc = Construct(
            qc_obj,
            QuantStore)
    rdm1 = proc.rdm1
    if Store.pr_m>1:
        print('1RDM from Quantum Computer: ')
        print(rdm1)
        print('Imaginary components:')
        print(np.imag(rdm1))
    if spin_mapping=='spin-free':
        unrestrict=True
        noccs,norbs = np.linalg.eig(rdm1)
        idx = noccs.argsort()[::-1]
        if Store.pr_m>1:
            print('Natural occupations:')
            print(noccs)
            print('Natural orbitals:')
            print(norbs)
        noccs_sort = noccs[idx]
        norbs_sort = norbs[:,idx]
        nora = np.zeros(norbs.shape,dtype=np.complex_)
        norb = np.zeros(norbs.shape,dtype=np.complex_)
        for i in range(0,len(noccs)):
            if i in Store.alpha_mo['active']:
                nora[:,i]=norbs[:,i]
            elif i in Store.beta_mo['active']:
                norb[:,i]=norbs[:,i]
    else:
        if Store.pr_m>1:
            print(rdm1)
        Nso = rdm1.shape[0]
        rdma = rdm1[0:Nso//2,0:Nso//2]
        rdmb = rdm1[Nso//2:,Nso//2:]

        noca,nora = np.linalg.eig(rdma)
        idxa = noca.argsort()[::-1]
        noca = noca[idxa]
        nora = nora[:,idxa]

        nocb,norb = np.linalg.eig(rdmb)
        idxb = nocb.argsort()[::-1]
        nocb = nocb[idxb]
        norb = norb[:,idxb]
        if Store.pr_m>2:
            print('Spin orbitals: ')
            print('Alpha:')
            print(nora)
            print('Beta:')
            print(norb)


        noccs,norbs = np.linalg.eig(rdm1)
        idx = noccs.argsort()[::-1]
        norbs = norbs[:,idx]
        if Store.pr_m>1:
            print('Natural ocupations: ')
            print('alpha: {}'.format(noca))
            print('beta: {}'.format(nocb))
    wf,rdm2 = build_2e_2rdm(Store,noccs,idx,Store.pr_m)
    if spin_mapping=='spin-free':
        rdm2 = rdmf.rotate_2rdm_unrestricted(
                rdm2,
                con(norbs.T),
                Store.alpha_mo,
                Store.beta_mo)
    else:
        rdm2 = rdmf.rotate_2rdm(rdm2,
                con(nora.T),
                con(norb.T),
                Store.alpha_mo,
                Store.beta_mo,
                Store.s2s,
                region='active')
    rdm1 = rdmf.check_2rdm(rdm2,2)
    if spin_mapping=='default':
        rdm2t = rdmf.switch_alpha_beta(rdm2,
                Store.alpha_mo,
                Store.beta_mo)
        #rdm2 = 0.5*rdm2t + 0.5*rdm2
        rdm1 = rdmf.check_2rdm(rdm2,2)
        noccs,norbs = np.linalg.eig(rdm1)
    rdm2 = fx.contract(rdm2)
    E1 = reduce(np.dot, (Store.ints_1e,rdm1)).trace()
    E2 = reduce(np.dot, (Store.ints_2e,rdm2)).trace()
    E_t = np.real(E1+0.5*E2+Store.E_ne)
    Store.opt_update_rdm2(
            E_t,
            rdm2,
            para
            )
    if Store.pr_m>1:
        print('Energy: {}'.format(E_t))
        print('')
    return E_t


def energy_eval_nordm(
        para,
        Store,
        QuantStore
        #para,
        #wf_mapping,
        #algorithm,
        #method='stretch',
        #pr_m=0,
        #triangle=None,
        #store='default',
        #**kwargs
        ):
    '''
    Energy evaluation for the natural orbital approach.
    Has several different methods, which use similar pinning principals. 
    Main difference is between classical and quantum computer algorithms. 
    '''
    if Store.pr_m>1:
        print('Parameters, degrees: ')
        print(para)
    spin_mapping = QuantStore.spin_mapping
    unrestrict=False
    if spin_mapping=='spatial':
        para = para.tolist()
        para = para + para
    if QuantStore.use_radians==False:
        para = [i*pi/180 for i in para]
    else:
        para = [i*pi for i in para]
    QuantStore.parameters = para
    q_circ,qc_list = build_circuits(QuantStore)
    qc_obj = run_circuits(
            q_circ,
            qc_list,
            QuantStore
            )
    proc = Construct(
            qc_obj,
            QuantStore)
    rdm1 = proc.rdm1
    if Store.pr_m>1:
        print('1RDM from Quantum Computer: ')
        print(rdm1)
        print('Imaginary components:')
        print(np.imag(rdm1))
    if QuantStore.method=='generalized':
        pass
    elif QuantStore.method=='carlson-keller':
        # performing natural orbital approach for a 2e system
        if spin_mapping=='default':
            proc.find_signs()
            if Store.pr_m>1:
                print(rdm1)
            Nso = rdm1.shape[0]
            rdma = rdm1[0:Nso//2,0:Nso//2]
            rdmb = rdm1[Nso//2:,Nso//2:]

            noca,nora = np.linalg.eig(rdma)
            idxa = noca.argsort()[::-1]
            noca = noca[idxa]
            nora = nora[:,idxa]

            nocb,norb = np.linalg.eig(rdmb)
            idxb = nocb.argsort()[::-1]
            nocb = nocb[idxb]
            norb = norb[:,idxb]
            if Store.pr_m>2:
                print('Spin orbitals: ')
                print('Alpha:')
                print(nora)
                print('Beta:')
                print(norb)
            noccs,norbs = np.linalg.eig(rdm1)
            idx = noccs.argsort()[::-1]
            norbs = norbs[:,idx]
            if Store.pr_m>1:
                print('Natural ocupations: ')
                print('alpha: {}'.format(noca))
                print('beta: {}'.format(nocb))
        wf,rdm2 = build_2e_2rdm(
                Store,noccs,
                idx,proc.signs,
                Store.pr_m)
        if spin_mapping=='spin-free':
            rdm2 = rdmf.rotate_2rdm_unrestricted(
                    rdm2,
                    con(norbs.T),
                    Store.alpha_mo,
                    Store.beta_mo)
        else:
            rdm2 = rdmf.rotate_2rdm(rdm2,
                    con(nora.T),
                    con(norb.T),
                    Store.alpha_mo,
                    Store.beta_mo,
                    Store.s2s,
                    region='active')
        rdm1 = rdmf.check_2rdm(rdm2,2)
        if spin_mapping=='sdefault':
            rdm2t = rdmf.switch_alpha_beta(rdm2,
                    Store.alpha_mo,
                    Store.beta_mo)
            #rdm2 = 0.5*rdm2t + 0.5*rdm2
            rdm1 = rdmf.check_2rdm(rdm2,2)
            noccs,norbs = np.linalg.eig(rdm1)
    elif QuantStore.method=='borland-dennis':
        on, onv = np.linalg.eig(rdm1)
        R = len(Store.ints_1e)
        N = R//2
        p = len(para)
        if p==2:
            p1 = para[0]
            p2 = para[1]
            P1 = (para[0]+45)//90
            P2 = (para[1]+45)//90
            p1 = (p1+45)%90 - 45
            p2 = (p2+45)%90 - 45
            p1 = p1*((-1)**P1)
            p2 = p2*((-1)**P2)
            theta = p1*(np.pi/180)
            phi   = p2*(np.pi/180)
        sgn1,sgn2,sgn3 = 1,1,1
        if theta<0:
            if (phi>=-theta and phi>=0):
                sgn3=-1
            else:
                sgn2=-1
        if phi<0:
            if (-phi>=theta and theta>=0):
                sgn2=-1
            else:
                sgn3=-1
        on.sort()
        m_on = on[::-1]
        m_ON = np.matrix([[m_on[0],m_on[1],m_on[2],1]])
        Tri  = triangle.map([m_on[0],m_on[1],m_on[2]])
        r_ON = np.dot(Tri,m_ON.T)
        r_on = np.array([r_ON[0,0],r_ON[1,0],r_ON[2,0]])
        if Store.pr_m>1:
            print('Measure ON: {}'.format(m_ON))
            print('Rotated ON: {}'.format(r_ON.T))
        nrdm  = np.diag(
                [   r_ON[0,0],
                    r_ON[1,0],
                    r_ON[2,0],
                    1-r_ON[2,0],
                    1-r_ON[1,0],
                    1-r_ON[0,0]
                    ]
                )
        alp,bet,gam,dist = rdmf.project_gpc(nrdm,s1=sgn1,s2=sgn2,s3=sgn3)
        if Store.pr_m>1:
            print('Parameters: \n{}'.format(para))
            print('Fitted wavefunction: ')
            print('Alpha: {}, Beta: {}, Gamma: {}'.format(alp,bet,gam))
            print('Distance from GPC plane: {}'.format(dist))
        wf = rdmf.wf_BD(alp,bet,gam)
        wf = fx.map_wf(wf,wf_mapping)
        wf = fx.extend_wf(
                wf,
                Store.Norb_tot,
                Store.Nels_tot,
                Store.alpha_mo,
                Store.beta_mo)
        if type(Store.rdm2)==type(None):
            region='full'
            rdm2 = rdmf.build_2rdm(
                    wf,
                    alpha=Store.alpha_mo,
                    beta=Store.beta_mo,
                    region=region)
        else:
            region='active'
            rdm2 = rdmf.build_2rdm(
                    wf,
                    alpha=Store.alpha_mo,
                    beta=Store.beta_mo,
                    region=region,
                    rdm2=Store.rdm2.copy())
        rdm1 = rdmf.check_2rdm(rdm2,Store.Nels_tot)
        np.set_printoptions(precision=5)
        norb = Store.Norb_tot
        if method=='diagnostic':
            on, onv = np.linalg.eig(rdm1)
            on.sort()
            p_on = on[::-1]
            p_ON = np.matrix([[on[0],on[1],on[2],1]])
            E_h1 = np.dot(store.ints_1e,rdm1).trace()
            E_h2 = 0.5*np.dot(store.ints_2e,rdm2.T).trace()
            E_t = np.real(E_h1+E_h2+store.E_ne)
            return m_on[0:3],r_on[0:3],p_on[0:3],E_t
    elif method=='classical-diagnostic':
        N = store.Norb_as//2
        if N==3:
            Theta,Phi = (para[0]+45)%90 - 45, (para[1]+45)%90 - 45
            pt,pp = (para[0]+45)//90, (para[1]+45)//90
            Theta = ((para[0]+45)%90 - 45)*((-1)**pt)
            Phi = ((para[1]+45)%90 - 45)*((-1)**pp)
            theta,phi = (np.pi/180)*Theta,(np.pi/180)*Phi
            gam = np.sin(theta)*np.sin(phi)
            bet = np.sin(theta)*np.cos(phi)
            alp = np.cos(theta)
            if abs(bet)<abs(gam):
                bet,gam = gam,bet
            if abs(alp)<abs(bet):
                alp,bet = bet,alp
            if abs(bet)<abs(gam):
                bet,gam = gam,bet
            wf = rdmf.wf_BD(alp,bet,gam) # generate wavefunction with BD
            wf = rdmf.wf_BD(alp,bet,gam) # generate wavefunction with BD
        wf = fx.map_wf(wf,wf_mapping) # map wavefunction to Hamiltonian wf
        wf = fx.extend_wf(
                wf,
                store.Norb_tot,
                store.Nels_tot,
                store.alpha_mo,
                store.beta_mo)
        if type(store.rdm2)==type(None):
            region='full'
            rdm2 = rdmf.build_2rdm(
                    wf,
                    alpha=Store.alpha_mo,
                    beta=Store.beta_mo,
                    region=region)
        else:
            region='active'
            rdm2 = rdmf.build_2rdm(
                    wf,
                    alpha=Store.alpha_mo,
                    beta=Store.beta_mo,
                    region=region,
                    rdm2=Store.rdm2.copy())
        rdm1 = rdmf.check_2rdm(rdm2,Store.Nels_tot) # from that, build the spin 1RDM
        on, onv = np.linalg.eig(rdm1)
        on.sort()
        on = on[A::-1]
        if on[0]+on[1]-on[2]-1 <= -0.2:
            if Store.pr_m>0:
                print(wf)
                print(on)
        rdm2 = fx.contract(rdm2) # reshape to ik form
        E_h1 = np.dot(store.ints_1e,rdm1).trace()      # take trace of 1RDM*1e int
        E_h2 = 0.5*np.dot(store.inits_2e,rdm2.T).trace()  # take trace of 2RDM*2e int
        E_t = np.real(E_h1+E_h2+store.E_ne)
        return on[0:3], E_t
    elif method=='classical-default':
        N = store.Norb_as
        if N==3:
            pt = (para[0]+45)//90
            pp = (para[1]+45)//90
            Theta = ((para[0]+45)%90 - 45)*((-1)**pt)
            Phi = ((para[1]+45)%90 - 45)*((-1)**pp)
            theta = (np.pi/180)*Theta
            phi   = (np.pi/180)*Phi
            gam = np.sin(theta)*np.sin(phi)
            bet = np.sin(theta)*np.cos(phi)
            alp = np.cos(theta)
            wf = rdmf.wf_BD(alp,bet,gam) # generate wavefunction with BD
        elif N==4:
            theta = (np.pi/180)*para[0] # input degrees, output radians
            phi   = (np.pi/180)*para[1] # 
            psi   = (np.pi/180)*para[1] # 
            gam = np.sin(theta)*np.cos(phi)
            bet = np.sin(theta)*np.sin(phi)
            alp = np.cos(theta)
            wf = rdmf.wf_BD(alp,bet,gam) # generate wavefunction with BD
        wf = fx.map_wf(wf,wf_mapping) # map wavefunction to Hamiltonian wf
        wf = fx.extend_wf(
                wf,
                store.Norb_tot,
                store.Nels_tot,
                store.alpha_mo,
                store.beta_mo)
        if type(store.rdm2)==type(None):
            region='full'
            rdm2 = rdmf.build_2rdm(
                    wf,
                    alpha=store.alpha_mo,
                    beta=store.beta_mo,
                    region=region)
        else:
            region='active'
            rdm2 = rdmf.build_2rdm(
                    wf,
                    alpha=store.alpha_mo,
                    beta=store.beta_mo,
                    region=region,
                    rdm2=store.rdm2.copy())
        rdm1 = rdmf.check_2rdm(rdm2,store.Nels_tot) # from that, build the spin 1RDM
    else:
        sys.exit('Not configured yet. Goodbye!')
    rdm2 = fx.contract(rdm2)
    E_h1 = np.dot(Store.ints_1e,rdm1).trace()
    E_h2 = 0.5*np.dot(Store.ints_2e,rdm2.T).trace()
    E_t = np.real(E_h1+E_h2+Store.E_ne)
    if Store.pr_m>1:
        print('One Electron Energy: {}'.format(E_h1))
        print('Two Electron Energy: {}'.format(np.real(E_h2)))
        print('Nuclear Repulsion Energy: {}'.format(Store.E_ne))
        print('Total Energy: {} Hartrees'.format(E_t))
        print('Trace of the 1-RDM: {}'.format(rdm1.trace()))
        print('----------')
    Store.opt_update_wf(E_t,wf,para)
    if type(Store.rdm2)==type(None):
        Store.update_rdm2()
    if Store.pr_m>1:
        print('Energy: {}'.format(E_t))
        print('')
    return E_t

