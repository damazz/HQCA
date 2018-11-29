import subprocess
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import time
import timeit
import sys
np.set_printoptions(precision=6,suppress=True)
from hqca.tools.QuantumFramework import build_circuits,run_circuits,construct
from hqca.tools.QuantumFramework import wait_for_machine
try:
    from hqca.tools import Functions as fx
    from hqca.tools import Chem as chem
    from hqca.tools import RDMFunctions as rdmf
except ImportError:
    import Functions as fx
    import Chem as chem
    import RDMFunctions as rdmf

def energy_eval_rdm(
        para,
        method,
        store,
        ec=False,
        triangle=False,
        verbose=True,
        S2=0,
        **kwargs
        ):
    '''
    Energy evaluation for single shot quantum computer where we measure the full
    1-RDM. Phase cna be assigned with some 2-RDM values. 
    '''
    def build_2e_2rdm(
            store,
            nocc,
            norb,
            verbose=True,
            S2=0
            ):
        '''
        builds and returns 2rdm for a wavefunction in the NO basis
        '''
        Nso = nocc.shape[0]
        Ne_tot = nocc.trace()
        wf = {}
        for i in range(0,Nso//2):
            if S2==0:
                if verbose:
                    print('Difference in alpha/beta occupations')
                    print(nocc[2*i]-nocc[2*i+1])
                term ='0'*(i)+'1'+'0'*(Nso//2-i-1)
                term += term
                wf[term]=(nocc[2*i]+nocc[2*i+1])/2
        wf = fx.extend_wf(wf,
                store.Norb_tot,
                store.Nels_tot,
                store.alpha,
                store.beta)
        rdm2 = rdmf.build_2rdm(
                wf,
                store.alpha,
                store.beta)
        return rdm2
    kwargs['store']=store
    kwargs['verbose']=verbose
    qc,qcl,q = build_circuits(**kwargs)
    kwargs['qb2so']=q
    qco = run_circuits(qc,qcl,**kwargs)
    rdm1 = construct(qco,**kwargs)
    Nso = rdm1.shape[0]
    rdma = rdm1[0:Nso,0:Nso]
    rdmb = rdm1[Nso:,Nso:]
    noccs,norbs = rdm1.eig()
    noca,nora = rdm1a.eig()
    nocb,norb = rdm1b.eig()
    idxa = noca.argsort()[::-1]
    idxb = nocb.argsort()[::-1]
    idx = noccs.argsort()[::-1]
    noca = noca[idxa]
    nocb = nocb[idxb]
    noccs = noccs[idx]
    nora = nora[:,idxa]
    norb = norb[:,idxb]
    norbs = norbs[:,idx]
    rdm2 = build_2e_2rdm(store,noccs,norbs,verbose,S2)
    rdm2 = rotate_2rdm(rdm2,
            nora,
            norb,
            store.alpha,
            store.beta,
            store.spin2spac,
            region='active')
    rdm1 = rdmf.check_2rdm(rdm2,2)
    rdm2 = fx.contract(rdm2)
    E1 = reduce(np.dot, (store.ints_1e,rdm1))
    E2 = reduce(np.dot, (store.ints_2e,rdm2))
    return E1+0.5*E2+store.E_ne







def energy_eval_nordm(
        para,
        wf_mapping, 
        algorithm,
        method='stretch',
        print_run=False,
        triangle=None,
        store='default',
        **kwargs
        ):
    kwargs['para']=para
    kwargs['algorithm']=algorithm
    '''
    Energy evaluation for the quantum computer. Has quite alot of inputs, as it
    has to write to the ibmqx module, which need significant input as well.

    In general, takes a ibm algorithm, executes it on a backend, reads the
    output data (1-RDM diagonal, full 1-RDM, or 2-RDM), and using a certain
    method, will generate the appropriate energy.

    Most options are covered in /doc/options, but important ones are:

    method:
        stretch:
            measures the accessible space of the quantum computer by spanning
            thethe limits of the parameters being used, and then maps that onto
            the GPC surface. Optimized mostly for the 3/6 3 qubit case, so might
            not be generalizable.
    '''
    if method in ['stretch','stretched','diagnostic']:
        #data = evaluate(
        #        **kwargs
        #        )
        #on = data.on
        #rdm = data.rdm1
        qc,qcl,q = build_circuits(**kwargs)
        kwargs['qb2so']=q
        qco = run_circuits(qc,qcl,**kwargs)
        rdm = construct(qco,**kwargs)
        on, onv = np.linalg.eig(rdm)
        R = len(store.ints_1e)
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
        if print_run:
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
        if round(dist,10)==0:
            if print_run:
                print('Point on the GPC surface. Proceeding.')
        else:
            if print_run:
                print('Point on the GPC surface. Proceeding.')
        if print_run:
            print('Parameters: \n{}'.format(para))
            print('Fitted wavefunction: ')
            print('Alpha: {}, Beta: {}, Gamma: {}'.format(alp,bet,gam))
            print('Distance from GPC plane: {}'.format(dist))
        wf = rdmf.wf_BD(alp,bet,gam)
        wf = fx.map_wf(wf,wf_mapping)
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
        rdm1 = rdmf.check_2rdm(rdm2,store.Nels_tot)
        np.set_printoptions(precision=5)
        norb = store.Norb_tot
        rdm2 = np.reshape(rdm2,((2*norb)**2,(2*norb)**2))
        if method=='diagnostic':
            on, onv = np.linalg.eig(rdm1)
            on.sort()
            p_on = on[::-1]
            p_ON = np.matrix([[on[0],on[1],on[2],1]])
            E_h1 = np.dot(store.ints_1e,rdm1).trace()
            E_h2 = 0.5*np.dot(store.ints_2e,rdm2.T).trace()
            E_t = np.real(E_h1+E_h2+store.E_ne)
            return m_on[0:3],r_on[0:3],p_on[0:3],E_t
    elif method=='measure_2rdm':
        sys.exit('Not configured yet. Goodbye!')
    elif method=='classical-diagnostic':
        R = len(store.ints_1e)
        N = R//2
        if N==3:
            Theta = (parameters[0]+45)%90 - 45
            Phi = (parameters[1]+45)%90 - 45
            pt = (parameters[0]+45)//90
            pp = (parameters[1]+45)//90
            Theta = ((parameters[0]+45)%90 - 45)*((-1)**pt)
            Phi = ((parameters[1]+45)%90 - 45)*((-1)**pp)


            #theta,phi = (theta_mod,phi_mod
            theta = (np.pi/180)*Theta
            #parameters[0] # input degrees, output radians
            phi   = (np.pi/180)*Phi
            #parameters[1] # 
            #theta = (np.pi/180)*parameters[0] # input degrees, output radians
            #phi   = (np.pi/180)*parameters[1] # 
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
            #print(wf)
        elif N==4:
            theta = (np.pi/180)*parameters[0] # input degrees, output radians
            phi   = (np.pi/180)*parameters[1] # 
            psi   = (np.pi/180)*parameters[1] # 
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
        #print(wf,alp,bet)

        rdm2 = rdmf.build_2rdm(
                wf,
                alpha=alp,
                beta=bet
                ) # build the 2RDM (spin 2RDM)
        rdm1 = rdmf.check_2rdm(rdm2,store.Nels_tot) # from that, build the spin 1RDM
        on, onv = np.linalg.eig(rdm1)
        on.sort()
        on = on[::-1]
        if on[0]+on[1]-on[2]-1 <= -0.2:
            print(wf)
            print(on)
        rdm2 = np.reshape(rdm2,(36,36)) # reshape to ik form
        E_h1 = np.dot(ints_1e_no,rdm1).trace()      # take trace of 1RDM*1e int
        E_h2 = 0.5*np.dot(ints_2e_no,rdm2.T).trace()  # take trace of 2RDM*2e int
        E_t = np.real(E_h1+E_h2+store.E_ne)
        return on[0:3], E_t
    elif method=='classical-default':
        R = len(ints_1e_no)
        N = R//2
        if N==3:
            pt = (parameters[0]+45)//90
            pp = (parameters[1]+45)//90
            Theta = ((parameters[0]+45)%90 - 45)*((-1)**pt)
            Phi = ((parameters[1]+45)%90 - 45)*((-1)**pp)
            theta = (np.pi/180)*Theta
            phi   = (np.pi/180)*Phi
            gam = np.sin(theta)*np.sin(phi)
            bet = np.sin(theta)*np.cos(phi)
            alp = np.cos(theta)
            wf = rdmf.wf_BD(alp,bet,gam) # generate wavefunction with BD
        elif N==4:
            theta = (np.pi/180)*parameters[0] # input degrees, output radians
            phi   = (np.pi/180)*parameters[1] # 
            psi   = (np.pi/180)*parameters[1] # 
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
        rdm2 = rdmf.build_2rdm(
                wf,
                alpha=alp,
                beta=bet
                )
        rdm1 = rdmf.check_2rdm(rdm2,store.Nels_tot) # from that, build the spin 1RDM
    else:
        sys.exit('Not configured yet. Goodbye!')
    E_h1 = np.dot(store.ints_1e,rdm1).trace()
    E_h2 = 0.5*np.dot(store.ints_2e,rdm2.T).trace()
    E_t = np.real(E_h1+E_h2+store.E_ne)
    if print_run:
        print('One Electron Energy: {}'.format(E_h1))
        print('Two Electron Energy: {}'.format(np.real(E_h2)))
        print('Nuclear Repulsion Energy: {}'.format(store.E_ne))
        print('Total Energy: {} Hartrees'.format(E_t))
        print('Trace of the 1-RDM: {}'.format(rdm1.trace()))
        print('----------')
    rdm2=np.reshape(rdm2,(norb*2,norb*2,norb*2,norb*2))
    store.opt_update_wf(E_t,wf,para)
    if type(store.rdm2)==type(None):
        store.update_rdm2()
    return E_t

