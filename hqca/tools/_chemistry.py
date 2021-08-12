import numpy as np
import sys
from pyscf import scf, gto, mcscf, ao2mo
from functools import reduce
import traceback

mss = {0:0,1:1,2:2,3:0,4:1,5:2}

def reorder(rdm1,orbit):
    '''
    Finds the transformation to obtain the Aufbau ordering for spatial orbitals: the spatial orbitals according to the eigenvalues of the 1-RDM (sometimes, diagonalization procedure will swap the orbital ordering). 
    '''
    ordered=False
    T = np.identity(orbit)
    for i in range(0,orbit):
        for j in range(i+1,orbit):
            if rdm1[i,i]>=rdm1[j,j]:
                continue
            else:
                temp= np.identity(orbit)
                temp[i,i] = 0 
                temp[j,j] = 0
                temp[i,j] = -1
                temp[j,i] = 1
                T = np.dot(temp,T)
    return T


def get_spin_ei(
        mol,
        elect,
        orbit,
        orbitals='FCI',
        seed=False,
        seed_mol=None):
    '''
    With a mol object (class, needs to be imported or setup from pyscf),
    runs a mcscf calculation in pyscf and generates the spin transformed 
    electron integrals.

    Also needs the number of electrons and orbitals. Finally, says the 
    order of computation that the orbitals should be generated at. I.e., 
    if orbitals='FCI', then will output the integrals in the full CI 
    natural orbital (1RDM eigenvector) basis. Options should also be 
    included for orbitals='AO' and orbitals='HF'.

    Note, the electron integrals come in dimension: (spin_orbit**2,
    spin_orbit**2), but are processed internally as spatial orbitals:
    (i.e, (orbit,orbit)). 2-RDM comes out in a similar way. Something like,
    ik*ki. This applies only for the FULL CI SOLUTION.

    For Hartree Fock, the ao integrals come out in their orbit/orbit form. So,
    for a 3 orbit system, the ao integrals are 3x3 or the 1e, and 9x9 for 2e. 
    '''
    # First, get electron count

    hf = scf.ROHF(mol)
    hf.kernel()
    mol_els = mol.nelec
    mol_orb = hf.mo_coeff.shape[0]
    #if (mol_els > elect) and (mol_orb>orbit):
    #    print('Performing a MCSCF calculation. Not FCI.')
    #    full_active_space=False
    #else:
    #    full_active_space=True


    if orbitals=='FCI':
        # run the MCSCF
        mc = mcscf.CASSCF(hf,elect,orbit)
        mc.kernel()

        d1 = mc.fcisolver.make_rdm1s(mc.ci,elect,orbit) # get spin 1RDM

        nocca, norba = np.linalg.eig(d1[0]) # diagonalize alpha
        noccb, norbb = np.linalg.eig(d1[1]) # diagonalize beta 
        Ta = reorder(reduce(np.dot, (norba.T,d1[0],norba)),orbit)
        # reorder according to eigenvalues for alpha
        Tb = reorder(reduce(np.dot, (norbb.T,d1[1],norbb)),orbit)
        # same as above for beta
        # generate proper 1-RDM in NO basis, alpha beta
        D1_a = reduce(np.dot, (Ta.T, norba.T, d1[0], norba, Ta))
        D1_b = reduce(np.dot, (Tb.T, norbb.T, d1[1], norbb, Tb))
        # transformation from AO to NO for alpha, beta, using the 
        # provided HF solution as well
        ao2no_a = reduce(np.dot, (mc.mo_coeff, norba, Ta))
        ao2no_b = reduce(np.dot, (mc.mo_coeff, norbb, Tb))
        # Note, these are in (AO,NO) form, so they are: "ao to no"
        # now, transorm the electron integrals 
        ints_1e_no_a = reduce(np.dot, (ao2no_a.T, mc.get_hcore(), ao2no_a))
        ints_1e_no_b = reduce(np.dot, (ao2no_b.T, mc.get_hcore(), ao2no_b))
        # important function, generates the full size 1e no (NOT 
        # in the spatial orbital basis, but in the spin basis) 
        wf = {'{}{}'.format('1'*elect,'0'*orbit):1}
        wf,alp,bet = fx.extend_wf(wf,mol_els,mol_orb)
        ints_1e = gen_spin_1ei(
                mc.get_hcore(),
                ao2no_a.T,
                ao2no_b.T,
                alp,
                bet,
                spin2spac=fx.map_spatial
                )
        ints_2e_ao = ao2mo.kernel(mol,np.identity(mol_orb),compact=False)
        ints_2e_ao = np.reshape(
                ints_2e_ao,(
                    mol_orb,
                    mol_orb,
                    mol_orb,
                    mol_orb
                    )
                )
        ints_2e_no = gen_spin_2ei(
                ints_2e_ao, ao2no_a.T, ao2no_b.T,
                alp,bet,spin2spac=fx.map_spatial
                )
        ints_2e = np.reshape(ints_2e_no,((2*mol_orb)**2,(2*mol_orb)**2))

    elif orbitals=='HF':
        class Dummy:
            def __init__(self,
                    seeded_hf_obj,
                    seeded_mo
                    ):
                self.hf = seeded_hf_obj
                self.e_tot=self.hf.e_tot
                self.mo_coeff = seeded_mo
                self.mcscf_e_tot = None

            def get_hcore(self):
                return self.hf.get_hcore()
        if seed:
            mc = mcscf.CASSCF(hf,elect,orbit)
            mc.kernel()
            try:
                shf = scf.ROHF(seed_mol)
                shf.kernel()
            except Exception:
                text = 'Some sort of error in the seeded mol object. Goodbye!'
                sys.exit(text)
            smc = mcscf.CASSCF(hf,elect,orbit)
            orb = mcscf.project_init_guess(smc,shf.mo_coeff,seed_mol)
            smc.kernel(orb)
            smci = mcscf.CASCI(hf,elect,orbit)
            smci.kernel(smc.mo_coeff)
            hf = Dummy(hf,smc.mo_coeff)
            hf.mcscf_e_tot = smci.e_tot
        else:
            mc = mcscf.CASCI(hf,elect,orbit)
            mc.kernel()

        ints_2e_ao = ao2mo.kernel(mol,np.identity(mol_orb),compact=False)
        ints_2e_ao = np.reshape(
                ints_2e_ao,(
                    mol_orb,
                    mol_orb,
                    mol_orb,
                    mol_orb
                    )
                )
        ints_2e = ints_2e_ao
        ints_1e = hf.get_hcore()

    E_fci = mc.e_tot

    return ints_1e, ints_2e, E_fci, hf

def get_ei(mol,elect,orbit):
    hf = scf.ROHF(mol)
    hf.kernel()
    mc = mcscf.CASSCF(hf,elect,orbit)
    mc.kernel()
    d1 = mc.fcisolver.make_rdm1(mc.ci,elect,orbit)

    occnum, occvec = np.linalg.eig(d1)
    T1 = reorder(reduce(np.dot, (occvec.T,d1,occvec)),3)
    ao2no = reduce(np.dot, (mc.mo_coeff, occvec,T1))

    nuc_en = mol.energy_nuc()
    ints_1e_no = reduce(np.dot, (ao2no.T, mc.get_hcore(), ao2no))
    ints_2e_no = ao2mo.kernel(mol,ao2no,compact=False)
    return ints_1e_no, ints_2e_no

def generate_spin_1ei(
        ei1,
        U_a,
        U_b,
        alpha,
        beta,
        region='active',
        spin2spac=None,
        new_ei=None):
    N= len(U_a)
    if region=='full':
        alpha = alpha['inactive']+alpha['active']+alpha['virtual']
        beta = beta['inactive']+beta['active']+beta['virtual']
        new_ei = np.zeros((N*2,N*2))
    elif region in ['active','active_space','as']:
        alpha=alpha['active']+alpha['inactive']
        beta =beta['active']+beta['inactive']
        if type(new_ei)==type(None):
            new_ei = np.zeros((N*2,N*2))
    temp1 = np.zeros((N,N))
    for i in alpha:
        P=spin2spac[i]
        for a in range(0,N):
            temp1[P,:] += U_a[P,a]*ei1[a,:]
        for j in alpha:
            new_ei[i,j]=0
            Q=spin2spac[j]
            for b in range(0,N):
                new_ei[i,j]+= U_a.T[b,Q]*temp1[P,b]
    temp1 = np.zeros((N,N))
    for i in beta:
        P=spin2spac[i]
        for a in range(0,N):
            temp1[P,:] += U_b[P,a]*ei1[a,:]
        for j in beta:
            new_ei[i,j]=0
            Q=spin2spac[j]
            for b in range(0,N):
                new_ei[i,j]+= U_b.T[b,Q]*temp1[P,b]
    return new_ei

def gen_spin_1ei_lr(
        ei1,
        Ua_l,
        Ua_r,
        Ub_l,
        Ub_r,
        alpha,
        beta,
        spin2spac=None,
        region='full',
        new_ei=None):
    N= len(Ua_l)
    if region=='full':
        alpha = alpha['inactive']+alpha['active']
        beta = beta['inactive']+beta['active']
        new_ei = np.zeros((N*2,N*2))
    elif region in ['active','active_space','as']:
        alpha=alpha['active']
        beta =beta['active']
    temp1 = np.zeros((N,N))
    for i in alpha:
        P=spin2spac[i]
        for a in range(0,N):
            temp1[P,:] += Ua_l[P,a]*ei1[a,:]
        for j in alpha:
            new_ei[i,j]=0
            Q=spin2spac[j]
            for b in range(0,N):
                new_ei[i,j]+= Ua_r[b,Q]*temp1[P,b]
    temp1 = np.zeros((N,N))
    for i in beta:
        P=spin2spac[i]
        for a in range(0,N):
            temp1[P,:] += Ub_l[P,a]*ei1[a,:]
        for j in beta:
            new_ei[i,j]=0
            Q=spin2spac[j]
            for b in range(0,N):
                new_ei[i,j]+= Ub_r[b,Q]*temp1[P,b]
    return new_ei

def generate_spin_2ei(
        ei2,
        U_a,
        U_b,
        alpha,
        beta,
        region='active',
        spin2spac=mss,
        new_ei=None):
    '''
    Input is the standard electron integral matrices, ik format where i,k are
    spatial orbitals. 

    Output is a matrix with indices, i,k,l,j
    except that j and l have been switched, so that the first index refers to
    electron 1 and the second index refers to electron 2.

    '''
    N = len(U_a)
    ei2 = np.reshape(ei2,(N,N,N,N))
    if region=='full':
        alpha = alpha['inactive']+alpha['active']+alpha['virtual']
        beta = beta['inactive']+beta['active']+beta['virtual']
        new_ei = np.zeros((N*2,N*2,N*2,N*2))
    elif region in ['active','as','active_space']:
        alpha=alpha['active']+alpha['inactive']
        beta =beta['active']+beta['inactive']
        if type(new_ei)==type(None):
            new_ei = np.zeros((N*2,N*2,N*2,N*2))
        else:
            new_ei = np.reshape(new_ei,(N*2,N*2,N*2,N*2))
    else:
        print('Error?')
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a.T[b,Q]*temp1[P,b,:,:]
            for k in alpha:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        new_ei[i,k,j,l]+= U_a.T[d,S]*temp3[P,Q,R,d]
    # now, alpha beta block 
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a.T[b,Q]*temp1[P,b,:,:]
            for k in beta:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,j,l]+= U_b.T[d,S]*temp3[P,Q,R,d]
    # beta alpha block
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b.T[b,Q]*temp1[P,b,:,:]
            for k in alpha:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_a[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,j,l]+= U_a.T[d,S]*temp3[P,Q,R,d]
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b.T[b,Q]*temp1[P,b,:,:]
            for k in beta:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,j,l]+= U_b.T[d,S]*temp3[P,Q,R,d]
    return new_ei #, temp0

def generate_spin_2ei_phys(
        ei2,
        U_a,
        U_b,
        alpha,
        beta,
        region='active',
        spin2spac=mss,
        new_ei=None):
    '''
    Input are the electron integrals in physicist notations, i.e. 
    (ij) (kl)
    i(e1)* k(e2)* j(e1) l(e2)

    Output is a matrix with same incidices, just in spin orbitals
    '''
    N = len(U_a)
    ei2 = np.reshape(ei2,(N,N,N,N))
    if region=='full':
        alpha = alpha['inactive']+alpha['active']+alpha['virtual']
        beta = beta['inactive']+beta['active']+beta['virtual']
        new_ei = np.zeros((N*2,N*2,N*2,N*2))
    elif region in ['active','as','active_space']:
        alpha=alpha['active']+alpha['inactive']
        beta =beta['active']+beta['inactive']
        if type(new_ei)==type(None):
            new_ei = np.zeros((N*2,N*2,N*2,N*2))
        else:
            new_ei = np.reshape(new_ei,(N*2,N*2,N*2,N*2))
    else:
        print('Error?')
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,:,Q,:] += U_a.T[b,Q]*temp1[P,:,b,:]
            for k in alpha:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,R,Q,:] += U_a[R,c]*temp2[P,c,Q,:]
                for l in alpha:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        new_ei[i,k,j,l]+= U_a.T[d,S]*temp3[P,R,Q,d]
    # now, alpha beta block 
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,:,Q,:] += U_a.T[b,Q]*temp1[P,:,b,:]
            for k in beta:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,R,Q,:] += U_b[R,c]*temp2[P,c,Q,:]
                for l in beta:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,j,l]+= U_b.T[d,S]*temp3[P,R,Q,d]
    # beta alpha block
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,:,Q,:] += U_b.T[b,Q]*temp1[P,:,b,:]
            for k in alpha:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,R,Q,:] += U_a[R,c]*temp2[P,c,Q,:]
                for l in alpha:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_a[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,j,l]+= U_a.T[d,S]*temp3[P,R,Q,d]
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,:,Q,:] += U_b.T[b,Q]*temp1[P,:,b,:]
            for k in beta:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,R,Q,:] += U_b[R,c]*temp2[P,c,Q,:]
                for l in beta:
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,j,l]+= U_b.T[d,S]*temp3[P,R,Q,d]
    return new_ei #, temp0


def gen_spin_2ei_QISKit(
        ei2,
        U_a,
        U_b,
        alpha,
        beta,
        region='active',
        spin2spac=mss,
        new_ei=None):
    '''
    Input is the standard electron integral matrices, ik format where i,k are
    spatial orbitals.

    Output is a matrix with indices, i,k,l,j
    except that j and l have been switched, so that the first index refers to
    electron 1 and the second index refers to electron 2.

    '''
    ei2 = fx.expand(ei2)
    N = len(U_a)
    if region=='full':
        alpha = alpha['inactive']+alpha['active']+alpha['virtual']
        beta = beta['inactive']+beta['active']+beta['virtual']
        new_ei = np.zeros((N*2,N*2,N*2,N*2))
    elif region in ['active','as','active_space']:
        alpha=alpha['active']+alpha['inactive']
        beta =beta['active']+beta['inactive']
        if type(new_ei)==type(None):
            new_ei = np.zeros((N*2,N*2,N*2,N*2))
        else:
            new_ei = np.reshape(new_ei,(N*2,N*2,N*2,N*2))
    else:
        print('Error?')
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a.T[b,Q]*temp1[P,b,:,:]
            for k in alpha:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    S = spin2spac[l]
                    for d in range(0,N):
                        new_ei[i,k,l,j]+= U_a.T[d,S]*temp3[P,Q,R,d]
    # now, alpha beta block 
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a.T[b,Q]*temp1[P,b,:,:]
            for k in beta:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    S = spin2spac[l]
                    for d in range(0,N):
                        new_ei[i,k,l,j]+= U_b[S,d]*temp3[P,Q,R,d]
    # beta alpha block
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b.T[b,Q]*temp1[P,b,:,:]
            for k in alpha:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    S = spin2spac[l]
                    for d in range(0,N):
                        new_ei[i,k,l,j]+= U_a[S,d]*temp3[P,Q,R,d]
                        #new_ei[i,k,j,l]+= U_a.T[d,S]*temp3[P,Q,R,d]
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b.T[b,Q]*temp1[P,b,:,:]
            for k in beta:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,l,j]+= U_b.T[d,S]*temp3[P,Q,R,d]
    return new_ei


def generate_spin_2ei_pyscf(
        ei2,
        U_a,
        U_b,
        alpha,
        beta,
        region='active',
        spin2spac=mss,
        new_ei=None):
    '''
    Input is the standard electron integral matrices, ik format where i,k are
    spatial orbitals. 

    Output is a matrix with indices, i,k,l,j
    except that j and l have been switched, so that the first index refers to
    electron 1 and the second index refers to electron 2.

    '''
    N = len(U_a)
    ei2 = np.reshape(ei2,(N,N,N,N))
    if region=='full':
        alpha = alpha['inactive']+alpha['active']+alpha['virtual']
        beta = beta['inactive']+beta['active']+beta['virtual']
        new_ei = np.zeros((N*2,N*2,N*2,N*2))
    elif region in ['active','as','active_space']:
        alpha=alpha['active']+alpha['inactive']
        beta =beta['active']+beta['inactive']
        if type(new_ei)==type(None):
            new_ei = np.zeros((N*2,N*2,N*2,N*2))
        else:
            new_ei = np.reshape(new_ei,(N*2,N*2,N*2,N*2))
    else:
        print('Error?')
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a.T[b,Q]*temp1[P,b,:,:]
            for k in alpha:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        new_ei[i,j,k,l]+= U_a.T[d,S]*temp3[P,Q,R,d]
    # now, alpha beta block 
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a.T[b,Q]*temp1[P,b,:,:]
            for k in beta:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,j,k,l]+= U_b.T[d,S]*temp3[P,Q,R,d]
    # beta alpha block
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b.T[b,Q]*temp1[P,b,:,:]
            for k in alpha:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    #new_ei[i,k,j,l]=0
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_a[S,d]*temp3[P,Q,R,d]
                        new_ei[i,j,k,l]+= U_a.T[d,S]*temp3[P,Q,R,d]
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = spin2spac[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = spin2spac[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b.T[b,Q]*temp1[P,b,:,:]
            for k in beta:
                R = spin2spac[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    S = spin2spac[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,j,k,l]+= U_b.T[d,S]*temp3[P,Q,R,d]
    return new_ei #, temp0
