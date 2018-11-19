import h3_test2
import rdmf
import sys
from pyscf import scf,gto,mcscf,ao2mo
from pyscf import fci
from functools import reduce
import numpy as np
import optimizers as opt

mss = {
    0:0, 1:1, 2:2,
    3:0, 4:1, 5:2}

def gen_spin_1ei(ei1,U_a,U_b,alpha=[0,1,2],beta=[3,4,5], spin2spac=mss ):
    N= len(U_a)
    new_ei = np.zeros((N*2,N*2))
    temp1 = np.zeros((N,N))
    for i in alpha:
        P=spin2spac[i]
        for a in range(0,N):
            temp1[P,:] += U_a[P,a]*ei1[a,:]
        for j in alpha:
            Q=spin2spac[j]
            for b in range(0,N):
                new_ei[i,j]+= U_a[Q,b]*temp1[P,b]
    temp1 = np.zeros((N,N))
    for i in beta:
        P=spin2spac[i]
        for a in range(0,N):
            temp1[P,:] += U_b[P,a]*ei1[a,:]
        for j in beta:
            Q=spin2spac[j]
            for b in range(0,N):
                new_ei[i,j]+= U_b[Q,b]*temp1[P,b]
    return new_ei

def gen_spin_2ei(ei2,U_a,U_b,alpha=[0,1,2],beta=[3,4,5],spin2spac=mss):
    #ei[i,j,k,l] = c_a^i c_b^j c_c^k c_d^l
    # unitary transformation should act on a column vector...
    # i.e., AO to MO
    # 
    N = len(U_a)
    new_ei = np.zeros((N*2,N*2,N*2,N*2))
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    #temp0 = np.zeros((N,N,N,N))
    ## alpha alpha portion
    for i in alpha:
        P = mss[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = mss[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a[Q,b]*temp1[P,b,:,:]
            for k in alpha:
                R = mss[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    S = mss[l]
                    for d in range(0,N):
                        new_ei[i,k,l,j]+= U_a[S,d]*temp3[P,Q,R,d]
                        #new_ei[i,j,k,l]+= U_a[S,d]*temp3[P,Q,R,d]
                        #temp0[P,Q,R,S] += U_a[S,d]*temp3[P,Q,R,d]

    # now, alpha beta block 

    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = mss[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ei2[a,:,:,:]
        for j in alpha:
            Q = mss[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a[Q,b]*temp1[P,b,:,:]
            for k in beta:
                R = mss[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    S = mss[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,l,j]+= U_b[S,d]*temp3[P,Q,R,d]

    # beta alpha block

    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = mss[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = mss[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b[Q,b]*temp1[P,b,:,:]
            for k in alpha:
                R = mss[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    S = mss[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_a[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,l,j]+= U_a[S,d]*temp3[P,Q,R,d]


    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = mss[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*ei2[a,:,:,:]
        for j in beta:
            Q = mss[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b[Q,b]*temp1[P,b,:,:]
            for k in beta:
                R = mss[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    S = mss[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        new_ei[i,k,l,j]+= U_b[S,d]*temp3[P,Q,R,d]

    return new_ei #, temp0












def rotate_2rdm(aa,ab,bb,U_a,U_b,alpha=[0,1,2],beta=[3,4,5],spin2spac=mss):
    # should still be output as i j k l, which is also the input
    N = len(aa)
    rdm2 = np.zeros((N*2,N*2,N*2,N*2))
    ba = np.zeros((N,N,N,N))
    '''
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                for l in range(0,N):
                    rdm2[i,j,k,l]+= aa[i,j,k,l]
                    rdm2[i,j,k+3,l+3]+= ab[i,j,k,l]
                    rdm2[i+3,j+3,k,l]+= ab[i,j,k,l]
                    rdm2[i+3,j+3,k+3,l+3]+= bb[i,j,k,l]
    '''

    n2rdm = np.zeros((N*2,N*2,N*2,N*2))
    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    #temp0 = np.zeros((N,N,N,N))
    ## alpha alpha portion
    for i in alpha:
        P = mss[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*aa[a,:,:,:]
        for j in alpha:
            Q = mss[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a[Q,b]*temp1[P,b,:,:]
            for k in alpha:
                R = mss[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    S = mss[l]
                    for d in range(0,N):
                        n2rdm[i,k,l,j]+= U_a[S,d]*temp3[P,Q,R,d]

    # now, alpha beta block 

    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in alpha:
        P = mss[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_a[P,a]*ab[a,:,:,:]
        for j in alpha:
            Q = mss[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_a[Q,b]*temp1[P,b,:,:]
            for k in beta:
                R = mss[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    S = mss[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        n2rdm[i,k,l,j]+= U_b[S,d]*temp3[P,Q,R,d]
    for i in alpha:
        for j in alpha:
            for k in beta:
                for l in beta:
                    n2rdm[k,i,j,l] = n2rdm[i,k,l,j]
    '''
    # beta alpha block

    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = mss[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*rdm2[a,:,:,:]
        for j in beta:
            Q = mss[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b[Q,b]*temp1[P,b,:,:]
            for k in alpha:
                R = mss[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_a[R,c]*temp2[P,Q,c,:]
                for l in alpha:
                    S = mss[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_a[S,d]*temp3[P,Q,R,d]
                        n2rdm[i,k,l,j]+= U_a[S,d]*temp3[P,Q,R,d]
    '''

    temp1 = np.zeros((N,N,N,N))
    temp2 = np.zeros((N,N,N,N))
    temp3 = np.zeros((N,N,N,N))
    for i in beta:
        P = mss[i]
        for a in range(0,N):
            temp1[P,:,:,:] += U_b[P,a]*bb[a,:,:,:]
        for j in beta:
            Q = mss[j]
            for b in range(0,N):
                temp2[P,Q,:,:] += U_b[Q,b]*temp1[P,b,:,:]
            for k in beta:
                R = mss[k]
                for c in range(0,N):
                    temp3[P,Q,R,:] += U_b[R,c]*temp2[P,Q,c,:]
                for l in beta:
                    S = mss[l]
                    for d in range(0,N):
                        #new_ei[i,j,k,l]+= U_b[S,d]*temp3[P,Q,R,d]
                        n2rdm[i,k,l,j]+= U_b[S,d]*temp3[P,Q,R,d]



    return n2rdm

np.set_printoptions(linewidth=200, precision=4,suppress=True)
mol = gto.Mole()
mol.atom = '''H 0 0 0; H 0 0 0.9374; H 0 0 -0.9374'''
mol.basis = 'sto-3g'
mol.spin=1
mol.verbose=0
mol.build()
ne = mol.energy_nuc()

m = scf.ROHF(mol)
m.kernel()
mc = mcscf.CASSCF(m,3,3)
mc.kernel()

d1 = mc.fcisolver.make_rdm1s(mc.ci,3,3)
rdm1sf = mc.fcisolver.make_rdm1(mc.ci,3,3)
rdm2sf = mc.fcisolver.make_rdm2(mc.ci,3,3)
rdm2sf = np.reshape(rdm2sf,(9,9))
d1,d2 = mc.fcisolver.make_rdm12s(mc.ci,3,3)
#print(d2[0],d2[1],d2[2])
nocca, norba = np.linalg.eig(d1[0])
noccb, norbb = np.linalg.eig(d1[1])
nocc, norb = np.linalg.eig(rdm1sf)
T = h3_test2.reorder(reduce(np.dot, (norb.T,rdm1sf,norb)),3)
Ta = h3_test2.reorder(reduce(np.dot, (norba.T,d1[0],norba)),3)
Tb = h3_test2.reorder(reduce(np.dot, (norbb.T,d1[1],norbb)),3)
D1_a = reduce(np.dot, (Ta.T, norba.T, d1[0], norba, Ta))
D1_b = reduce(np.dot, (Tb.T, norbb.T, d1[1], norbb, Tb))
D1 = reduce(np.dot, (T.T, norb.T, rdm1sf, norb, T))
mo2no_a = reduce(np.dot, (norba, Ta))
mo2no_b = reduce(np.dot, (norbb, Tb))

D2 = rotate_2rdm(d2[0],d2[1],d2[2],mo2no_a.T,mo2no_b.T)
D1_sno = rdmf.check_2rdm(D2)
D2 = np.reshape(D2, (36,36))
print(D2)

ao2no_a = reduce(np.dot, (mc.mo_coeff, norba, Ta))
ao2no_b = reduce(np.dot, (mc.mo_coeff, norbb, Tb))
ao2no = reduce(np.dot, (mc.mo_coeff, norb, T))
ints_1e_mo = reduce(np.dot, (mc.mo_coeff.T, mc.get_hcore(),mc.mo_coeff))
ints_1e_no_a = reduce(np.dot, (ao2no_a.T,mc.get_hcore(),ao2no_a))
ints_1e_no = reduce(np.dot, (ao2no.T,mc.get_hcore(),ao2no))
ints_1e_no_b = reduce(np.dot, (ao2no_b.T,mc.get_hcore(),ao2no_b))

ints_2e_mo = ao2mo.kernel(mol,m.mo_coeff,compact=False)
ints_2e_ao = ao2mo.kernel(mol,np.identity(3),compact=False)
ints_2e_ao = np.reshape(ints_2e_ao,(3,3,3,3))
ints_2e_no = gen_spin_2ei(ints_2e_ao,ao2no_a.T,ao2no_b.T,[0,1,2],[3,4,5],spin2spac=mss) # spin NO basis
#ints_2e_no = gen_spin_2ei(ints_2e_ao,ao2no.T,ao2no.T,[0,1,2],[3,4,5]) # NO basis
#ints_2e_no = gen_spin_2ei(ints_2e_ao,mc.mo_coeff.T,mc.mo_coeff.T,[0,1,2],[3,4,5])
ints_2e_no_sf = ao2mo.kernel(mol,ao2no,compact=False)

ints_2e_no = np.reshape(ints_2e_no,(36,36))
ints_1e_no_t = gen_spin_1ei(mc.get_hcore(), ao2no_a.T,ao2no_b.T,[0,1,2],[3,4,5],spin2spac=mss)
ints_1e_no_t = np.reshape(ints_1e_no_t,(6,6))

zeta_inv = {v:k for k,v in rdmf.map_zeta.items()}
zeta = rdmf.map_zeta
mca = mcscf.CASSCF(m,3,3)
mca.natorb=True
mca.kernel()

wf={}
for c,ia,ib in mc.fcisolver.large_ci(mca.ci,3,(2,1),tol=0.01, return_strs=False):
    #print('     %s          %s          %.12f' % (ia,ib,c*c))
    det = '000000'
    i1 = int(ia[0])
    i2 = int(ia[1])
    i3 = int(ib[0])+3
    det = det[0:i1]+'1'+det[i1+1:]
    det = det[0:i2]+'1'+det[i2+1:]
    det = det[0:i3]+'1'+det[i3+1:]
    wf[det]=c
    # ah, this is already in the alpha beta basis

alpha = np.sqrt(D1_b[0,0])
beta = np.sqrt(D1_b[1,1])
gamma = np.sqrt(1 - alpha**2 - beta**2)
print(alpha,beta,gamma)
wf = rdmf.wf_BD(alpha,beta,gamma)
#print(wf)
wf = rdmf.map_wf(wf,rdmf.map_kappa)
rdm2 = rdmf.build_2rdm(wf)
print(wf)
sf_2rdm_ss = rdmf.spin_free_rdm2(rdm2,rdmf.map_spatial)
sf_2rdm_ss = np.reshape(sf_2rdm_ss, (9,9))
rdm1 = rdmf.check_2rdm(rdm2)
'''
#
test = np.zeros((6,6,6,6))
for i in range(0,6):
    for j in range(0,6):
        for k in range(0,6):
            for l in range(0,6):
                 test[i,j,k,l] = rdm2[i,k,l,j]

rdm2 = test.copy()
#
'''
rdm2 = np.reshape(rdm2,(36,36))


print('Comparing 1e- energies from the spatial and spin approaches.')
spat_E1 = np.dot(ints_1e_mo,rdm1sf).trace()
spac_E1 = np.dot(ints_1e_no,D1).trace()
spin_E1 = np.dot(ints_1e_no_a,D1_a).trace()+np.dot(ints_1e_no_b,D1_b).trace()
test_E1 = np.dot(ints_1e_no_t,D1_sno).trace()

print(ints_1e_no_a)
print(ints_1e_no_b)
print(ints_1e_no_t)

print(spat_E1)
print(spac_E1)
print(spin_E1)
print(test_E1)
print('Comparing 2e- energies from the spatial and spin approaches.')
spat_E2 = 0.5*np.dot(ints_2e_mo,rdm2sf).trace()
spac_E2 = 0.5*np.dot(ints_2e_no_sf,sf_2rdm_ss).trace() # doesnt 

spin_E2 = 0.5*np.dot(ints_2e_no,rdm2).trace()
# doesnt work because.....no....rdm2 is in the no basis - so why different? Oh, it is in the NO basis, but not te SNO basis
test_E2 = 0.5*np.dot(ints_2e_no,D2).trace()

print(spat_E2)
print(spac_E2)
print(spin_E2)
print(test_E2)

#print(D2[7])
#print(rdm2[7])

#print(rdm2sf)
#print(ints_2e_no_sf)
#print(ints_2e_no)
#print(sf_2rdm_ss)


print('Total Energy of spatial and spin approaches:')
print('Spatial (pyscf int/ pyscf rdm: {} Hartrees'.format(ne+spat_E1+spat_E2))
print('Space   (pyscf int/ ss    rdm: {} Hartrees'.format(ne+spac_E1+spac_E2))
print('Spin    (ss    int/ ss    rdm: {} Hartrees'.format(ne+spin_E1+spin_E2))
print('Spin 2  (ss    int/ ss    rdm: {} Hartrees'.format(ne+test_E1+test_E2))

#
#
#
#
sys.exit()
#
## sys.exit()
#
#
#
#
#
#


def energy_call(nuc_en,ints_1e_a,ints_1e_b,ints_2ei,nwf):
    rdm2 = rdmf.build_2rdm(nwf)
    rdm1 = rdmf.check_2rdm(rdm2)
    rdm2 = np.reshape(rdm2,(36,36))
    rdm1a = np.zeros((3,3),dtype=np.complex_)
    rdm1b = np.zeros((3,3),dtype=np.complex_)
    for i in range(0,3):
        rdm1a[i,i]=rdm1[i,i]
        rdm1b[i,i]=rdm1[i+3,i+3]
    spin_E1 = np.dot(ints_1e_a,rdm1a).trace()+np.dot(ints_1e_b,rdm1b).trace()
    spin_E2 = 0.5*np.dot(ints_2ei,rdm2).trace()
    #print(spin_E1,spin_E2)
    return nuc_en + spin_E1 + spin_E2


maps = rdmf.map_lambda
gamma = 0.1
crit = 1
alp,bet,gam = 1,0,0
wf0 = rdmf.wf_BD(1,0,0)
step = 0.00001
Alp = np.sqrt(1-step**2)
wf1 = rdmf.wf_BD(Alp,step,0)
wf2 = rdmf.wf_BD(Alp,0,step)
wf0 = rdmf.map_wf(wf0,  maps)
print(wf0)
wf1 = rdmf.map_wf(wf1,  maps)
wf2 = rdmf.map_wf(wf2,  maps)

e0 = energy_call(ne,ints_1e_no_a,ints_1e_no_b,ints_2e_no,wf0)
e1 = energy_call(ne,ints_1e_no_a,ints_1e_no_b,ints_2e_no,wf1)
e2 = energy_call(ne,ints_1e_no_a,ints_1e_no_b,ints_2e_no,wf2)
print(e0,e1,e2)
dEdbet = (e1-e0)/step
dEdgam = (e2-e0)/step
grad = np.array([dEdbet,dEdgam])
par = np.array([0,0])
while crit>=0.00001:
    par = par - grad*gamma
    if par[0]<0:
        par[0]=0
    if par[1]<0:
        par[1]=0
    wf0 = rdmf.wf_BD(np.real(np.sqrt(1-np.sum(np.square(par)))),np.real(par[0]),np.real(par[1]))
    wf1 = rdmf.wf_BD(np.real(np.sqrt(1-(par[0]+step)**2-par[1]**2)),np.real(par[0]+step),np.real(par[1]))
    wf2 = rdmf.wf_BD(np.real(np.sqrt(1-(par[0])**2-(par[1]+step)**2)),np.real(par[0]),np.real(par[1]+step))
    wf0 = rdmf.map_wf(wf0,maps)
    wf1 = rdmf.map_wf(wf1,maps)
    wf2 = rdmf.map_wf(wf2,maps)
    e0 = energy_call(ne,ints_1e_no_a,ints_1e_no_b,ints_2e_no,wf0)
    e1 = energy_call(ne,ints_1e_no_a,ints_1e_no_b,ints_2e_no,wf1)
    e2 = energy_call(ne,ints_1e_no_a,ints_1e_no_b,ints_2e_no,wf2)
    dEdbet = (e1-e0)/step
    dEdgam = (e2-e0)/step
    grad = np.array([dEdbet,dEdgam])
    print('Gradient: {}'.format(grad))
    print('Parameter: {}'.format(par))
    print('Energy: {}'.format(e0))
    crit  = np.sqrt(np.sum(np.square(grad)))
    print(crit)







