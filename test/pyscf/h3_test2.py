import sys
from pyscf import scf,gto,mcscf,ao2mo
from pyscf import fci
from functools import reduce
import numpy as np
import rdmf
def reorder(rdm1,orbit):
    # reorders the spatial orbitals, likely according to something...the     eigenvalues of the 
    # spatial 1-RDM (NOT spin, which is good)
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
def antisymmetrize(one,two):
    # takes two matrices and outputs the antisymmetric product
    prod = np.zeros((one.shape[0],one.shape[1],two.shape[0],two.shape[1]))
    for i in range(0,len(one)):
        for j in range(0,len(one)):
            for k in range(0,len(two)):
                for l in range(0,len(two)):
                    prod[i,j,k,l] =  one[i,j]*two[k,l]
                    prod[i,j,k,l]+=  -one[k,j]*two[i,l]
                    prod[i,j,k,l]+=  -one[i,l]*two[k,j]
                    prod[i,j,k,l]+=  one[k,l]*two[i,j]
    return prod
'''
np.set_printoptions(linewidth=200, precision=4,suppress=True)
mol = gto.Mole()
#mol.atom = ''''H 0 0 0; H 0 0 0.6374; H 0 0 -0.6374''''
mol.basis = 'sto-3g'
mol.spin=1
mol.verbose=4
mol.build()
ne = mol.energy_nuc()

m = scf.ROHF(mol)
m.max_cycle=100
m.kernel()

mc = mcscf.CASSCF(m,3,3)
#mc.natorb=True
mc.kernel()
d1 = mc.fcisolver.make_rdm1(mc.ci,3,3)
on, onv = np.linalg.eig(d1)

d1_diag = np.diag(on)
T = reorder(d1_diag,3)
ao2no = reduce(np.dot, (mc.mo_coeff,onv,T))
ints_1e_mo = reduce(np.dot, (mc.mo_coeff.T, mc.get_hcore(),mc.mo_coeff))
ints_1e_no = reduce(np.dot, (ao2no.T,mc.get_hcore(),ao2no))
#print(ints_1e_no)
#print(ints_1e_mo)
print('Comparing 1e- energies from the natural and molecular orbitals.')
print(np.dot(ints_1e_mo,d1).trace())
print(np.dot(ints_1e_no,reduce(np.dot, (T.T,d1_diag,T))).trace())



d1s, d2s = mc.fcisolver.make_rdm12s(mc.ci,3,3)
print(d2s[1])
#print(d2s[1])
#print(d2s[2])

#d2 = np.reshape(d2,(36,36))
ints_2e_mo = ao2mo.kernel(mol,m.mo_coeff,compact=False)
ints_2e_no = ao2mo.kernel(mol,ao2no,compact=False)

#print('Electron integrals: mocoeff, and ao2no:')


mca = mcscf.CASSCF(m,3,3)
mca.natorb=True
mca.kernel()
ints_2e_mca=ao2mo.kernel(mol,mca.mo_coeff,compact=False)
lamb_ints_2e_no = ao2mo.kernel(mol,mca.mo_coeff,compact=False)
D2 = mc.fcisolver.make_rdm2(mca.ci,3,3)
D2 = np.reshape(D2, (9,9))

mapping = {0:0,1:3,2:1,3:4,4:2,5:5}
Mapping = {0:0,1:1,2:2,3:0,4:1,5:2}
wf = {}
nelec = (3,3)

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

rdm2 = rdmf.build_2rdm(wf,[0,1,2],[3,4,5])
rdm2sf = rdmf.spin_free_rdm2(rdm2,Mapping)
rdm2sf = np.reshape(rdm2sf,(9,9))
rdm1 = rdmf.check_2rdm(rdm2)
print('Comparing 2e- energies from the natural and molecular orbitals.')
print(0.5*np.dot(lamb_ints_2e_no,D2).trace())
print(0.5*np.dot(ints_2e_mca,rdm2sf).trace())

trace1 = 0
trace2 = 0
for p in range(0,3):
    for q in range(0,3):
        trace1+=D2[3*p+p,3*q+q]
        trace2+=rdm2sf[3*p+p,3*q+q]
print(rdm2sf)


print('Trace of the 2-RDM (pyscf): {}'.format(trace1))
print('Trace of the 2-RDM (smart): {}'.format(trace2))




sys.exit()
#check density matrix())




ints_2e_no = ao2mo.kernel(mol,ao2no,compact=False)

d1,d2 = mc.fcisolver.make_rdm12(mc.ci,3,3)


mca = mcscf.CASSCF(m,3,3)
mca.natorb=True
mca.kernel()
d2 = mc.fcisolver.make_rdm2(mca.ci,3,3)
print(ao2mo.kernel(mol,mca.mo_coeff,compact=False))
print(ints_2e_no)
d2 = np.reshape(D2,(9,9))


aleph = ao2mo.kernel(mol,mca.mo_coeff,compact=False)
a = np.dot(ints_2e_no,d2).trace()
ap = np.dot(aleph,d2).trace()
b = np.dot(ints_1e_no,D1).trace()


print(a*0.5)
print(ap*0.5)
print(b)
print(ne)
print('Total: {}'.format(ap*0.5+b+ne))
print(ints_1e_no)

#print(mc.ci)
#print(test)
print('  det-alpha,    det-beta,   CI coefficients')
occslst = fci.cistring._gen_occslst(range(3), 3//2)
'''
'''
for i,occsa in enumerate(occslst):
    for j,occsb in enumerate(occslst):
        print('     %s          %s          %.12f' % (occsa, occsb, test[i,j]))

'''
'''
print('  det-alpha,    det-beta,   CI coefficients')
for c,ia,ib in mc.fcisolver.large_ci(mc.ci,3,(2,1),tol=0.01, return_strs=False):
    print('     %s          %s          %1.12f' % (ia,ib,c))
'''
'''
'''
