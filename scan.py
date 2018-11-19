'''
scan.py 

Copy of components of main program for scanning energy functions. 

First, a parameter file and molecular file are specified as arguments to
main.py. These must be provided. A mol.py file from pyscf which fills out the
atomic information is all that is required. For the program input file, should
just copy something already in use, or see the documentation in
/doc/options.txt. 

Integrals at the correct method level are computed, and then an optimization
procedure is carried out. 

Energies for the optimization can be carried in a number of ways. There are
classical options for certain problems, but the more general approach is to use
the IBM quantum computer and QISKIT modules, which will compute the energy of a
certain wavefunction with a quantum computer. The optimization then uses that,
and will proceed as needed. Current implementation favors the Nelder-Mead
process. 

Most of the functionality for the program is in /tools/. The interface for the
quantum computer is in /ibmqx/, where documentation is a little outdated for
certtain aspects. Critical for functionality are:
/tools/chem.py          - Manages chemical attributes, electron integrals
/tools/energy.py        - Energy functions to be called 
/tools/functions.py     - Common functions for various applications
/tools/optimizers.py    - Houses optimizers functionality
/tools/rdmf.py          - Functions related to RDM manipulation and creation
/tools/printing.py      - Has compact print statements


'''

import subprocess
import pickle
import os
import numpy as np
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from tools import Functions as fx
from tools import Optimizers as opt
from tools import Chem as chem
from tools import RDMFunctions as rdmf
from tools import EnergyDeterminant as end
from tools import EnergyOrbital as eno
from tools import EnergyFunctions as enf
from tools import Triangulation as tri
import datetime
import sys
np.set_printoptions(precision=8)


# Setting the input file and molecular locations  

#try:
#    filename = sys.argv[1]
#except:
#    filename = './test/h3_dis/h3.txt'
filename = './diag.txt'
try:
    mol_loc = sys.argv[1]
except:
    mol_loc = './test/h3_dis/h3_2.py'


with open('./config.txt','w') as fp:
    fp.write('# Pointer for input file \n')
    fp.write('input_file= {} \n'.format(filename))

# Now, read in the variables

print('----------')
print('--START---')
print('-SCAN-RUN-')
print('----------')
print('Computational parameters are taken from: {}'.format(filename))
print('Molecular parameters are taken from: {}'.format(mol_loc))
print('Importing run parameters.')
print('----------')
print('Run on: {}'.format(datetime.datetime.now().isoformat()))
print('----------')
import pre
print('----------')
if pre.occ_energy=='qc':
    print('Hello. Beginning a scan of your optimization method. ')
elif pre.occ_energy=='classical':
    print('Hello. Beginning a scan of your optimization method. ')
print('Let\'s begin!')
print('----------')
# writing the mol.py to a local file so we can import the parameters

print('Molecular parameters:' )
with open('./mol.py','w') as fp:
    with open(mol_loc,'r') as bb:
        for line in bb:
            fp.write(line)
            if line[0:4]=='from':
                continue
            print(line[:-1])
print('----------')

if pre.restart_run:
    restart_file = sys.argv[3]

# NOW, importing from the mol.py file, and get the electron integrals 
# Currently, only support for FCI orbitals, but going to add orbital 
# optimization procedure. 

if pre.chem_orbitals=='FCI':
    print('Calculating electron integrals in the full CI basis.')
else:
    print('Getting the electron integrals in the Hartree-Fock basis.')
els  = 3
orbs = 3
import mol
E_ne = mol.mol.energy_nuc()
ints_1e, ints_2e, E_fci, hf_obj = chem.get_spin_ei(
        mol=mol.mol,elect=els,
        orbit=orbs,orbitals=pre.chem_orbitals
        )
mol_els = mol.mol.nelec[0]+mol.mol.nelec[1]
mol_orb = hf_obj.mo_coeff.shape[0]
if pre.chem_orbitals=='FCI':
    opt_orb = False
elif pre.chem_orbitals=='HF':
    opt_orb = True
    Np = orbs*(orbs-1)
# spin orbital basis (i.e. natural orbitals, or SCF orbitals)

print('Electron integrals obtained. Moving forward.')
print('----------')
print('Wavefunction mapping is: {}'.format(pre.mapping))
print('Nuclear energy: {:.8f} Hartrees'.format(E_ne))
print('Quantum algorithm: {}'.format(pre.qc_algorithm))
print('Quantum backend: {}'.format(pre.qc_use_backend))
print('----------')

# Setting mapping for system. Should be size specific. 

mapping = fx.get_mapping(pre.mapping)


#
#
# Now, beginning optimization procedure. 
#
#

if pre.restart_run:
    try:
        with open(restart_file,'rb') as fb_in:
            dat = pickle.load(fb_in)
            Run = dat[0]
            Store = dat[1]
            keys = dat[2]
            orb_keys = dat[3]
    except:
        traceback.print_exc()
        sys.exit('Something is wrong with reading .tmp file. Goodbye!')
    Run.error = False
    Run.check()
else:
    store_keys = {
            'Nels_tot':mol_els,
            'Norb_tot':mol_orb,
            'Nels_as':els,
            'Norb_as':orbs
            }
    Store  = enf.Storage(
        **store_keys
            )
    if opt_orb:
        orb_keys = {
            'ints_1e_ao':ints_1e,
            'ints_2e_ao':ints_2e,
            'E_ne':E_ne,
            'print_run':False,
            'energy':'orbitals',
            'mo_coeff_a':hf_obj.mo_coeff,
            'mo_coeff_b':hf_obj.mo_coeff,
            'store':Store
            }
    else:
        orb_keys={}
    if pre.chem_orbitals=='HF':
        ints_1e = chem.gen_spin_1ei(
                ints_1e,
                hf_obj.mo_coeff.T,
                hf_obj.mo_coeff.T,
                alpha=Store.alpha_mo,
                beta=Store.beta_mo,
                region='full',
                spin2spac=Store.s2s
                )
        ints_2e = chem.gen_spin_2ei(
                ints_2e,
                hf_obj.mo_coeff.T,
                hf_obj.mo_coeff.T,
                alpha=Store.alpha_mo,
                beta=Store.beta_mo,
                region='full',
                spin2spac=Store.s2s
                )
        ints_2e = np.reshape(ints_2e,(36,36))
    if pre.occ_energy=='classical':
        # Energy optimization procedure is classical, very few key word arguments 
        # necessary for energy 
        keys = {
            'wf_mapping':mapping,
            'ints_1e_no':ints_1e,
            'ints_2e_no':ints_2e,
            'E_ne': E_ne,
            'energy':pre.occ_energy,
            'method':pre.occ_method,
            #'print_run':pre.print_extra,
            'print_run':False,
            'store':Store
            }
        pre.occ_increase_runs=False
    elif pre.occ_energy=='qc':
        # Energy function is computed through the quantum computer 
        keys = {
            'wf_mapping':mapping,
            'ints_1e_no':ints_1e,
            'ints_2e_no':ints_2e,
            'E_ne': E_ne,
            'algorithm':pre.qc_algorithm,
            'backend':pre.qc_use_backend,
            'order':pre.qc_qubit_order,
            'num_shots':pre.qc_num_shots,
            'split_runs':pre.qc_combine_run,
            'connect':pre.qc_connect,
            'method':pre.occ_method,
            'print_run':False,
            'energy':pre.occ_energy,
            'verbose':pre.qc_verbose,
            'wait_for_runs':pre.wait_for_runs,
            'store':Store
            }
        keys['triangle']=tri.find_triangle(
                Ntri=pre.occ_method_Ntri,
                **keys)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
N = 25
x = np.linspace(-45,45,N)
y = np.linspace(-45,45,N)
X,Y = np.meshgrid(x,y,indexing='ij')

if pre.occ_method=='diagnostic':
    d1= []
    d2=[]
    d3 = []
    E = []
    for i in range(0,N):
        for j in range(0,N):
            if pre.occ_energy=='qc':
                on1,on2,on3,E_t = end.energy_eval_quantum(
                        para=[X[i,j],Y[i,j]],
                        **keys)
                d1.append(on1.tolist())
                d2.append(np.asarray(on2).tolist())
                d3.append(np.asarray(on3).tolist())
                E.append(E_t)
            elif pre.occ_energy=='classical':
                on1,E_t = end.energy_eval_sim(
                        parameters=[X[i,j],Y[i,j]],
                        **keys)
                d3.append(on1.tolist())
                E.append(E_t)
        print('{:.1f}%'.format((i+1)*100/N))
    d1 = np.asmatrix(d1)
    d2 = np.asmatrix(d2)
    d3 = np.real(np.asmatrix(d3))
    #print(d3)
    #E = np.asmatrix(E)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(2/3,2/3,2/3,s=20,c='k')
    #points = ax.scatter(d3[:,0],d3[:,1],d3[:,2],c=E,cmap=cm.coolwarm)
    points = ax.scatter(d1[:,0],d1[:,1],d1[:,2],c=E,cmap=cm.coolwarm)
    plt.colorbar(points)
    ax.set_xlim(0.5,1)
    ax.set_ylim(0.5,1)
    ax.set_zlim(0.5,1)
else:
    Z = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            if pre.occ_energy=='qc':
                Z[i,j] = end.energy_eval_quantum(
                        para=[X[i,j],Y[i,j]],
                        **keys)
            elif pre.occ_energy=='classical':
                Z[i,j] = end.energy_eval_classical(
                        parameters=[X[i,j],Y[i,j]],
                        **keys)
        print('{:.1f}%'.format((i+1)*100/N))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel('theta')
    ax.set_ylabel('phi')
    maps = ax.plot_surface(X, Y, Z,
            cmap=cm.coolwarm,
            linewidth=0)
    plt.colorbar(maps)
    # Plot the surface.
print(np.min(Z))
plt.show()
