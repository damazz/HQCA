'''
'''

from pyscf import gto,scf,mcscf,fci
import numpy as np
from math import pi
np.set_printoptions(suppress=True,precision=8)
from copy import deepcopy as copy
#
# Add location for the HQCA module
#
from functools import reduce,partial
from hqca.hamiltonian import *
from hqca.tools import Operator as Op
from hqca.tools import PauliString as Pauli
from hqca.tools import Stabilizer
from hqca.transforms import *
from hqca.acse import *
from hqca.instructions import *
from hqca.processes import *
from hqca.maple import purify
from _instruct_22 import BenzyneInstruct
from _instruct_3q_44 import Line3Q
from _instruct_4q_44 import Line4Q
from _noise_model import noisy_model_from_ibmq
#
#
#

class InputError(Exception):
    """Exception raised for errors in the input
    (from: https://docs.python.org/3/tutorial/errors.html)

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def _init__(self,expression,message):
        self.expression= expression
        self.message = message

#
# set up user inputs
#

isomer = input('Select benzyne isomer: (o) ortho-, (m) meta-, (p) para- \n')
if isomer in ['o','m','p']:
    pass
else: 
    raise InputError('{} '.format(isomer)+ r'is improper isomer. Select from \o,m,p}.')

act = input('Select active space: (2) [2e,2o], (4), [4e,4o] \n')
if act=='2':
    qubits = '1'
elif act=='4':
    qubits = input('Select number of qubits: 3, 4, 6\n')
    if qubits in ['3','4','6']:
        pass
    else:
        raise InputError('{} '.format(qubits) +r'is incorrect number of qubits.')
else:
    raise InputError('{} '.format(qubits) +r'is incorrect active space.')
qubits = int(qubits)
act = int(act)


run = 'statevector'
run = 'qasm'
acse_sol = 'classical'
acse_sol = 'quantum'

if run=='ibm':
    provider = 'IBMQ'
    backend = 'ibmq_bogota'
elif run=='statevector':
    provider = 'Aer'
    backend = 'statevector_simulator'
elif run=='noisy':
    provider = 'Aer'
    backend = 'qasm_simulator'
elif run=='qasm':
    provider = 'Aer'
    backend = 'qasm_simulator'
else:
    raise InputError('Incorrect backend selected.')

#
# obtain CASCI energies and set up dummy molecule
#

eidir = './eis_{}{}/'.format(str(act),str(act))+isomer+'benzyne/'
ei1 = np.load(eidir+'EI1.npy')
ei2 = np.load(eidir+'EI2.npy')
d1 = np.load(eidir+'D1.npy')
d2 = np.load(eidir+'D2.npy')

mol = gto.Mole()
mol.atom =[['H',(i,0,0)] for i in range(act)]
mol.basis = 'sto-3g'
mol.spin=0
mol.verbose=0
mol.build()

cisolver = fci.direct_spin1.FCI(mol)
e_fci,cis = cisolver.kernel(
        ei1,
        ei2.transpose(0,2,1,3),act,act,
        ecore=0,
        nroots=1)

# 
# get symmetry tapered operators
#

c = np.sqrt(1/2)
if qubits==1:
    U1 = Op()+Pauli('IXII',c)+Pauli('ZZII',c)
    U2 = Op()+Pauli('IXI' ,c)+Pauli('ZZI' ,c)
    U3 = Op()+Pauli('IX'  ,c)+Pauli('ZZ'  ,c)
    tap_qb = [1,1,1]
    symm_eval = [-1,+1,-1]
    tran = JordanWigner
    tr_init = partial(tran,initial=True)
    Us =[U1,U2,U3]
elif qubits==3 and isomer in ['o','p']:
    U1 = Op()+Pauli('IIIXIIII',c)+Pauli('ZZZZIIII',c)
    U2 = Op()+Pauli('IIIXIII' ,c)+Pauli('ZIIZIII' ,c)
    U3 = Op()+Pauli('IIIXII'  ,c)+Pauli('IZIZII'  ,c)
    U4 = Op()+Pauli('IIIXI'   ,c)+Pauli('IIZZI'   ,c)
    U5 = Op()+Pauli('IIIX'    ,c)+Pauli('ZZZZ'    ,c)
    Us = [U1,U2,U3,U4,U5]
    tap_qb = [3,3,3,3,3]
    symm_eval = [+1,+1,+1,+1,+1]
elif qubits==3 and isomer in ['m']:
    U1 = Op()+Pauli('IIIIIIIX',c)+Pauli('IIIIZIIZ',c)
    U2 = Op()+Pauli('IIIIIIX',c)+Pauli('ZZIIZIZ',c)
    U3 = Op()+Pauli('IIIIIX',c)+Pauli('ZZIIZZ',c)
    U4 = Op()+Pauli('IIIXI',c)+Pauli('ZIIZI',c)
    U5 = Op()+Pauli('IIXI',c)+Pauli('IZZI',c)
    Us = [U1,U2,U3,U4,U5]
    tap_qb = [7,6,5,3,2]
    symm_eval = [-1,-1,+1,-1,-1]
elif qubits==4:
    U1 = Op()+Pauli('IIIXIIII',c)+Pauli('ZZZZIIII',c)
    U2 = Op()+Pauli('IIIXIII' ,c)+Pauli('ZIZZIZI' ,c)
    U3 = Op()+Pauli('IIIXII'  ,c)+Pauli('IZZZZI'  ,c)
    U4 = Op()+Pauli('IIIIX'   ,c)+Pauli('ZZIZZ'   ,c)
    Us = [U1,U2,U3,U4]
    tap_qb = [3,3,3,4]
    symm_eval = [+1,+1,+1,+1]
elif qubits==6:
    U1 = Op()+Pauli('IIIIIIIX',c)+Pauli('IIIIZZZZ',c)
    U2 = Op()+Pauli('IIIXIII' ,c)+Pauli('ZZZZIII' ,c)
    Us = [U1,U2]
    tap_qb = [7,3]
    symm_eval = [+1,+1]


#
# obtain transformations using a recursive approach 
#

def modify(
        ops,fermi,U,Ut,
        qubits,paulis,eigvals,
        initial=False,
        ):
    '''
    Return modified fermionic transformation. 
    '''
    new = fermi(ops,initial=initial)
    new = change_basis(new,U,Ut)
    new = trim_operator(new,
            qubits=qubits,
            paulis=paulis,
            null=int(initial),
            eigvals=eigvals)
    return new
#
tran = JordanWigner
for qb,ev,U in zip(tap_qb,symm_eval,Us):
    tran = partial(
            modify,
            fermi=copy(tran),
            U=U,Ut=U,
            qubits=[qb],
            paulis=['X'],
            eigvals=[ev])
tr_init = partial(tran,initial=True)

#
# now, set up Hamiltonian
#

ham = FermionicHamiltonian(mol,
        int_thresh=1e-8,
        ints_1e=ei1,
        ints_2e=ei2,
        normalize=False,
        transform=tran,verbose=False,
        en_fin=e_fci,
        )

Ins = PauliSet
st = StorageACSE(ham,closed_ansatz=0)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend=backend,
        transpiler_keywords={'optimization_level':0},
        num_shots=8192*2,
        Nq=qubits,
        backend_initial_layout=[i for i in range(qubits)],
        provider=provider)
qs.initial_transform = tr_init
qs.path_to_maple = '/home/scott/maple2021/bin/maple'
qs.spin_rdm = True
#
# here we apply the diagonal filter on the RDM, as well as the relevant 
# error mitigation strategies 
# we also have a custom noise model based on the device paramters to use if necessary
#
proc = StandardProcess()
#
# generating tomography
#

tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,transform=tran)
tomoD3 = ReducedTomography(qs,order=3)
tomoD3.generate(real=True,imag=False,transform=tran)

# 
# here we are setting parameters
# the quality of the simulation will depend on a number of factors, including whether or 
# not we have a classical or quantum ACSE solution
# how long the ansatz is, how many iterations, etc.
#

if qubits in [1]:
    kwargs = {
            'method':'euler',
            'update':'quantum',
            'opt_thresh':0.05,
            'S_thresh_rel':0.0,
            'convergence_type':'norm',
            'max_iter':50,
            'restrict_S_size':0.5,
            'processor':proc,
            'hamiltonian_step_size':0.01,
            }
    tomoIm = ReducedTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=tran)
    kwargs['tomo_S'] = tomoIm
else:
    kwargs = {
            'method':'newton',
            'update':acse_sol,
            'opt_thresh':0.01,
            'S_thresh_rel':0.5,
            'S_min':1e-6,
            'hamiltonian_step_size':0.5,
            'convergence_type':'norm',
            'use_trust_region':True,
            'processor':proc,
            'max_iter':50,
            'newton_step':-1,
            'initial_trust_region':4,
            'restrict_S_size':1.0,
            'output':1,
            #'D3':tomoD3,
            }
    if acse_sol=='quantum':
        tomoIm = ReducedTomography(qs)
        tomoIm.generate(real=False,imag=True,simplify=True,method='gt',strategy='lf',
                transform=tran,verbose=False,
                )
        kwargs['tomo_S'] =tomoIm
kwargs['tomo_Psi']=tomoRe
kwargs['verbose']=False

E = []
Ep = []
for i in range(10):
    #
    # running the ACSE
    #
    acse = RunACSE(
            st,qs,Ins,
            **kwargs
            )
    acse.build()
    ebest = acse.best,
    dbest = acse.Store.rdm
    while not acse.total.done:
        acse._run_acse()
        acse._check()
        if acse.e_k<ebest:
            ebest = copy(acse.best)
            dbest = copy(acse.Store.rdm)
    print('')
    print('E init: {:+.12f} U'.format(np.real(acse.ei)))
    print('E run : {:+.12f} U'.format(np.real(acse.best)))
    try:
        diff = 1000 * (acse.best - acse.Store.H.ef)
        print('E goal: {:+.12f} U'.format(acse.Store.H.ef))
        print('Energy difference from goal: {:.12f} mU'.format(diff))
        E.append(diff)
    except KeyError:
        pass
    except AttributeError:
        pass
    pure = purify(dbest,qs)
    e = acse.Store.evaluate(pure)
    Ep.append(1000*(e-acse.Store.H.ef))

print('Average energy difference: {} +/- {}'.format(np.average(E),np.std(E)))
print(Ep)
print(min(Ep))
print('Average energy difference: {} +/- {}'.format(np.average(Ep),np.std(Ep)))

