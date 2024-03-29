from hqca.acse import *
from pyscf import gto
from hqca.hamiltonian import *
from hqca.instructions import *
from delayed_assert import delayed_assert as da
from hqca.acse._ansatz_S import *
from _generic import *
from hqca.operators.quantum_strings import PauliString as Pauli
from hqca.operators.quantum_strings import FermiString as Fermi
from hqca.operators.quantum_strings import QubitString as Qubit
from hqca.operators._operator import Operator as Op
from hqca.tools._stabilizer import Stabilizer
from hqca.transforms import *
from functools import partial
import hqca.config as config
config._use_multiprocessing=False


def test_reduce():
    H = Op()
    H+= Pauli('XX',1)
    H+= Pauli('YY',0.5)
    H+= Pauli('ZI',0.25)
    H+= Pauli('IZ',0.25)
    # diagnostic
    #X = Op([Pauli('IX',1/np.sqrt(2))])
    #Z = Op([Pauli('ZZ',1/np.sqrt(2))])
    #print(Z*H*X+X*H*Z+Z*H*Z+X*H*X)
    #state = Pauli('XI')
    #print(Hp)
    Hp= H.transform(InverseJordanWigner)
    s = Stabilizer(H,verbose=True)
    s.gaussian_elimination()
    s.find_symmetry_generators()
    Tr,iTr = get_transform_from_symmetries(
            Transform=JordanWigner,
            symmetries=['ZZ'],
            qubits=[1],
            eigvals=[-1],
            )
    tr,itr = get_transform_from_symmetries(
            Transform=JordanWigner,
            symmetries=['ZZ'],
            qubits=[1],
            eigvals=[+1],
            )
    da.expect(len(Hp.transform(Tr))==1)
    da.expect(Hp.transform(Tr)['X']==Pauli('X',1))
    da.expect(len(Hp.transform(tr))==2)
    da.assert_expectations()

test_reduce()

def test_acse_reduced():
    ham = generic_molecular_hamiltonian()
    s = Stabilizer(ham._qubOp,verbose=True)
    s.gaussian_elimination()
    s.find_symmetry_generators()
    Tr,iTr = get_transform_from_symmetries(
            Transform=JordanWigner,
            symmetries=['ZIIZ','ZIZ','ZZ'],
            qubits=[3,2,1],
            eigvals=[-1,+1,-1],
            )
    ham = MolecularHamiltonian(mol=generic_mol(),transform=Tr)
    ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=1,
            provider='Aer')
    qs.initial_transform = iTr
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=Tr,use_multiprocessing=False,)
    tomoIm = StandardTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=Tr,use_multiprocessing=False,)
    proc = StandardProcess()
    st = StorageACSE(ham,closed_ansatz=-1)
    acse = RunACSE(
            st,qs,ins,processor=proc,
            method='euler',
            update='quantum',
            opt_thresh=1e-10,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='norm',
            hamiltonian_step_size=0.000001,
            max_iter=1,
            newton_step=-1,
            epsilon=1.0,
            tomo_A = tomoIm,
            tomo_psi = tomoRe,
            verbose=False,
            )
    acse.build()
    da.expect(abs(acse.e0+0.783792654277353)<=1e-10)
    acse.run()
    e1 = -0.831285219165
    da.expect(abs(acse.e0-e1)<=1e-8)
    da.assert_expectations()


'''
U1 = Operator()
U2 = Operator()
U3 = Operator()

x1 = 'IXII'#IXII  tr 0
z1 = 'ZZII'#ZZII

x2 = 'IXI' #I_XI
z2 = 'ZZI' #Z_ZI  tr 3

x3 = 'IX'  #I__X
z3 = 'ZZ'  #Z__Z  tr 4

c = 1/np.sqrt(2)
U1+= PauliString(x1,c)
U1+= PauliString(z1,c)

U2+= PauliString(x2,c)
U2+= PauliString(z2,c)

U3+= PauliString(x3,c)
U3+= PauliString(z3,c)


tr0 = partial(
        modify,
        fermi=JordanWigner,
        U=U1,Ut=U1,
        qubits=[1],
        paulis=['X'],
        eigvals=[-1],
        )
tr1 = partial(
        modify,
        fermi=tr0,
        U=U2,Ut=U2,
        qubits=[1],
        paulis=['X'],
        eigvals=[+1],
        )
tr2 = partial(modify,
        fermi=tr1,
        U=U3,Ut=U3,
        qubits=[1],
        paulis=['X'],
        eigvals=[-1],
        )
tran = tr2
tr_init = partial(tran,initial=True)
Nq = 1

ham = FermionicHamiltonian(mol,
        int_thresh=1e-5,
        ints_1e=ei1,
        ints_2e=ei2,
        normalize=False,
        transform=tran,verbose=True,
        en_fin=e_fci[0],
        )
print('Checking hamiltonian for additional symmetries...')
par = Stabilizer(ham._qubOp,verbose=True)
par.gaussian_elimination()
par.find_symmetry_generators()
'''
