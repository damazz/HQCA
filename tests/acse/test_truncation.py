from hqca.acse import *
from pyscf import gto
from hqca.hamiltonian import *
from hqca.instructions import *
from delayed_assert import delayed_assert as da
from hqca.acse._ansatz_S import *
from _generic_acse import *
from hqca.operators.quantum_strings import PauliString as Pauli
from hqca.operators.quantum_strings import FermiString as Fermi
from hqca.operators.quantum_strings import QubitString
from hqca.operators._operator import Operator as Op
from hqca.tools._stabilizer import Stabilizer
from hqca.transforms import *
from functools import partial
import hqca.config as config
config._use_multiprocessing=False
import pytest

def test_acse_h3():
    Tr,iTr = get_transform_from_symmetries(
            Transform=JordanWigner,
            symmetries=['IZIZIZ','IZIIZ','ZZZI'],
            qubits=[5,4,2],
            eigvals=[+1,-1,+1],
            )
    Tr,iTr = parity_free(3,3,+1,-1,JordanWigner)
    #op = Operator()
    #op+= Fermi(s='+p-iii',coeff=1)
    #op+= Fermi(s='-p+iii',coeff=1)
    #print(op.transform(Tr))
    #sys.exit()
    ham = MolecularHamiltonian(mol=advanced_mol(),transform=Tr)
    ins = PauliSet
    st = StorageACSE(ham,closed_ansatz=-2)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=4,
            provider='Aer')
    qs.initial_transform = iTr
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=Tr,use_multiprocessing=False,)
    tomoIm = StandardTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=Tr,use_multiprocessing=False,)
    proc = StandardProcess()
    norm = []

    for method in ['bfgs']:
        acse = RunACSE(
                st,qs,ins,processor=proc,
                method=method,
                update='quantum',
                opt_thresh=1e-2,
                S_thresh_rel=1.0,
                S_min=1e-6,
                convergence_type='norm',
                hamiltonian_step_size=0.000001,
                use_trust_region=True,
                max_iter=5,
                tomo_A = tomoIm,
                tomo_psi = tomoRe,
                trunc_method = 'delta',
                trunc_include = True,
                verbose=True,
                )
        acse.build()
        norm.append(acse.norm)
        da.expect(abs(acse.e0-acse.store.ei)<=1e-10)
        while not acse.total.done:
            print('--------------------')
            acse.next_step()
            print(acse.psi)
            for n,i in enumerate(acse._log_psi):
                print('vec n',n)
                nz = np.nonzero(i)
                for j in nz[0]:
                    print(j)
            assert len(acse.psi.A)==len(acse._log_psi)
        #msg = 'error in {} method'.format(method)
        #da.expect(abs(acse.e0-acse.store.H.ef)<=1e-3,msg)
    da.assert_expectations()

test_acse_h3()
