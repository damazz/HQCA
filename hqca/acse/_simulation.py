"""
RunACSE with default options in a lot of areas, and a generic approach suited for simulation
"""
from hqca.acse import *
from hqca.instructions import PauliSet
from hqca.transforms import *

class MolecularACSE(RunACSE):
    def __init__(self,
                 hamiltonian,
                 closed_ansatz=True,
                 update='quantum',
                 transform=JordanWigner,
                 qubit_transform=None,
                 initial_transform=None,
                 **kwargs,
                 ):
        st = StorageACSE(hamiltonian,closed_ansatz=closed_ansatz)
        qs = QuantumStorage()
        qs.set_algorithm(st)
        qs.set_backend(
            backend='statevector_simulator',
            Nq=len(next(iter(hamiltonian._qubOp)).s),
            provider='Aer')
        if not type(initial_transform)==type(None):
            qs.initial_transform=initial_transform
        tomoRe = ReducedTomography(qs)
        tomoRe.generate(real=True,imag=False,
            transform=transform,
                        )
        if update=='quantum':
            tomoA = ReducedTomography(qs)
            tomoA.generate(real=False,imag=True,
                transform=transform,
                            )
            kwargs['tomo_S'] = tomoA
        elif update in ['qubit','para']:
            qs.qubit_transform = qubit_transform
            tomoA = ReducedQubitTomography(qs)
            tomoA.generate(real=False,imag=True,
                           transform=qubit_transform)
            kwargs['tomo_S'] = tomoA

        tomoRe = ReducedTomography(qs)
        tomoRe.generate(real=True,imag=False,
            transform=transform,
                        )
        Ins = PauliSet
        args = (st,qs,Ins)
        kwargs['update']=update
        kwargs['tomo_Psi']=tomoRe
        RunACSE.__init__(self,*args,**kwargs)
        RunACSE.build(self)
