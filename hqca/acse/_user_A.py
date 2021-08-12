'''
way to include a user-specified A matrix, with a quantum-mechanical generation of A
'''

import numpy as np
import sys
from hqca.core import *
from hqca.tools import *
from hqca.tomography import *
from hqca.operators import *

def findUserA(
        operator,
        process,
        instruct,
        store,
        quantstore,
        hamiltonian_step_size,
        tomo=None,
        verbose=False,
        matrix=False,
        **kwargs
        ):
    """
    :param operator: operator fed into Instruct
    :param process: for running circuit
    :param instruct: ^
    :param store: ^
    :param quantstore: ^
    :param hamiltonian_step_size: specifiees accuracy of trotterization
    :param tomo: specified tomography object
    :param verbose:
    :param matrix:
    :return:
    """
    Psi1 = instruct(
        operator=operator,
        Nq=quantstore.Nq,
        quantstore=quantstore,
        propagate=True,
        HamiltonianOperator=store.H.qubit_operator,
        scaleH=hamiltonian_step_size,
    )
    H2 = store.H.qubit_operator * (-1)
    Psi2 = instruct(
        operator=operator,
        Nq=quantstore.Nq,
        quantstore=quantstore,
        propagate=True,
        HamiltonianOperator=H2,
        scaleH=hamiltonian_step_size,
    )
    if type(tomo) == type(None):
        raise TomographyError('Need to provide a tomography object.')
    else:
        Circ1 = QubitTomography(
            tomo_type='pauli',
            QuantStore=quantstore,
            preset=True,
            Tomo=tomo,
            verbose=verbose,
        )
        Circ2 = QubitTomography(
            QuantStore=quantstore,
            tomo_type='pauli',
            preset=True,
            Tomo=tomo,
            verbose=verbose,
        )
    Circ1.set(Psi1)
    Circ2.set(Psi2)
    if verbose:
        print('Running circuits...')
    Circ1.simulate(verbose=verbose)
    Circ2.simulate(verbose=verbose)
    if verbose:
        print('Constructing the RDMs...')
    Circ1.construct(processor=process,vector=matrix)
    Circ2.construct(processor=process,vector=matrix)
    # rdm = newCirc.rdm.rdm-store.rdm.rdm
    hss = (1 / (2 * hamiltonian_step_size))
    res = np.imag(Circ1.result - Circ2.result) * hss
    #print(res)
    return res
