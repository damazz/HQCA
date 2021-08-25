from copy import copy as copy
import numpy as np
from functools import partial
from hqca.tomography import *
from hqca.operators import *

def _euler_step(acse):
    '''
    function of Store.build_trial_ansatz
    '''
    testS = copy(acse.A)
    for s in testS:
        s.c*=acse.delta
    acse.S = acse.S+testS
    ins = acse.Instruct(
            operator=acse.S,
            Nq=acse.QuantStore.Nq,
            quantstore=acse.QuantStore,
            )
    circ = StandardTomography(
            QuantStore=acse.QuantStore,
            preset=acse.tomo_preset,
            Tomo=acse.tomo_Psi,
            verbose=acse.verbose,
            )
    if not acse.tomo_preset:
        circ.generate(real=acse.Store.H.real,imag=acse.Store.H.imag)
    circ.set(ins)
    circ.simulate()
    circ.construct(processor=acse.process)
    en = np.real(acse.Store.evaluate(circ.rdm))
    acse.Store.update(circ.rdm)
    if acse.total.iter==0:
        if en<acse.e0:
            pass
        elif en>acse.e0:
            print('Euler step caused an increase in energy....switching.')
            acse.delta*=-1
            for s in testS:
                s.c*=-1
            acse.S+= testS
            ins = acse.Instruct(operator=acse.S,
                    Nq=acse.QuantStore.Nq,
                    quantstore=acse.QuantStore,
                    )
            circ = StandardTomography(
                    QuantStore=acse.QuantStore,
                    preset=acse.tomo_preset,
                    Tomo=acse.tomo_Psi,
                    verbose=acse.verbose,
                    )
            if not acse.tomo_preset:
                circ.generate(
                        real=acse.Store.H.real,
                        imag=acse.Store.H.imag)
            circ.set(ins)
            circ.simulate()
            circ.construct(processor=acse.process)
            en = np.real(acse.Store.evaluate(circ.rdm))
            acse.Store.update(circ.rdm)
    acse.circ = circ
