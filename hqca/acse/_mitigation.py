import numpy as np
import sys
from copy import deepcopy as copy

def check_mitigation(acse):
    '''
    '''
    if 'shift' in acse.qs.method:
        _calculate_zo_correction(acse)

def _calculate_zo_correction(acse):
    '''
    Refers to the forward-projected method

    Full issues a complete recalculation of each iterative point.

    Current only calculates the next point.

    Zero sets the shift to the Hartree Fock point by setting all terms to 0. 
    This is the fastest method but is limited.
    '''
    print('-- -- -- -- -- -- -- -- -- -- --')
    print('checking ansatz length')
    print('--------------')
    print('Assessing error mitigation.')
    if acse.qs.shift_protocol=='full':
        _find_zo_full(acse)
    elif acse.qs.shift_protocol=='current':
        _find_zo_current(acse)
    elif acse.qs.shift_protocol=='zero':
        _find_zo_zero(acse)
    else:
        pass


def _find_zo_full(acse):
    acse.qs.Gamma = None
    testS = copy(acse.A)
    currS = copy(acse.S)
    total = currS+testS
    s1 = copy(total)
    s1.truncate(1)
    for f in s1.A[-1]:
        f.c*=0.000001
    print('initial step: ')
    print(s1)
    Circ = acse._generate_circuit(s1)
    Gamma = acse.store.hf_rdm-Circ.rdm
    e0 = acse.store.evaluate(acse.store.hf_rdm)
    e1 = acse.store.evaluate(Circ.rdm)
    et = e1-e0
    print('Energies: ')
    print('E0 (HF): {:.8f}'.format(np.real(e0)))
    print('E1 (HF-qc): {:.8f}'.format(np.real(e1)))
    print('Energy shift: {:.8f}'.format(np.real(e1-e0)))
    print('- - - -')
    for d in range(1,len(total)):
        print('Depth: {}'.format(d))
        S0 = copy(total)
        S1 = copy(total)
        S0.truncate(d)
        S1.truncate(d+1)
        for f in S1.A[-1]:
            f.c*= 0.000001
        print('Adjusted operator 0: ')
        print(S0)
        print('Adjusted operator 1: ')
        print(S1)
        Circ0 = acse._generate_circuit(S0)
        Circ1 = acse._generate_circuit(S1)
        Gamma+= (Circ0.rdm-Circ1.rdm)
        e0 = acse.store.evaluate(Circ0.rdm)
        e1 = acse.store.evaluate(Circ1.rdm)
        print('Energies: ')
        print('E0 (qc): {:.8f}'.format(np.real(e0)))
        print('E1 (qc): {:.8f}'.format(np.real(e1)))
        print('Energy shift: {:.8f}'.format(np.real(e1-e0)))
        print('- - - -')
        et+= (e1-e0)
    #print('Total Gamma: ')
    #Gamma.analysis()
    print('----------------------------------')
    print('Total energy shift: {:.8f}'.format(np.real(et)))
    print('----------------------------------')
    acse.qs.Gamma = Gamma
    if acse.log:
        acse.log_Gamma.append(Gamma)


def _find_zo_current(acse):
    if type(acse.qs.Gamma)==type(None):
        testS = copy(acse.A)
        currS = copy(acse.S)
        S = currS+testS
        S.truncate(1)
        for f in S.A[0]:
            f.c*= 0.0001
        print('initial step: ')
        print(S)
        Circ1 = acse._generate_circuit(S)
        Circ2 = acse._generate_circuit(S)
        gam = (Circ1.rdm+Circ2.rdm)*0.5
        Gamma = acse.store.hf_rdm-gam

        e0 = acse.store.evaluate(acse.store.hf_rdm)
        e1 = acse.store.evaluate(gam)
        et = e1-e0
        acse.qs.Gamma = Gamma
        acse.S._store.append(Gamma)
        print('Energies: ')
        print('E0 (HF): {:.8f}'.format(np.real(e0)))
        print('E1 (HF-qc): {:.8f}'.format(np.real(e1)))
        print('Energy shift: {:.8f}'.format(np.real(et)))
        print('Norm: {}'.format(np.linalg.norm(Gamma.rdm)))
        print('- - - -')
        #Gamma.analysis()
    else:
        testA = copy(acse.A)
        currS = copy(acse.S)
        S = currS+testA
        if len(S)==len(currS):
            # if we are not changing depth....than we do not need to change Gamma
            # need to replace gamma on last step
            print('Gamma is sufficient. Continuing calculation.')
        else:
            for f in S.A[-1]:
                f.c*= 0.0001
            print('current S: ')
            print(S)
            #  we added one.....new one will be zero
            Circ1 = acse._generate_circuit(S)
            Circ2 = acse._generate_circuit(S)
            # actually...gamma should be the same...ish? 
            gam = (Circ1.rdm+Circ2.rdm)*0.5
            nGamma = copy(acse.store.rdm)-gam #-acse.qs.Gamma-acse.qs.Gamma)
            # 
            e0 = copy(acse.store.evaluate(acse.store.rdm))
            e1 = acse.store.evaluate(gam)
            print('Energies: ')
            print('E_n(e_n) (qc): {:.8f}'.format(np.real(e0)))
            print('E_n+1(0) (qc): {:.8f}'.format(np.real(e1)))
            print('Gamma shift: {:.8f}'.format(np.real(e1-e0)))
            print('Norm: {}'.format(np.linalg.norm(nGamma.rdm)))
            #nGamma.analysis()
            acse.qs.Gamma+= nGamma
            print('- - - -')
            print('Total energy: {:.8f}'.format(
                acse.store.evaluate(acse.qs.Gamma).real))
            print('Norm: {}'.format(np.linalg.norm(acse.qs.Gamma.rdm)))
            print('- - - -')

def _find_zo_zero(acse):
    acse.qs.Gamma = None
    testS = copy(acse.A)
    currS = copy(acse.S)
    total = currS+testS
    S = copy(total)
    # s1.truncate(1) # 
    for f in S:
        f.c*=0.0001
        print(f)
    print('Setting shift to H0:')
    print(s1)
    Circ = acse._generate_circuit(S)
    Gamma = acse.store.hf_rdm-Circ.rdm
    e0 = acse.store.evaluate(acse.store.hf_rdm)
    e1 = acse.store.evaluate(Circ.rdm)
    et = e1-e0
    print('Energies: ')
    print('E0 (HF): {:.8f}'.format(np.real(e0)))
    print('E1 (HF-qc): {:.8f}'.format(np.real(e1)))
    print('Energy shift: {:.8f}'.format(np.real(e1-e0)))
    print('- - - -')
    print('Total Gamma: ')
    Gamma.analysis()
    print('----------------------------------')
    print('Total energy shift: {:.8f}'.format(np.real(et)))
    print('----------------------------------')
    acse.qs.Gamma = Gamma
    if acse.log:
        acse.log_Gamma.append(Gamma)
