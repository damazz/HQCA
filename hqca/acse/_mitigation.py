import numpy as np
from copy import deepcopy as copy

def check_mitigation(acse):
    '''
    '''
    if 'shift' in acse.QuantStore.method:
        _calculate_zo_correction(acse)

def _calculate_zo_correction(acse):
    print('-- -- -- -- -- -- -- -- -- -- --')
    print('checking ansatz length')
    print('--------------')
    print('Assessing error mitigation.')
    if acse.QuantStore.shift_protocol=='full':
        _find_zo_full(acse)
    elif acse.QuantStore.shift_protocol=='current':
        _find_zo_current(acse)
    elif acse.QuantStore.shift_protocol=='zero':
        _find_zo_zero(acse)
    else:
        pass


def _find_zo_full(acse):
    acse.QuantStore.Gamma = None
    testS = copy(acse.A)
    currS = copy(acse.S)
    total = currS+testS
    s1 = copy(total)
    s1.truncate(1)
    for f in s1.A[-1]:
        f.c*=0.000001
    print('initial step: ')
    print(s1)
    Circ = acse._generate_real_circuit(s1)
    Gamma = acse.Store.hf_rdm-Circ.rdm
    e0 = acse.Store.evaluate(acse.Store.hf_rdm)
    e1 = acse.Store.evaluate(Circ.rdm)
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
        Circ0 = acse._generate_real_circuit(S0)
        Circ1 = acse._generate_real_circuit(S1)
        Gamma+= (Circ0.rdm-Circ1.rdm)
        e0 = acse.Store.evaluate(Circ0.rdm)
        e1 = acse.Store.evaluate(Circ1.rdm)
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
    acse.QuantStore.Gamma = Gamma
    if acse.log:
        acse.log_Gamma.append(Gamma)


def _find_zo_current(acse):
    if type(acse.QuantStore.Gamma)==type(None):
        testS = copy(acse.A)
        currS = copy(acse.S)
        S = currS+testS
        S.truncate(1)
        for f in S.A[0]:
            f.c*= 0.0001
        print('initial step: ')
        print(S)
        Circ = acse._generate_real_circuit(S)
        Gamma = acse.Store.hf_rdm-Circ.rdm
        e0 = acse.Store.evaluate(acse.Store.hf_rdm)
        e1 = acse.Store.evaluate(Circ.rdm)
        et = e1-e0
            # initial
        acse.QuantStore.Gamma = Gamma
        acse.S._store.append(Gamma)
        print('Energies: ')
        print('E0 (HF): {:.8f}'.format(np.real(e0)))
        print('E1 (HF-qc): {:.8f}'.format(np.real(e1)))
        print('Energy shift: {:.8f}'.format(np.real(et)))
        print('- - - -')
        Gamma.analysis()
    else:
        testS = copy(acse.A)
        currS = copy(acse.S)
        S = currS+testS
        for f in S.A[-1]:
            f.c*= 0.0001
        if len(S)==len(currS):
            # if we are not changing depth....than we do not need to change Gamma
            # need to replace gamma on last step
            print('Gamma is sufficient. Continuing calculation.')
        else:
            #  we added one.....new one will be zero
            Circ = acse._generate_real_circuit(S)
            # actually...gamma should be the same...ish? 
            nGamma = (acse.Store.rdm-Circ.rdm)#-acse.QuantStore.Gamma-acse.QuantStore.Gamma)
            # 
            e0 = copy(acse.Store.evaluate(acse.Store.rdm))
            e1 = acse.Store.evaluate(Circ.rdm)
            print('Energies: ')
            print('E0 (qc): {:.8f}'.format(np.real(e0)))
            print('E1 (qc): {:.8f}'.format(np.real(e1)))
            print('Gamma shift: {:.8f}'.format(np.real(e0-e1)))
            nGamma.analysis()
            acse.QuantStore.Gamma+= nGamma
            #print('- - - -')
            #print('Total energy: {:.8f}'.format(
            #    acse.Store.evaluate(acse.QuantStore.Gamma).real))
            #print('- - - -')

def _find_zo_zero(acse):
    acse.QuantStore.Gamma = None
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
    Circ = acse._generate_real_circuit(S)
    Gamma = acse.Store.hf_rdm-Circ.rdm
    e0 = acse.Store.evaluate(acse.Store.hf_rdm)
    e1 = acse.Store.evaluate(Circ.rdm)
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
    acse.QuantStore.Gamma = Gamma
    if acse.log:
        acse.log_Gamma.append(Gamma)
