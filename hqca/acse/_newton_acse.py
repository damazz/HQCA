from copy import deepcopy as copy
import numpy as np
from hqca.tomography import *
from hqca.core import *


def _newton_step(acse): 
    # TODO: update or remove newton step option...potentially simplify
    coeff_best = 0.0
    e0 = copy(acse.e_k)
    testS = copy(acse.A)

    op1 = acse.psi+testS*acse.epsilon
    c1 = acse._generate_circuit(op1)
    e1 = np.real(acse.store.evaluate(c1.rdm))
    #
    op2 = acse.psi+testS*acse.d*acse.epsilon
    c2 = acse._generate_circuit(op2)
    e2 = np.real(acse.store.evaluate(c2.rdm))
    print('Energies: ',acse.e_k.real,e1,e2)
    g1,g2= e1-acse.e_k,e2-acse.e_k
    d2D = (2*g2-2*acse.d*g1)/(acse.d*acse.epsilon*acse.epsilon*(acse.d-1))
    d1D = (g1*acse.d**2-g2)/(acse.d*acse.epsilon*(acse.d-1))
    if abs(d2D)<1e-16:
        d2D=1e-16
    elif abs(d1D)<1e-16:
        d1D = 1e-16
    if acse.verbose:
        print('')
        print('--- Newton Step --- ')
        print('dE(d1): {:.10f},  dE(d2): {:.10f}'.format(
            np.real(g1),np.real(g2)))
        print('dE\'(0): {:.10f}, dE\'\'(0): {:.10f}'.format(
            np.real(d1D),np.real(d2D)))
        print('Step: {:.6f}'.format(
            np.real(-d1D/d2D)
            ))
    acse.grad = d1D
    acse.hess = d2D
    # 
    # setting limit for experimental and theoretical newton step coeff
    if acse.qs.be_type in ['sv','qasm']:
        lim=1e-6
    else:
        lim= 0.1
    if not acse.use_trust_region:
        raise QuantumRunError('Why are you not using the trust region? \nSet use_trust_region=True')

    if acse.verbose:
        print('Carrying out trust region step:')
    if acse.hess<0:
        if acse.verbose:
            print('Hessian non-positive.')
            print('Trying again.')
    else:
        trust = False
        nv = acse.tr_nv
        ns = acse.tr_ns
        gi = acse.tr_gi
        gd = acse.tr_gd
        trust_iter = 0
        while not trust: # perform sub routine
            if abs(acse.grad/acse.hess)<acse.tr_Del:
                if acse.verbose:
                    print('Within trust region.')
                # found ok answer! 
                coeff = -acse.grad/acse.hess
                lamb=1
            else:
                if acse.verbose:
                    print('Outside trust region.')
                if -acse.grad/acse.hess<0:
                    coeff = acse.tr_Del*(-1)
                else:
                    coeff = acse.tr_Del
            coeff = np.real(coeff)
            opf = acse.psi+testS*coeff
            cf = acse._generate_circuit(opf)
            ef = np.real(acse.store.evaluate(cf.rdm))
            if acse.verbose:
                print('Current: {:.10f}'.format(np.real(ef)))
            def m_qk(s):
                return acse.e_k + s*acse.grad+0.5*s*acse.hess*s
            acse.tr_taylor =  acse.e_k-m_qk(coeff)
            acse.tr_object = acse.e_k-ef
            if acse.verbose:
                print('Coefficient: {}'.format(coeff))
                print('Taylor series step: {:.14f}'.format(
                    np.real(acse.tr_taylor)))
                print('Objective fxn step: {:.14f}'.format(
                    np.real(acse.tr_object)))
            if abs(acse.tr_object)<=acse.tr_obj_crit:
                trust=True
                if acse.verbose:
                    print('Convergence in objective function.')
            elif abs(acse.tr_taylor)<=acse.tr_ts_crit:
                trust=True
                if acse.verbose:
                    print('Convergence in Taylor series model.')
            else:
                # now, adjusting trust region for future steps
                rho = acse.tr_object/acse.tr_taylor
                if rho>=nv:
                    if acse.verbose:
                        print('Result in trust region. Increasing TR.')
                    trust = True
                    acse.tr_Del*=gi
                elif rho>=ns:
                    if acse.verbose:
                        print('Trust region held. Continuing.')
                    trust = True
                else:
                    acse.tr_Del*=gd
                    if acse.verbose:
                        print('Trust region did not hold. Shrinking.')
                        print('Trial energy: {:.10f}'.format(ef))
            if acse.verbose:
                print('Current trust region: {:.14f}'.format(
                    np.real(acse.tr_Del)))
            #if abs(coeff)>0.1:
            #    acse.store.update(df)
            trust_iter+=1
            if acse.qs.be_type in ['nm','qc']:
                if trust_iter>=1:
                    trust=True
            else:
                if trust_iter>=2:
                    trust=True
                #
    if abs(coeff_best)<lim and acse.qs.be_type in ['nm','qc']:
        acse.accept_previous_step = False
        if acse.verbose:
            print('Rejecting Newton Step...')
        return

    acse.accept_previous_step = True
    lowest = min(e0,e1,e2,ef)
    if lowest==e1:
        best = acse.epsilon
    elif lowest==e2:
        best = acse.epsilon*acse.d
    elif lowest==ef:
        best = coeff

    acse.psi = acse.psi+testS*coeff
    circ = acse._generate_circuit(opf)
    acse.store.update(circ.rdm)
    acse.circ = circ
