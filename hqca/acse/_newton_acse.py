from copy import deepcopy as copy
import numpy as np
from hqca.tomography import *
from hqca.core import *

def _newton_step(acse):
    coeff_best = 0.0
    rdm_best = copy(acse.Store.rdm)
    e_best = copy(acse.e_k)

    testS =  copy(acse.A)
    max_val = 0.0
    for s in testS:
        if abs(s.c)>abs(max_val):
            max_val = copy(s.c)
    if acse.verbose:
        print('Maximum value: {:+.10f}'.format(max_val))
        print('Running first point...')
    e1,rdm1 = acse._test_acse_function([acse.delta],testS)
    # 
    if e1<e_best:
        e_best = copy(e1)
        coeff_best = copy(acse.delta)
        rdm_best = copy(rdm1)
        acse.current_counts = acse.circ.operator_count
    if acse.verbose:
        print('Running second point...')
    e2,rdm2 = acse._test_acse_function([acse.d*acse.delta],testS)
    if e2<e_best:
        e_best = copy(e2)
        coeff_best = copy(acse.delta*acse.d)
        rdm_best = copy(rdm2)
        acse.current_counts = acse.circ.operator_count

    if acse.verbose:
        print('Energies: ',acse.e_k.real,e1,e2)
    g1,g2= e1-acse.e_k,e2-acse.e_k
    d2D = (2*g2-2*acse.d*g1)/(acse.d*acse.delta*acse.delta*(acse.d-1))
    d1D = (g1*acse.d**2-g2)/(acse.d*acse.delta*(acse.d-1))
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
        print('Step: {:.6f}, Largest: {:.6f}'.format(
            np.real(-d1D/d2D),
            np.real(max_val*d1D/d2D))
            )
    acse.grad = d1D
    acse.hess = d2D
    # 
    # setting limit for experimental and theoretical newton step coeff
    if acse.QuantStore.be_type in ['sv','qasm']:
        lim=1e-6
    else:
        lim= 0.1
    if acse.use_trust_region:
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
                ef,df = acse._test_acse_function([coeff],testS)
                if ef<e_best:
                    e_best = copy(ef)
                    rdm_best = copy(df)
                    coeff_best = copy(coeff)
                    acse.current_counts = acse.circ.operator_count
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
                #    acse.Store.update(df)
                trust_iter+=1
                if acse.QuantStore.be_type in ['nm','qc']:
                    if trust_iter>=1:
                        trust=True
                else:
                    if trust_iter>=2:
                        trust=True
                    #
        if abs(coeff_best)<lim and acse.QuantStore.be_type in ['nm','qc']:
            acse.accept_previous_step = False
            if acse.verbose:
                print('Rejecting Newton Step...')
        else:
            acse.accept_previous_step = True
            for f in testS:
                f.c*= coeff_best
            acse.S = acse.S+testS
            acse.Store.update(rdm_best)
    else:
        raise QuantumRunError('Why are you not using the trust region? \nSet use_trust_region=True')
        acse.S = acse.S+testS
        # eval energy is in check step
        Ins = acse.Instruct(
                operator=acse.S.op_form(),
                Nq=acse.QuantStore.Nq,
                quantstore=acse.QuantStore,
                )
        Psi= StandardTomography(
                QuantStore=acse.QuantStore,
                preset=acse.tomo_preset,
                Tomo=acse.tomo_Psi,
                verbose=acse.verbose,
                )
        if not acse.tomo_preset:
            Psi.generate(real=True,imag=False)
        Psi.set(Ins)
        Psi.simulate()
        Psi.construct(processor=acse.process)
        acse.Store.update(Psi.rdm)
        acse.current_counts = Psi.operator_count
        Psi.rdm.switch()
        acse.circ = Psi

