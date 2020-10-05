import pickle
from copy import deepcopy as copy
import threading
import os, sys
from importlib import reload
import numpy as np
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from functools import reduce,partial
import datetime
import sys
from scipy import stats
from hqca.core import *
from hqca.acse._class_S_acse import *
from hqca.acse._ansatz_S import *
from hqca.acse._quant_S_acse import *
from hqca.tools import *
from optss import *

class RunACSE(QuantumRun):
    '''
    '''
    def __init__(self,
            Storage, #instance
            QuantStore, #instance
            Instructions, #class?
            **kw
            ):
        self.Store = Storage
        self.QuantStore = QuantStore
        self.Instruct = Instructions
        self._update_acse_kw(**kw)

    def _update_acse_kw(self,
            method='newton',
            update='quantum',
            opt_thresh=1e-8,
            max_iter=100,
            trotter=1,
            pr_a=1,
            ansatz_depth=1,
            commutative_ansatz=False,
            S_ordering='default',
            S_thresh_rel=0.1,
            S_min=1e-10,
            convergence_type='default',
            hamiltonian_step_size=0.1,
            restrict_S_size=0.5,
            propagation='trotter',
            separate_hamiltonian=False,
            verbose=True,
            tomo_S=None,
            tomo_Psi=None,
            statistics=False,
            processor=None,
            max_depth=None,
            max_constant_depth=None,
            **kw):
        '''
        Updates the ACSE keywords. 
        '''
        if update in ['quantum','Q','q']:
            self.acse_update = 'q'
        elif update in ['class','classical','c','C']:
            self.acse_update ='c'
        if not method in ['NR','EM','opt','trust','newton','euler','line']:
            print('Specified method not valid. Update acse_kw: \'method\'')
            sys.exit()
        self.process =processor
        self.verbose=verbose
        self.stats=statistics
        self.acse_method = method
        self.S_trotter= ansatz_depth
        self.S_commutative = commutative_ansatz
        self.N_trotter = trotter
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_constant_depth = max_constant_depth
        self.crit = opt_thresh
        self.hamiltonian_step_size = hamiltonian_step_size
        self.sep_hamiltonian = separate_hamiltonian
        self.S_thresh_rel = S_thresh_rel
        self.S_min = S_min
        self.delta = restrict_S_size
        self.propagate_method=propagation
        self.S_ordering = S_ordering
        self._conv_type = convergence_type
        self.tomo_S=tomo_S
        self.tomo_Psi=tomo_Psi
        if type(self.tomo_Psi)==type(None):
            self.tomo_preset=False
        else:
            self.tomo_preset=True
        print('\n\n')
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('      --  ACSE KEYWORDS --      ')
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('algorithm')
        print('-- -- -- --')
        print('ACSE method: {}'.format(method))
        print('ACSE update: {}'.format(update))
        print('max iterations: {}'.format(max_iter))
        print('max depth: {}'.format(max_depth))
        print('convergence type: {}'.format(convergence_type))
        print('convergence threshold: {}'.format(self.crit))

        print('-- -- -- --')
        print('ACSE solution')
        print('-- -- -- --')
        if self.acse_update=='q':
            print('trotter-H: {}'.format(trotter))
            print('hamiltonian delta: {}'.format(hamiltonian_step_size))
        print('S rel threshold: {}'.format(S_thresh_rel))
        print('S max threshold: {}'.format(S_min))
        print('-- -- -- --')
        print('ansatz')
        print('-- -- -- --')
        print('S epsilon: {}'.format(self.delta))
        print('trotter-S: {}'.format(ansatz_depth))
        print('-- -- -- --')
        print('optimization')
        print('-- -- -- --')

        if self.acse_method=='newton':
            self._update_acse_newton(**kw)
        elif self.acse_method in ['line','opt']:
            self._update_acse_opt(**kw)
        self.grad=0

    def _update_acse_opt(self,
            optimizer='nm',
            optimizer_threshold='default',
            **kw,
            ):
        print('optimizer : {}'.format(optimizer))
        print('optimizer threshold: {}'.format(optimizer_threshold))
        self._optimizer = optimizer
        self._opt_thresh = optimizer_threshold



    def _update_acse_newton(self,
            use_trust_region=False,
            newton_step=2,
            initial_trust_region=np.pi/2,
            tr_taylor_criteria=1e-10,
            tr_objective_criteria=1e-10,
            tr_gamma_inc=2,
            tr_gamma_dec=0.5,
            tr_nu_accept=0.9,
            tr_nu_reject=0.1
            ):
        self.use_trust_region = use_trust_region
        self.d = newton_step #for estimating derivative
        self.tr_ts_crit = tr_taylor_criteria
        self.tr_obj_crit = tr_objective_criteria
        self.tr_Del  = initial_trust_region # trust region
        self.tr_gi = tr_gamma_inc
        self.tr_gd = tr_gamma_dec
        self.tr_nv = tr_nu_accept # very good?
        self.tr_ns = tr_nu_reject #shrink
        print('newton step: {}'.format(newton_step))
        print('newton trust region: {}'.format(use_trust_region))
        print('trust region: {:.6f}'.format(initial_trust_region))
        self.tr_taylor = 1
        self.tr_object = 1


    def build(self,log=False):
        if self.verbose:
            print('\n\n')
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('building the ACSE run')
            print('-- -- -- -- -- -- -- -- -- -- --')

        try:
            self.Store.H
            self.QuantStore.Nq
            #self.Instruct.gates
        except Exception as e:
            print(e)
            sys.exit('Build error.')
        if self.Store.use_initial:
            self.S = copy(self.Store.S)
            for s in self.S:
                #s.qCo*=self.delta
                s.c*=self.delta
            ins = self.Instruct(
                    operator=self.S,
                    Nq=self.QuantStore.Nq,
                    depth=self.S_trotter,
                    quantstore=self.QuantStore,)
            circ = StandardTomography(
                    QuantStore=self.QuantStore,
                    preset=self.tomo_preset,
                    Tomo=self.tomo_Psi,
                    verbose=self.verbose,
                    )
            if not self.tomo_preset:
                circ.generate(real=self.Store.H.real,imag=self.Store.H.imag)
            circ.set(ins)
            circ.simulate()
            circ.construct(processor=self.process)
            en = np.real(self.Store.evaluate(circ.rdm))
            self.e0 = np.real(en)
            self.ei = np.real(en)
            print('Initial energy: {:.8f}'.format(self.e0))
            self.Store.rdm = circ.rdm
            print('S: ')
            print(self.S)
            print('Initial density matrix.')
            circ.rdm.contract()
            print(np.real(circ.rdm.rdm))
        else:
            self.S = copy(self.Store.S)
            self.e0 = self.Store.e0
            self.ei = self.Store.ei
            print('taking energy from storage')
            print('initial energy: {:.8f}'.format(np.real(self.e0)))
        self.best = self.e0
        self.best_avg = self.e0

        self.log = log
        self.log_depth = []
        if self.log:
            self.log_rdm = [self.Store.rdm]
            self.log_A = []
            self.log_Gamma = []
            self.log_S  = []
        self.log_norm = []
        self.log_E = [self.e0]
        self.log_G = []
        try:
            self.log_ci = [self.ci]
        except Exception: 
            pass
        self.total=Cache()
        self.__get_S()
        if self.log:
            self.log_A.append(copy(self.A))
        #self._check_norm(self.A)
        # check if ansatz will change length
        print('||A||: {:.10f}'.format(np.real(self.norm)))
        print('-- -- -- -- -- -- -- -- -- -- --')
        self.built=True


    def __get_S(self):
        if self.acse_update=='q':
            A_sq = findSPairsQuantum(
                    self.QuantStore.op_type,
                    operator=self.S,
                    process=self.process,
                    instruct=self.Instruct,
                    store=self.Store,
                    quantstore=self.QuantStore,
                    S_min=self.S_min,
                    ordering=self.S_ordering,
                    trotter_steps=self.N_trotter,
                    hamiltonian_step_size=self.hamiltonian_step_size,
                    propagate_method=self.propagate_method,
                    depth=self.S_trotter,
                    separate_hamiltonian=self.sep_hamiltonian,
                    verbose=self.verbose,
                    tomo=self.tomo_S,
                    )
        elif self.acse_update=='c':
            A_sq  = findSPairs(
                    self.Store,
                    self.QuantStore,
                    S_min=self.S_min,
                    verbose=self.verbose,
                    )
        max_val,norm = 0,0
        new = Operator()
        for op in A_sq:
            norm+= op.norm
            if abs(op.c)>=abs(max_val):
                max_val = copy(op.c)
        for op in A_sq:
            if abs(op.c)>=abs(self.S_thresh_rel*max_val):
                new+= op
        if self.S_commutative:
            new.ca=True
        else:
            new.ca=False
        self.norm = norm**(0.5)
        self.A = new
        if self.verbose:
            print('qubit A operator: ')
            print(self.A)
        print('-- -- -- -- -- -- -- -- -- -- --')


    def _run_acse(self):
        '''
        Function to the run the ACSE algorithm

        Note, the algorithm is configured to optimize the energy, and then
        calculate the residual of the ACSE.
        '''
        if self.verbose:
            print('\n\n')
        self._check_length()
        try:
            self.built
        except AttributeError:
            sys.exit('Not built! Run acse.build()')
        if self.acse_method in ['NR','newton']:
            self.__newton_acse()
        elif self.acse_method in ['default','em','EM','euler']:
            self.__euler_acse()
        elif self.acse_method in ['line']:
            self.__opt_line_acse()
        self.__get_S()
        #self._check_norm(self.A)
        # check if ansatz will change length
        if self.log:
            self.log_rdm.append(self.Store.rdm)
            self.log_A.append(copy(self.A))
            self.log_S.append(copy(self.S))

    def _check_length(self,full=True):
        qsp = self.QuantStore.post
        try:
            met = self.QuantStore.method in ['shift']
        except Exception:
            met = False
        if full and qsp and met:
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('checking ansatz length')
            print('--------------')
            print('recalculating Gamma error mitigation.')
            self.QuantStore.Gamma = None
            testS = copy(self.A)
            currS = copy(self.S)
            total = currS+testS
            s1 = copy(total)
            s1.truncate(1)
            for f in s1.A[-1]:
                f.c*=0.000001
            print('initial step: ')
            print(s1)

            I0=self.Instruct(
                    operator=s1,
                    Nq=self.QuantStore.Nq,
                    quantstore=self.QuantStore,
                    depth=self.S_trotter,
                    )
            Circ= StandardTomography(
                    QuantStore=self.QuantStore,
                    preset=self.tomo_preset,
                    Tomo=self.tomo_Psi,
                    #verbose=self.verbose,
                    verbose=False,
                    )
            if not self.tomo_preset:
                Circ.generate(
                        real=self.Store.H.real,
                        imag=self.Store.H.imag)
            Circ.set(I0)
            Circ.simulate()
            Circ.construct(processor=self.process)
            Gamma = self.Store.hf_rdm-Circ.rdm
            e0 = self.Store.evaluate(self.Store.hf_rdm)
            e1 = self.Store.evaluate(Circ.rdm)
            et = e1-e0
            print('Energies: ')
            print('E0 (HF): {:.8f}'.format(np.real(e0)))
            print('E1 (HF-qc): {:.8f}'.format(np.real(e1)))
            print('Energy shift: {:.8f}'.format(np.real(e1-e0)))
            print('- - - -')
            for d in range(1,total.d):
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
                I0=self.Instruct(
                        operator=S0,
                        Nq=self.QuantStore.Nq,
                        quantstore=self.QuantStore,
                        depth=self.S_trotter,
                        )
                I1=self.Instruct(
                        operator=S1,
                        Nq=self.QuantStore.Nq,
                        quantstore=self.QuantStore,
                        depth=self.S_trotter,
                        )
                Circ0= StandardTomography(
                        QuantStore=self.QuantStore,
                        preset=self.tomo_preset,
                        Tomo=self.tomo_Psi,
                        #verbose=self.verbose,
                        verbose=False,
                        )
                if not self.tomo_preset:
                    Circ0.generate(
                            real=self.Store.H.real,
                            imag=self.Store.H.imag)
                Circ0.set(I0)
                Circ0.simulate()
                Circ0.construct(processor=self.process)
                Circ1= StandardTomography(
                        QuantStore=self.QuantStore,
                        preset=self.tomo_preset,
                        Tomo=self.tomo_Psi,
                        #verbose=self.verbose,
                        verbose=False,
                        )
                if not self.tomo_preset:
                    Circ1.generate(
                            real=self.Store.H.real,
                            imag=self.Store.H.imag)
                Circ1.set(I1)
                Circ1.simulate()
                Circ1.construct(processor=self.process)
                Gamma+= (Circ0.rdm-Circ1.rdm)
                e0 = self.Store.evaluate(Circ0.rdm)
                e1 = self.Store.evaluate(Circ1.rdm)
                print('Energies: ')
                print('E0 (qc): {:.8f}'.format(np.real(e0)))
                print('E1 (qc): {:.8f}'.format(np.real(e1)))
                print('Energy shift: {:.8f}'.format(np.real(e1-e0)))
                print('- - - -')
                et+= (e1-e0)
            print('Total Gamma: ')
            Gamma.analysis()
            print('----------------------------------')
            print('Total energy shift: {:.8f}'.format(np.real(et)))
            print('----------------------------------')
            self.QuantStore.Gamma = Gamma
            if self.log:
                self.log_Gamma.append(Gamma)

    def _check_norm(self,testS):
        '''
        evaluate norm of S calculation
        '''
        self.norm = 0
        for item in testS.op:
            self.norm+= item.norm
        self.norm = self.norm**(0.5)

    def __opt_line_acse(self):
        '''
        '''
        testS = copy(self.A)
        self._opt_log = []
        self._opt_en  = []
        func = partial(self.__opt_acse_function,newS=testS)
        if self._opt_thresh=='default':
            thresh = self.delta/4
        else:
            thresh = self._opt_thresh
        opt = Optimizer(self._optimizer,
                function=func,
                verbose=True,
                shift= -1.01*self.delta,
                initial_conditions='old',
                unity=self.delta,
                conv_threshold=thresh,
                diagnostic=True,
                )
        opt.initialize([self.delta])
        # use if nelder mead
        print('Initial Simplex: ')
        for x,e in zip(opt.opt.simp_x,opt.opt.simp_f):
            print(x,e)
        opt.run()
        # run optimization, then choose best run
        for r,e in zip(self._opt_log,self._opt_en):
            if abs(e-opt.opt.best_f)<=1e-5:
                self.Store.update(r.rdm)
        for s in testS:
            s.c*=opt.opt.best_x[0]
        self.S = self.S + testS
        print(self.S)

    def __euler_acse(self):
        '''
        function of Store.build_trial_ansatz
        '''
        testS = copy(self.A)
        # where is step size from? 
        # test procedure
        for s in testS:
            #s.qCo*=self.delta
            s.c*=self.delta
        self.S = self.S+testS
        ins = self.Instruct(
                operator=self.S,
                Nq=self.QuantStore.Nq,
                depth=self.S_trotter,
                quantstore=self.QuantStore,
                )
        circ = StandardTomography(
                QuantStore=self.QuantStore,
                preset=self.tomo_preset,
                Tomo=self.tomo_Psi,
                verbose=self.verbose,
                )
        if not self.tomo_preset:
            circ.generate(real=self.Store.H.real,imag=self.Store.H.imag)
        circ.set(ins)
        circ.simulate()
        circ.construct(processor=self.process)
        if self.stats==False:
            pass
        else:
            if self.stats=='N':
                self.ci = circ.evaluate_error(
                        numberOfSamples=256,
                        sample_size=2048,
                        f=self._particle_number)
            elif self.stats in ['E','en']:
                self.ci = circ.evaluate_error(
                        numberOfSamples=256,
                        sample_size=2048,
                        f=self.Store.evaluate)
            print('Variance in {}: {:.6f} (CLT)'.format(
                self.stats,np.real(self.ci)))
            print('')
        en = np.real(self.Store.evaluate(circ.rdm))
        self.Store.update(circ.rdm)
        if self.total.iter==0:
            if en<self.e0:
                pass
            elif en>self.e0:
                self.delta*=-1
                for s in testS:
                    #s.qCo*=-2
                    s.c*=-2
                self.S+= testS
                ins = self.Instruct(operator=self.S,
                        Nq=self.QuantStore.Nq,
                        depth=self.S_trotter,
                        quantstore=self.QuantStore,
                        )
                circ = StandardTomography(
                        QuantStore=self.QuantStore,
                        preset=self.tomo_preset,
                        Tomo=self.tomo_Psi,
                        verbose=self.verbose,
                        )
                if not self.tomo_preset:
                    circ.generate(
                            real=self.Store.H.real,
                            imag=self.Store.H.imag)
                circ.set(ins)
                circ.simulate()
                circ.construct(processor=self.process)
                en = np.real(self.Store.evaluate(circ.rdm))
                self.Store.update(circ.rdm)
        self.circ = circ

    def __opt_acse_function(self,parameter,newS=None,verbose=False):
        testS = copy(newS)
        currS = copy(self.S)
        for f in testS:
            f.c*= parameter[0]
        temp = currS+testS
        tIns =self.Instruct(
                operator=temp,
                Nq=self.QuantStore.Nq,
                quantstore=self.QuantStore,
                depth=self.S_trotter,
                )
        tCirc= StandardTomography(
                QuantStore=self.QuantStore,
                preset=self.tomo_preset,
                Tomo=self.tomo_Psi,
                verbose=self.verbose,
                )
        if not self.tomo_preset:
            tCirc.generate(
                    real=self.Store.H.real,
                    imag=self.Store.H.imag)
        tCirc.set(tIns)
        tCirc.simulate()
        tCirc.construct(processor=self.process)
        en = np.real(self.Store.evaluate(tCirc.rdm))
        self._opt_log.append(tCirc)
        self._opt_en.append(en)
        return en

    def __test_acse_function(self,parameter,newS=None,verbose=False):
        testS = copy(newS)
        currS = copy(self.S)
        for f in testS:
            f.c*= parameter[0]
        temp = currS+testS
        tIns =self.Instruct(
                operator=temp,
                Nq=self.QuantStore.Nq,
                quantstore=self.QuantStore,
                depth=self.S_trotter,
                )
        tCirc= StandardTomography(
                QuantStore=self.QuantStore,
                preset=self.tomo_preset,
                Tomo=self.tomo_Psi,
                verbose=self.verbose,
                )
        if not self.tomo_preset:
            tCirc.generate(
                    real=self.Store.H.real,
                    imag=self.Store.H.imag)
        tCirc.set(tIns)
        tCirc.simulate()
        tCirc.construct(processor=self.process)
        if self.stats==False:
            pass
        else:
            if self.stats=='N':
                self.ci = tCirc.evaluate_error(
                        numberOfSamples=256,
                        sample_size=2048,
                        f=self._particle_number)
            elif self.stats in ['E','en']:
                self.ci = tCirc.evaluate_error(
                        numberOfSamples=256,
                        sample_size=2048,
                        f=self.Store.evaluate)
            print('Variance in {}: {:.6f} (CLT)'.format(
                self.stats,np.real(self.ci)))
            print('')
        en = np.real(self.Store.evaluate(tCirc.rdm))
        self.circ = tCirc
        return en,tCirc.rdm

    def _particle_number(self,rdm):
        return rdm.trace()

    def __newton_acse(self):
        testS =  copy(self.A)
        max_val = 0
        for s in testS.op:
            if abs(s.c)>abs(max_val):
                max_val = copy(s.c)
        print('Maximum value: {:+.10f}'.format(max_val))
        print('Running first point...')
        e1,rdm1 = self.__test_acse_function([self.delta],testS)
        print('Running second point...')
        e2,rdm2 = self.__test_acse_function([self.d*self.delta],testS)
        print('Energies: ',self.e0,e1,e2)
        g1,g2= e1-self.e0,e2-self.e0
        d2D = (2*g2-2*self.d*g1)/(self.d*self.delta*self.delta*(self.d-1))
        d1D = (g1*self.d**2-g2)/(self.d*self.delta*(self.d-1))
        if abs(d2D)<1e-16:
            d2D=1e-16
        elif abs(d1D)<1e-16:
            d1D = 1e-16
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
        self.grad = d1D
        self.hess = d2D
        if self.use_trust_region:
            print('Carrying out trust region step:')
            if self.hess<0:
                print('Hessian non-positive. Taking Euler step.')
                if e2<e1:
                    coeff = self.delta*self.d
                    self.Store.update(rdm2)
                else:
                    coeff = self.delta
                    self.Store.update(rdm1)
            else:
                trust = False
                nv = self.tr_nv
                ns = self.tr_ns
                gi = self.tr_gi
                gd = self.tr_gd
                trust_iter = 0
                while not trust: # perform sub routine
                    if abs(self.grad/self.hess)<self.tr_Del:
                        print('Within trust region.')
                        # found ok answer! 
                        coeff = -self.grad/self.hess
                        lamb=1
                    else:
                        print('Outside trust region.')
                        #lamb = -self.grad/self.tr_Del-self.hess
                        #print('Lambda: {}'.format(lamb))
                        #coeff = -self.grad/(self.hess+lamb)
                        if -self.grad/self.hess<0:
                            coeff = self.tr_Del*(-1)
                        else:
                            coeff = self.tr_Del

                    ef,df = self.__test_acse_function([coeff],testS)
                    print('Current: {:.10f}'.format(np.real(ef)))
                    def m_qk(s):
                        return self.e0 + s*self.grad+0.5*s*self.hess*s
                    self.tr_taylor =  self.e0-m_qk(coeff)
                    self.tr_object = self.e0-ef
                    print('Coefficient: {}'.format(coeff))
                    print('Taylor series step: {:.14f}'.format(
                        np.real(self.tr_taylor)))
                    print('Objective fxn step: {:.14f}'.format(
                        np.real(self.tr_object)))
                    if abs(self.tr_object)<=self.tr_obj_crit:
                        trust=True
                        print('Convergence in objective function.')
                    elif abs(self.tr_taylor)<=self.tr_ts_crit:
                        trust=True
                        print('Convergence in Taylor series model.')
                    else:
                        rho = self.tr_object/self.tr_taylor
                        if rho>=nv:
                            print('Result in trust region. Increasing TR.')
                            trust = True
                            self.tr_Del*=gi
                        elif rho>=ns:
                            print('Trust region held. Continuing.')
                            trust = True
                        else:
                            self.tr_Del*=gd
                            print('Trust region did not hold. Shrinking.')
                            print('Trial energy: {:.10f}'.format(ef))
                    print('Current trust region: {:.14f}'.format(
                        np.real(self.tr_Del)))
                    #print('Rho: {:.10f},Num: {:.16f}, Den: {:.16f}'.format(
                    #    np.real(rho),
                    #    np.real(self.e0-ef),
                    #    np.real(self.e0-m_qk(coeff))))
                    #print('Lamb: {:.12f}, Coeff: {:.12f}'.format(
                    #    np.real(lamb),
                    #    np.real(coeff)))
                    self.Store.update(df)
                    trust_iter+=1
                    if trust_iter>=2:
                        trust=True
            for f in testS:
                #f.qCo*= coeff
                f.c*= coeff
            self.S = self.S+testS
        else:
            for f in testS:
                #f.qCo*= -(d1D/d2D)
                f.c*= -(d1D/d2D)
            self.S = self.S+testS
        if not self.use_trust_region:
            self.S = self.S+testS
            # eval energy is in check step
            Ins = self.Instruct(
                    operator=self.S,
                    Nq=self.QuantStore.Nq,
                    quantstore=self.QuantStore,
                    depth=self.S_trotter)
            Psi= StandardTomography(
                    QuantStore=self.QuantStore,
                    preset=self.tomo_preset,
                    Tomo=self.tomo_Psi,
                    verbose=self.verbose,
                    )
            if not self.tomo_preset:
                Psi.generate(real=True,imag=False)
            Psi.set(tIns)
            Psi.simulate()
            Psi.construct(processor=self.process)
            self.Store.update(Psi.rdm)
            Psi.rdm.switch()
            self.circ = Psi
        print('Current S: ')
        print(self.S)

    def next_step(self):
        if self.built:
            self._run_acse()
            self._check()
            print('E,init: {:+.12f} U'.format(np.real(self.ei)))
            print('E, run: {:+.12f} U'.format(np.real(self.best)))
            try:
                diff = 1000*(self.best-self.Store.H.ef)
                print('E, fin: {:+.12f} U'.format(self.Store.H.ef))
                print('E, dif: {:.12f} mU'.format(diff))
            except KeyError:
                pass
            except AttributeError:
                pass
            print('-------------------------')

    def run(self):
        '''
        Note, run for any ACSE has the generic steps:
            - find the S matrix,
            - build the S ansatz
            - evaluate the ansatz, or D, evaluate energy
        '''
        if self.built:
            while not self.total.done:
                self._run_acse()
                self._check()
            print('E,init: {:+.12f} U'.format(np.real(self.ei)))
            print('E, run: {:+.12f} U'.format(np.real(self.best)))
            try:
                diff = 1000*(self.best-self.Store.H.ef)
                print('E, fin: {:+.12f} U'.format(self.Store.H.ef))
                print('Energy difference from goal: {:.12f} mU'.format(diff))
            except KeyError:
                pass
            except AttributeError:
                pass
        self.save

    def _check(self,full=True):
        '''
        Internal check on the energy as well as norm of the S matrix
        '''
        en = self.Store.evaluate(self.Store.rdm)
        if self.total.iter==0:
            self.best = copy(self.e0)
        self.total.iter+=1
        if self.total.iter==self.max_iter:
            print('Max number of iterations met. Ending optimization.')
            self.total.done=True
        elif self.S.d==self.max_depth:
            if copy(self.S)+copy(self.A)>self.max_depth:
                self.total.done=True
        self.log_E.append(en)
        self.log_depth.append(self.S.d)
        self.log_norm.append(self.norm)
        self.log_G.append(self.grad)

        try:
            self.log_ci.append(self.ci)
        except:
            pass
        i = 1
        temp_std_En = []
        temp_std_S = []
        temp_std_G = []
        while i<= min(3,self.total.iter):
            temp_std_En.append(self.log_E[-i])
            temp_std_S.append(self.log_norm[-i])
            temp_std_G.append(self.log_G[-i])
            i+=1
        avg_En = np.real(np.average(np.asarray(temp_std_En)))
        avg_S =  np.real(np.average(np.asarray(temp_std_S)))
        std_En = np.real(np.std(np.asarray(temp_std_En)))
        std_S  = np.real(np.std(np.asarray(temp_std_S)))
        std_G =  np.abs(np.real(np.average(np.asarray(temp_std_G))))
        self.Store.analysis()
        print('')
        print('---------------------------------------------')
        print('Step {:02}, Energy: {:.12f}, S: {:.12f}'.format(
            self.total.iter,
            np.real(en),
            np.real(self.norm)))
        print('Standard deviation in energy: {:+.12f}'.format(std_En))
        print('Average energy: {:+.12f}'.format(avg_En))
        print('Standard deviation in S: {:.12f}'.format(std_S))
        print('Average S: {:.12f}'.format(avg_S))
        if self.QuantStore.backend=='statevector_simulator':
            if en<self.best:
                self.best=np.real(en)
            if self._conv_type=='default':
                if avg_En>self.best_avg and self.total.iter>=20:
                    print('Average energy is increasing!')
                    print('Ending optimization.')
                    self.total.done=True
            else:
                self.best_avg = copy(avg_En)
        else:
            if en<self.best:
                self.best = np.real(en)
        #
        if self._conv_type=='default':
            if std_En<self.crit and self.norm<0.05:
                print('Criteria met. Ending optimization.')
                self.total.done=True
        elif self._conv_type in ['gradient','gradient_norm']:
            if self._conv_type=='gradient_norm':
                print('Normed gradient: {:.14f}'.format(
                    np.real(self.grad/self.norm)))
            print('Gradient size: {:.14f}'.format(np.real(self.grad)))
            if abs(self.grad)<self.crit:
                self.total.done=True
                print('Criteria met in gradient. Ending optimization.')
            if self.acse_method=='newton' and self.use_trust_region:
                if self.tr_Del<self.crit:
                    self.total.done=True
                    print('Trust region met criteria!')
                    print('Ending optimization')
            if avg_En>self.best_avg:
                print('Optimization status 2')
                print('Average energy is increasing!')
                print('Ending optimization.')
                self.total.done=True
        elif self._conv_type in ['trust']:
            if abs(self.tr_taylor)<=self.tr_ts_crit:
                self.total.done=True
                print('Optimization status 0')
                print('Criteria met in taylor series model.')
                print('Ending optimization.')
            elif abs(self.tr_object)<= self.tr_obj_crit:
                self.total.done=True
                print('Criteria met in objective function.')
                print('Ending optimization.')
        elif self._conv_type=='iterations':
            pass
        elif self._conv_type in ['S-norm','norm']:
            if self.norm<self.crit:
                self.total.done=True
        elif self._conv_type in ['custom']:
            # hm...rawr 
            # first, want to check how long we have had a constant depth d
            # 
            curr_depth = self.S.d
            constant = 0
            done = False
            while not done:
                if constant<len(self.log_depth):
                    done=True
                    continue
                if self.log_depth[::-1][constant]==curr_depth:
                    done=True
            if constant>self.max_constant_depth:
                # now, check if next step is same depth
                new = copy(self.S)+copy(self.A)
                if new.d==self.S.d:
                    print('Optimization status 1')
                    print('Ansatz failed to increase depth.')
                    self.total.done=True
            if avg_En>self.best_avg:
                print('Optimization status 2')
                print('Average energy is increasing!')
                print('Ending optimization.')
            if self.norm<self.crit:
                self.total.done=True
        else:
            print('Convergence type not specified.')
            sys.exit('Goodbye.')
        self.e0 = copy(en)
        print('---------------------------------------------')

    def save(self,
            name,
            ):
        try:
            self.log_A
        except AttributeError:
            sys.exit('Forgot to turn logging on!')
        data = {
                'log-A':self.log_A,
                'log-D':self.log_rdm,
                'log-S':self.log_S,
                'H':self.Store.H.matrix,
                'config':{

                    },
                }
        try:
            data['log-Gamma']=self.log_Gamma
        except AttributeError as e:
            pass
        with open(name+'.log','wb') as fp:
            pickle.dump(data,fp,pickle.HIGHEST_PROTOCOL)

