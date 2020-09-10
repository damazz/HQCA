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
            quantS_thresh_rel=0.1,
            quantS_ordering='default',
            quantS_max=1e-10,
            classS_thresh_rel=0.1,
            classS_max=1e-10,
            convergence_type='default',
            hamiltonian_step_size=0.1,
            restrict_S_size=0.5,
            propagation='trotter',
            separate_hamiltonian=False,
            verbose=True,
            tomo_S=None,
            tomo_Psi=None,
            one_body_kw={},
            statistics=False,
            processor=None,
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
        self.one_body_kw = one_body_kw
        self.S_trotter= ansatz_depth
        self.S_commutative = commutative_ansatz
        self.N_trotter = trotter
        self.max_iter = max_iter
        self.crit = opt_thresh
        self.hamiltonian_step_size = hamiltonian_step_size
        self.sep_hamiltonian = separate_hamiltonian
        self.qS_thresh_rel = quantS_thresh_rel
        self.qS_max = quantS_max
        self.delta = restrict_S_size
        self.propagate_method=propagation
        self.qS_ordering = quantS_ordering
        self.cS_thresh_rel = classS_thresh_rel
        self.cS_max = classS_max
        self._conv_type = convergence_type
        self.tomo_S=tomo_S
        self.tomo_Psi=tomo_Psi
        if type(self.tomo_Psi)==type(None):
            self.tomo_preset=False
        else:
            self.tomo_preset=True
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('      --  ACSE KEYWORDS --      ')
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('algorithm...')
        print('ACSE Method: {}'.format(method))
        print('ACSE Update: {}'.format(update))
        print('Max iterations: {}'.format(max_iter))
        print('Convergence type: {}'.format(convergence_type))
        print('Convergence threshold: {}'.format(self.crit))

        print('solution of ACSE...')
        if self.acse_update=='q':
            print('Trotter-H: {}'.format(trotter))
            print('Hamiltonian delta: {}'.format(hamiltonian_step_size))
            print('Quant-S rel threshold: {}'.format(quantS_thresh_rel))
            print('Quant-S max threshold: {}'.format(quantS_max))
        elif self.acse_update=='c':
            print('Class-S rel threshold: {}'.format(classS_thresh_rel))
            print('Class-S max threshold: {}'.format(classS_max))
        print('implementing the ansatz...')
        print('S epsilon: {}'.format(self.delta))
        print('Trotter-S: {}'.format(ansatz_depth))
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
        print('Optimizer threshold: {}'.format(optimizer_threshold))
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
        print('Newton step: {}'.format(newton_step))
        print('Newton trust region: {}'.format(use_trust_region))
        print('Trust region: {:.6f}'.format(initial_trust_region))
        print('-- -- -- -- -- -- -- -- -- -- --')
        self.tr_taylor = 1
        self.tr_object = 1


    def build(self,log=False):
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
            #self._calc_variance(circ)
        else:
            self.S = Operator()
            self.e0 = self.Store.e0
            self.ei = self.Store.ei
        self.best = self.e0
        self.best_avg = self.e0
        self.log_S = []
        self.log_E = [self.e0]
        self.log_G = []
        try:
            self.log_ci = [self.ci]
        except Exception: 
            pass
        #self.log=log_r
        #if self.lrdm:
        #    self.log_rdm = [self.Store.rdm]
        self.total=Cache()
        self.built=True



    def _run_acse(self):
        try:
            self.built
        except AttributeError:
            sys.exit('Not built! Run acse.build()')
        if self.acse_update=='q':
            testS = findSPairsQuantum(
                    self.QuantStore.op_type,
                    operator=self.S,
                    process=self.process,
                    instruct=self.Instruct,
                    store=self.Store,
                    quantstore=self.QuantStore,
                    qS_thresh_rel=self.qS_thresh_rel,
                    qS_max=self.qS_max,
                    ordering=self.qS_ordering,
                    trotter_steps=self.N_trotter,
                    hamiltonian_step_size=self.hamiltonian_step_size,
                    propagate_method=self.propagate_method,
                    depth=self.S_trotter,
                    commutative=self.S_commutative,
                    separate_hamiltonian=self.sep_hamiltonian,
                    verbose=self.verbose,
                    tomo=self.tomo_S,
                    )
        elif self.acse_update=='c':
            testS = findSPairs(
                    self.Store,
                    self.QuantStore,
                    classS_thresh_rel=self.cS_thresh_rel,
                    classS_max=self.cS_max,
                    commutative=self.S_commutative,
                    )
        self._check_norm(testS)
        # check if ansatz will change length
        self._check_length(testS)
        if self.acse_method in ['NR','newton']:
            self.__newton_acse(testS)
        elif self.acse_method in ['default','em','EM','euler']:
            self.__euler_acse(testS)
        elif self.acse_method in ['line']:
            self.__opt_line_acse(testS)

    def _check_length(self,newS):
        testS = copy(newS)
        currS = copy(self.S)
        for f in testS:
            f.c*= 0.0001
        I0=self.Instruct(
                operator=currS,
                Nq=self.QuantStore.Nq,
                quantstore=self.QuantStore,
                depth=self.S_trotter,
                )
        I1=self.Instruct(
                operator=currS+testS,
                Nq=self.QuantStore.Nq,
                quantstore=self.QuantStore,
                depth=self.S_trotter,
                )
        if len(I1._gates)>len(I0._gates):
            if self.verbose:
                print('Increasing gate length.')
            if self.QuantStore.post:
                if self.QuantStore.method=='shift':
                    #self.QuantStore.measure_shift=True
                    # then, we need to update
                    Circ= StandardTomography(
                            QuantStore=self.QuantStore,
                            preset=self.tomo_preset,
                            Tomo=self.tomo_Psi,
                            verbose=self.verbose,
                            )
                    if not self.tomo_preset:
                        Circ.generate(
                                real=self.Store.H.real,
                                imag=self.Store.H.imag)
                    Circ.set(I1)
                    Circ.simulate()

                    Circ.construct(processor=self.process)
                    if type(self.QuantStore.Gamma)==type(None):
                        self.QuantStore.Gamma = self.Store.rdm-Circ.rdm
                    else:
                        self.QuantStore.Gamma+= self.Store.rdm-Circ.rdm
                    print('---------------------------------')
                    print('New Gamma:')
                    print('---------------------------------')
                    self.QuantStore.Gamma.analysis()
                    print('---------------------------------')
                    print('---------------------------------')
                    #self.QuantStore.measure_shift=False

    def _check_norm(self,testS):
        '''
        evaluate norm of S calculation
        '''
        self.norm = 0
        for item in testS.op:
            self.norm+= item.norm
        self.norm = self.norm**(0.5)


    def __opt_line_acse(self,testS):
        '''
        '''
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
        for r,e in zip(self._opt_log,self._opt_en):
            if abs(e-opt.opt.best_f)<=1e-5:
                self.Store.update(r.rdm)
        for s in testS:
            s.c*=opt.opt.best_x[0]
        self.S = self.S + testS
        print(self.S)

    def __euler_acse(self,testS):
        '''
        function of Store.build_trial_ansatz
        '''
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

    def __newton_acse(self,testS):
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

    def _check(self,full=True):
        '''
        Internal check on the energy as well as norm of the S matrix
        '''
        # need to find energy
        #print('S Operator: ')
        #print(self.S)
        en = self.Store.evaluate(self.Store.rdm)
        #if self.lrdm:
        #    self.log_rdm.append(self.Store.rdm)
        if self.total.iter==0:
            self.old = copy(self.e0)
        self.total.iter+=1
        if self.total.iter==self.max_iter:
            print('Max number of iterations met. Ending optimization.')
            self.total.done=True
        try:
            self.old
        except AttributeError:
            self.old = en
        if en<=self.old:
            self.old = en
        self.log_E.append(en)
        self.log_S.append(self.norm)
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
            temp_std_S.append(self.log_S[-i])
            temp_std_G.append(self.log_G[-i])
            i+=1
        avg_En = np.real(np.average(np.asarray(temp_std_En)))
        avg_S =  np.real(np.average(np.asarray(temp_std_S)))
        std_En = np.real(np.std(np.asarray(temp_std_En)))
        std_S  = np.real(np.std(np.asarray(temp_std_S)))
        std_G =  np.abs(np.real(np.average(np.asarray(temp_std_G))))
        self.Store.analysis()
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
            #else:
            #    self.best = avg_En
        # implementing dynamic stopping criteria 
        if 'q' in self.acse_update or 'c' in self.acse_update:
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
                    print('Average energy is increasing!')
                    print('Ending optimization.')
                    self.total.done=True
            elif self._conv_type in ['trust']:
                if abs(self.tr_taylor)<=self.tr_ts_crit:
                    self.total.done=True
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
            else:
                print('Convergence type not specified.')
                sys.exit('Goodbye.')
        self.e0 = copy(en)
        print('---------------------------------------------')

    def save(self,
            name
            ):
        np.savetxt(name,np.asarray(
            [self.log_E,self.log_S]))


