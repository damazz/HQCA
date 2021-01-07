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
from hqca.acse._newton_acse import _newton_step
from hqca.acse._euler_acse import _euler_step
from hqca.acse._opt_acse import _opt_step
from hqca.acse._mitigation import *
from hqca.acse._check_acse import check_routine
from hqca.tools import *

class RunACSE(QuantumRun):
    '''
    Quantum ACSE method.
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
            pr_a=1,
            ansatz_depth=1,
            commutative_ansatz=False,
            S_ordering='default',
            S_thresh_rel=0.1,
            S_min=1e-10,
            S_num_terms=None,
            convergence_type='default',
            hamiltonian_step_size=0.1,
            restrict_S_size=0.5,
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
        self.S_commutative = commutative_ansatz
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_constant_depth = max_constant_depth
        self.crit = opt_thresh
        self.hamiltonian_step_size = hamiltonian_step_size
        self.sep_hamiltonian = separate_hamiltonian
        self.S_thresh_rel = S_thresh_rel
        self.S_min = S_min
        self.S_num_terms = S_num_terms
        self.delta = restrict_S_size
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
            print('hamiltonian delta: {}'.format(hamiltonian_step_size))
        print('S rel threshold: {}'.format(S_thresh_rel))
        print('S max threshold: {}'.format(S_min))
        print('-- -- -- --')
        print('ansatz')
        print('-- -- -- --')
        print('S epsilon: {}'.format(self.delta))
        print('-- -- -- --')
        print('optimization')
        print('-- -- -- --')

        self._optimizer = None
        self._opt_thresh = None
        if self.acse_method=='newton':
            self._update_acse_newton(**kw)
        elif self.acse_method in ['line','opt']:
            self._update_acse_opt(**kw)
        self._update_experimental(**kw)
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

    def _update_experimental(self,
            split_ansatz=False,
            split_threshold=1.0,
            **kw
            ):
        self.split_ansatz=split_ansatz
        self.split_threshold = split_threshold

    def _update_acse_newton(self,
            use_trust_region=False,
            newton_step=2,
            initial_trust_region=np.pi/2,
            tr_taylor_criteria=1e-10,
            tr_objective_criteria=1e-10,
            tr_gamma_inc=2,
            tr_gamma_dec=0.5,
            tr_nu_accept=0.9,
            tr_nu_reject=0.1,
            **kw
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

    def _generate_real_circuit(self,op):
        if isinstance(op,type(Ansatz())):
            op = op.op_form()
        else:
            sys.exit('Huh?')
        ins = self.Instruct(
                operator=op,
                Nq=self.QuantStore.Nq,
                quantstore=self.QuantStore)
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
        return circ

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
                s.c*=self.delta
            circ = self._generate_real_circuit(self.S)
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
        self.log_E_best=  [self.e0]
        self.log_G = []
        try:
            self.log_ci = [self.ci]
        except Exception: 
            pass
        self.total=Cache()
        self.accept_previous_step=True
        self.__get_S()
        if self.log:
            self.log_A.append(copy(self.A))
        print('||A||: {:.10f}'.format(np.real(self.norm)))
        print('-- -- -- -- -- -- -- -- -- -- --')
        #
        # run checks
        #
        check_routine(self)
        self.built=True

    def __get_S(self):
        if not self.accept_previous_step:
            print('Rejecting previous step. No recalculation of A.')
            return
            # 
        if self.acse_update=='q':
            A_sq = findSPairsQuantum(
                    self.QuantStore.op_type,
                    operator=self.S.op_form(),
                    process=self.process,
                    instruct=self.Instruct,
                    store=self.Store,
                    quantstore=self.QuantStore,
                    S_min=self.S_min,
                    ordering=self.S_ordering,
                    hamiltonian_step_size=self.hamiltonian_step_size,
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
        # check if operator is split #
        inc = Operator()
        exc = Operator()
        #   #
        if self.split_ansatz:
            for n in new:
                added=False
                for m in reversed(range(self.S.get_lim(),0)):
                    ai = self.S[m]
                    for o in ai:
                        if n==o:
                            inc+= n
                            added = True
                if not added:
                    exc+= n
            print('--------------')
            print('Included:')
            print(inc)
            print('Excluded:')
            print(exc)
            print('--------------')
            if inc.norm()==0 or exc.norm()==0:
                pass
            elif exc.norm()/inc.norm()>self.split_threshold:
                self.A = copy(exc)
            else:
                self.A = copy(inc)
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
        check_mitigation(self)
        try:
            self.built
        except AttributeError:
            sys.exit('Not built! Run acse.build()')
        if self.acse_method in ['NR','newton']:
            _newton_step(self)
        elif self.acse_method in ['default','em','EM','euler']:
            _euler_step(self)
        elif self.acse_method in ['line','opt']:
            _opt_step(self)
        self.__get_S()
        #self._check_norm(self.A)
        # check if ansatz will change length
        if self.log:
            self.log_rdm.append(self.Store.rdm)
            self.log_A.append(copy(self.A))
            self.log_S.append(copy(self.S))


    def _check_norm(self,testS):
        '''
        evaluate norm of S calculation
        '''
        self.norm = 0
        for item in testS.op:
            self.norm+= item.norm
        self.norm = self.norm**(0.5)


    def _opt_acse_function(self,parameter,newS=None,verbose=False):
        testS = copy(newS)
        currS = copy(self.S)
        for f in testS:
            f.c*= parameter[0]
        temp = currS+testS
        tCirc = self._generate_real_circuit(temp)
        en = np.real(self.Store.evaluate(tCirc.rdm))
        self._opt_log.append(tCirc)
        self._opt_en.append(en)
        return en

    def _test_acse_function(self,parameter,newS=None,verbose=False):
        testS = copy(newS)
        currS = copy(self.S)
        for f in testS:
            f.c*= parameter[0]
        temp = currS+testS
        tCirc = self._generate_real_circuit(temp)
        en = np.real(self.Store.evaluate(tCirc.rdm))
        self.circ = tCirc
        return en,tCirc.rdm

    def _particle_number(self,rdm):
        return rdm.trace()

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
        en = self.Store.evaluate(self.Store.rdm)
        if self.total.iter==0:
            self.best = copy(self.e0)
        self.total.iter+=1
        if self.total.iter==self.max_iter:
            print('Max number of iterations met. Ending optimization.')
            self.total.done=True
        elif len(self.S)==self.max_depth:
            if copy(self.S)+copy(self.A)>self.max_depth:
                self.total.done=True
        self.log_E.append(en)
        self.log_depth.append(len(self.S))
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
        self.log_E_best.append(self.best)
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
            description=None
            ):
        try:
            self.log_A
        except AttributeError:
            sys.exit('Forgot to turn logging on!')
        data = {
                'log-A':self.log_A,
                'log-D':self.log_rdm,
                'log-S':self.log_S,
                'log-E':self.log_E,
                'log-Ee':self.log_E_best,
                'H':self.Store.H.matrix,
                'run_config':{
                    'method':self.acse_method,
                    'verbose':self.verbose,
                    'S_thresh_rel':self.S_thresh_rel,
                    'S_min':self.S_min,
                    'S_num_terms':self.S_num_terms,
                    'update':self.acse_update,
                    'opt_threshold':self.crit,
                    'max_depth':self.max_depth,
                    'H size':self.hamiltonian_step_size,
                    'separate hamiltonian':self.sep_hamiltonian,
                    'convergence type':self._conv_type,
                    'optimizer':self._optimizer,
                    'optimizer_threshold':self._opt_thresh,
                    },
                'quantum_storage':{
                    'backend':self.QuantStore.backend,
                    'provider':self.QuantStore.provider,
                    'number of qubits':self.QuantStore.Nq,
                    'number of shots':self.QuantStore.Ns,
                    'stabilizers':self.QuantStore.method,
                    },
                'description':description,
                }
        try:
            data['log-Gamma']=self.log_Gamma
        except AttributeError as e:
            pass
        with open(name+'.log','wb') as fp:
            pickle.dump(data,fp,pickle.HIGHEST_PROTOCOL)

