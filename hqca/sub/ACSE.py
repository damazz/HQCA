import pickle
from copy import deepcopy as copy
import threading
import os, sys
from importlib import reload
import numpy as np
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from hqca.tools import Functions as fx
from hqca.tools import RDMFunctions as rdmf
from hqca.sub.BaseRun import QuantumRun,Cache
from hqca.acse import ClassOpS as classS
from hqca.acse import QuantOpS as quantS
from hqca.acse import FunctionsACSE as acse
from hqca.acse.BuildAnsatz import Ansatz
from hqca.quantum import ErrorCorrection as ec
from hqca.optimizers.Control import Optimizer
from hqca.quantum import QuantumFunctions as qf
from hqca.quantum import NoiseSimulator as ns
from hqca.quantum import Tomography as tomo
from hqca.tools.util import Errors
from functools import reduce,partial 
import datetime
import sys
from hqca.tools import Preset as pre
from scipy import stats
np.set_printoptions(precision=3,suppress=True,linewidth=200)

class RunACSE(QuantumRun):
    '''
    Object for running different forms of the ACSE method on a classical
    computer. Note, we have the following forms:
        - cc-acse:
            classical/classical acse, Euler equation - not implemented here. 
        - qc-acse:
            aka, qc-acse-euler
            quantum-classical acse, where the acse equations are solved
            classically, and the ansatz and change in the 2-RDM is performed on
            the quantum computer. requires 3-RDM reconstruction
        - qc-acse2:
            aka, qc-acse-newton
            also a quantum-classical acse approach, where the acse equations are
            solved according to Newton's method, where the optimial parameter
            delta is calcualted at each step. requires 3-RDM reconstruction.
        - qq-acse:
            similar to euler approach, but both the calculation of S and D is
            done on the quantum computer. S is found as the real remainder of
            the time evolution operator.
        - qq-acse2:
            same as above, but with a different optimization approach, and the
            quantum S calculation.
        - a-acse:
            adiabatic ACSE, where the solution is allowed to evolve and responed
            to the S operator; utilizes the Newton-Raphson method, experimental
    '''
    def __init__(self,
            mol,
            theory='qACSE',
            **kw
            ):
        self.theory=theory
        QuantumRun.__init__(self,**kw)
        kw['mol']=mol
        self.Store = acse.ModStorageACSE(**kw)
        if mol==None:
            print('Did you forget to specify a mol object form pyscf?')
            print('Please try again.')
            sys.exit()
        else:
            pass
        self.kw = pre.qACSE()
        self.pr_g = self.kw['pr_g']
        self.kw_qc = self.kw['qc']
        self.kw_opt = self.kw['acse']['opt']
        self.kw_acse = self.kw['acse']
        self.total=Cache()
        self.best = 0
        self.best_avg = 0

    def build(self):
        '''
        Build the quantum object, QuantStore
        '''
        QuantumRun._build_quantum(self)
        self._update_acse_kw(**self.kw['acse'])
        self.built=True
        self.log_S = []
        self.log_E = []
        reTomo = tomo.Tomography(
                self.QuantStore)
        reTomo.generate_2rdme(real=True,imag=False)
        self.QuantStore.reTomo_kw = {
                'mapping':reTomo.mapping,
                'preset_grouping':True,
                'rdm_elements':reTomo.rdme,
                'tomography_terms':reTomo.op
                }
        if 'q' in self.acse_update:
            self.Store._get_HamiltonianOperators(full=True)
            imTomo = tomo.Tomography(self.QuantStore)
            imTomo.generate_2rdme(real=False,imag=True)
            self.QuantStore.imTomo_kw = {
                    'mapping':imTomo.mapping,
                    'preset_grouping':True,
                    'rdm_elements':imTomo.rdme,
                    'tomography_terms':imTomo.op
                    }
        print('Done initializing. Beginning run...')
        print('---------------------------------------------')

    def update_var(self,target='acse',**kw):
        kw['target']=target
        if target=='acse':
            try:
                self.kw['acse']
            except KeyError:
                self.kw['acse']={}
            for k,v in kw.items():
                self.kw['acse'][k]=v
        else:
            QuantumRun.update_var(self,**kw)
        self.Store.pr_m = self.kw['pr_m']

    def _update_acse_kw(self,
            method='newton',
            update='quantum',
            opt_thresh=1e-3,
            max_iter=100,
            trotter=1,
            pr_a=1,
            ansatz_depth=1,
            use_damping=False,
            use_trust_region=False,
            damping_amplitude=np.pi/2,
            newton_damping=False,
            newton_step=2,
            quantS_thresh_max_rel=0.1,
            quantS_max=1e-10,
            classS_thresh_max_rel=0.1,
            convergence_type='default',
            hamiltonian_step_size=1.0,
            restrict_S_size=0.5,
            initial_trust_region=np.pi/2,
            **kw):
        if update in ['quantum','Q','q']:
            self.acse_update = 'q'
        elif update in ['class','classical','c','C']:
            self.acse_update ='c'
        if not method in ['NR','EM','opt','trust','newton']:
            print('Specified method not valid. Update acse_kw: \'method\'')
            sys.exit()
        self.acse_method = method
        self.ansatz_depth=1
        self.tr_Del  = initial_trust_region # trust region
        self.d = newton_step #for estimating derivative
        self.delta = restrict_S_size
        self.use_damping = use_damping
        self.use_trust_region = use_trust_region
        self.damp_sigma = damping_amplitude
        self.QuantStore.depth_S = ansatz_depth
        self.N_trotter = trotter
        self.max_iter = max_iter
        self.crit = opt_thresh
        self.hamiltonian_step_size = hamiltonian_step_size
        self.qS_thresh_max_rel = quantS_thresh_max_rel
        self.qS_max = quantS_max
        self.cS_thresh_max_rel = classS_thresh_max_rel
        self._conv_type = convergence_type
        self.newton_damping = newton_damping
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('      --  ACSE KEYWORDS --      ')
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('ACSE Method: {}'.format(method))
        print('ACSE Update: {}'.format(update))
        print('Max iterations: {}'.format(max_iter))
        print('Convergence type: {}'.format(convergence_type))
        print('Convergence threshold: {}'.format(self.crit))
        print('Hamiltonian epsilon: {}'.format(hamiltonian_step_size))
        print('Trotter-H: {}'.format(trotter))
        print('Trotter-S: {}'.format(ansatz_depth))
        print('Quant-S max threshold: {}'.format(quantS_max))
        print('Quant-S rel threshold: {}'.format(quantS_thresh_max_rel))
        print('Class-S rel threshold: {}'.format(classS_thresh_max_rel))
        print('Newton step: {}'.format(newton_step))
        print('Newton trust region: {}'.format(use_trust_region))
        print('Trust region: {}'.format(initial_trust_region))
        print('Newton damping: {}'.format(use_damping))
        print('S damping: {}'.format(restrict_S_size))
        print('Damping amplitude: {}'.format(damping_amplitude))
        print('-- -- -- -- -- -- -- -- -- -- --')

    def _run_acse(self):
        if self.acse_update=='q':
            testS = quantS.findSPairsQuantum(
                    self.Store,
                    self.QuantStore,
                    qS_thresh_max_rel=self.qS_thresh_max_rel,
                    qS_max=self.qS_max,
                    trotter_steps=self.N_trotter,
                    hamiltonian_step_size=self.hamiltonian_step_size,
                    verbose=True)
        elif self.acse_update=='c':
            testS = classS.findSPairs(self.Store)
        self._check_norm(testS)
        if self.acse_method in ['NR','newton']:
            self.__newton_acse(testS)
        elif self.acse_method in ['default','em','EM','euler']:
            self.__euler_acse(testS)
        elif self.acse_method in ['trust','TR']:
            pass
        elif self.acse_method in ['opt']:
            pass

    def _check_norm(self,testS):
        '''
        evaluate norm of S calculation
        '''
        self.norm = 0
        for item in testS:
            self.norm+= item.norm
        self.norm = self.norm**(0.5)

    def __euler_acse(self,testS):
        '''
        function of Store.build_trial_ansatz
        '''
        # where is step size from? 
        # test procedure
        for s in testS:
            s.qCo*=self.delta
            s.c*=self.delta
        self.Store.update_ansatz(testS)
        Psi = Ansatz(self.Store,self.QuantStore,**self.QuantStore.reTomo_kw)
        Psi.build_tomography()
        Psi.run_circuit()
        Psi.construct_rdm()
        self.Store.rdm2=Psi.rdm2

    def __opt_acse(self,testS):
        max_S_val = 0
        if self.total.iter==0:
            self.kw_opt['initial_left_bound']=copy(self.Store.hf.e_tot)
        for s in testS:
            if abs(s.c)>max_S_val:
                max_S_val = copy(s.c)
        para = [self.delta]
        self.kw_opt['unity']=np.pi
        f = partial(self.__test_acse_function,testS=testS)
        self.Run = Optimizer(
                function=f,
                **self.kw_opt)
        self.Run.initialize(para)
        sub = Cache()
        while not sub.done:
            self.Run.next_step()
            self.Run.check(sub)
            sub.iter+=1
        self.Store.rdm2
        for s in testS:
            s.c*=self.Run.opt.best_x*self.delta
            s.qCo*=self.Run.opt.best_x*self.delta
        self.Store.update_ansatz(testS)
        self.kw_opt['initial_left_bound']=copy(self.Run.opt.best_f)

    def __test_acse_function(self,parameter,testS=None):
        testAnsatz = copy(testS)
        for f in testAnsatz:
            f.c*= parameter[0]
            f.qCo*= parameter[0]
        self.Store.build_trial_ansatz(testAnsatz)
        tempPsi = Ansatz(self.Store,self.QuantStore,trialAnsatz=True,
                **self.QuantStore.reTomo_kw)
        tempPsi.build_tomography()
        tempPsi.run_circuit()
        tempPsi.construct_rdm()
        en = np.real(self.Store.evaluate_temp_energy(tempPsi.rdm2))
        return en,tempPsi.rdm2
    
    def __newton_acse(self,testS):
        max_val = 0
        for s in testS:
            if abs(s.c)>abs(max_val):
                max_val = copy(s.c)
        print('Maximum value: {:+.10f}'.format(np.real(max_val)))
        e1,tdm1 = self.__test_acse_function([self.delta],testS)
        e2,rdm2 = self.__test_acse_function([self.d*self.delta],testS)
        try:
            self.e0
        except AttributeError:
            self.e0 = self.Store.evaluate_energy()
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
            np.real(d1D/d2D),
            np.real(max_val*d1D/d2D))
            )
        def damping(x):
            if self.damp_sigma==0:
                return 1
            else:
                return np.exp(-(x**2)/((self.damp_sigma)**2))
        self.grad = d1D
        self.hess = d2D
        if self.use_trust_region:
            print('Trust region step.')
            if self.hess<0:
                print('Hessian non-positive. Taking Euler step.')
                if g2<g1:
                    coeff = self.delta*self.d
                elif g1<0:
                    coeff = self.delta
                else:
                    self.delta*=0.5
                    coeff = self.delta
            else:
                trust = False
                nv = 0.9
                ns = 0.1
                gi = 1.5
                gd = 0.5
                while not trust: # perform sub routine
                    if abs(d1D/d2D)<self.tr_Del:
                        # found ok answer! 
                        coeff = -d1D/d2D
                    else:
                        lamb = -d1D/self.tr_Del-d2D
                        coeff = -d1D/(d2D+lamb)
                    ef,df = self.__test_acse_function([coeff],testS)
                    def m_qk(s):
                        return self.e0 + s*self.grad+0.5*s*self.hess*s
                    rho = (self.e0 - ef)/(self.e0-m_qk(coeff))
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
                        trust = False
                    print('Current trust region: {:.6f}'.format(
                        np.real(self.tr_Del)))
                    print('Rho: {:.6f},Num: {:.6f}, Den: {:.6f}'.format(
                        np.real(rho),
                        np.real(self.e0-ef),
                        np.real(self.e0-m_qk(coeff))))
                    self.Store.rdm2=df
            for f in testS:
                f.qCo*= coeff
                f.c*= coeff
            self.Store.update_ansatz(testS)
        elif self.use_damping:
            damp = damping(max_val*(d1D/d2D))
            print('Step: {:.6f}, Largest: {:.6f}, Damping Factor: {:.6f}'.format(
                np.real(d1D/d2D),
                np.real(max_val*d1D/d2D),
                np.real(damp)))
            if d2D>0:
                if abs(damp)<(self.delta*self.d): #damping factor kills the run
                    print('Damping factor too large - taking Euler step.')
                    for f in testS:
                        f.qCo*= self.delta*self.d
                        f.c*= self.delta*self.d
                else:
                    print('Applying damped step.')
                    for f in testS:
                        f.qCo*= -(d1D/d2D)*damp
                        f.c*= -(d1D/d2D)*damp
            else:
                print('Hessian non-positive. Taking Euler step.')
                if g2<g1:
                    c = self.delta*self.d
                elif g1<0:
                    c = self.delta
                else:
                    self.delta*=0.5
                    c = self.delta
                for f in testS:
                    f.qCo*= c
                    f.c*= c
        else:
            for f in testS:
                f.qCo*= -(d1D/d2D)
                f.c*= -(d1D/d2D)
        if not self.use_trust_region:
            self.Store.update_ansatz(testS)
            Psi = Ansatz(self.Store,self.QuantStore,
                    **self.QuantStore.reTomo_kw)
            Psi.build_tomography()
            Psi.run_circuit()
            Psi.construct_rdm(variance=True)
            self.Store.rdm2=Psi.rdm2
            # eval energy is in check step
            if abs(Psi.rdm2.trace()-2)>1e-3:
                print('Trace of 2-RDM: {}'.format(Psi.rdm2.trace()))
            if self.total.iter%3==0:
                self._calc_variance(Psi.rdm2_var,Psi)






    def _calc_variance(self,vrdm2,psi,ci=0.90):
        if self.QuantStore.backend=='unitary_simulator':
            self.ci=1e-8
        elif self.QuantStore.backend=='statevector_simulator':
            self.ci=1e-8
        else:
            en = self.Store.evaluate_temp_energy(vrdm2)-self.Store.E_ne
            alp = 1-(1-ci)/2
            z = stats.norm.ppf(alp)
            nci = z*np.sqrt(en)/np.sqrt(self.QuantStore.Ns)
            self.ci2 = nci
            self.ci = psi.evaluate_error(
                    f=self.Store.evaluate_temp_energy)
            print('Variance 1: {:.6f} (CLT)'.format(np.real(self.ci)))
            print('Variance 2: {:.6f} (Bernoulli)'.format(np.real(self.ci2)))
            print('')

    def _run_adiabatic_acse(self):
        self.delta = 0.25
        # run adiabatic acse for a single step
        self.sub = Cache()
        self.Store.t+=self.Store.dt
        if self.method in ['ac-acse','ac-acse2']:
            while not self.sub.done:
                testS = classS.findS0Pairs(self.Store)
                if self.method=='ac-acse':
                    self.__euler_qc_acse(testS)
                elif self.method=='ac-acse2':
                    self.__newton_qc_acse(testS)
                self.__sub_check()

    def __sub_check(self):
        en = self.Store.evaluate_energy()
        self.sub.iter+=1
        print('Micro Step {:02}, Energy: {}'.format(self.sub.iter,en))
        try:
            if self.old<en-0.005:
                self.sub.done=True
        except Exception:
            self.old = en
        if en<self.old:
            self.old = en
        self.e0 = en

    def execute(self):
        self.run()

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
            print('E, scf: {:.12f} H'.format(self.Store.hf.e_tot))
            print('E, run: {:.12f} H'.format(self.best))
            try:
                diff = 1000*(self.best-self.Store.e_casci)
                print('E, fci: {:.12f} H'.format(self.Store.e_casci))
                print('Energy difference from FCI: {:.12f} mH'.format(diff))
            except KeyError:
                pass

    def _check(self,full=True):
        '''
        Internal check on the energy as well as norm of the S matrix
        '''
        # need to find energy
        if 'opt' in self.acse_method:
            en = copy(self.Run.opt.best_f)
        else:
            en = self.Store.evaluate_energy()

        if self.total.iter==0:
            self.old = self.Store.hf.e_tot
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
        std_G =  np.real(np.average(np.asarray(temp_std_G)))
        self.Store.acse_analysis()
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
                if avg_En>self.best_avg:
                    print('Average energy is increasing!')
                    print('Ending optimization.')
                    self.total.done=True
            else:
                self.best_avg = copy(avg_En)
        else:
            self.best = avg_En
        # implementing dynamic stopping criteria 
        if 'q' in self.acse_update or 'c' in self.acse_update:
            if self._conv_type=='default':
                if std_En<self.crit and self.norm<0.05:
                    print('Criteria met. Ending optimization.')
                    self.total.done=True
            elif self._conv_type=='gradient':

                print('Gradient size: {:.14f}'.format(np.real(self.grad)))
                if abs(self.grad)<self.crit:
                    self.total.done=True
                    print('Criteria met in gradient. Ending optimization.')
                if std_G<self.crit*0.1:
                    print('Alternative criteria met.')
                    print('Gradient variation smaller than gradient measurement')
                    self.total.done=True

        self.e0 = en
        print('---------------------------------------------')

    def save(self,
            name
            ):
        np.savetxt(name,np.asarray(
            [self.log_E,self.log_S]))


