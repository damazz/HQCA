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

    def build(self):
        '''
        Build the quantum object, QuantStore
        '''
        QuantumRun._build_quantum(self)
        self._update_acse_kw(**self.kw['acse'])
        self.method = self.QuantStore.method # set method
        self.Store.method = self.method
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
        if 'qq' in self.method:
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
            opt_thresh=1e-3,
            max_iter=100,
            trotter=1,
            pr_a=1,
            ansatz_depth=1,
            damping=np.pi/2,
            newton_step=2,
            quantS_thresh_max_rel=0.1,
            classS_thresh_max_rel=0.1,
            **kw):
        self.ansatz_depth=1
        self.d = newton_step
        self.damp_sigma = damping
        self.QuantStore.depth_S = ansatz_depth
        self.N_trotter = trotter
        self.max_iter = max_iter
        self.crit = opt_thresh
        self.qS_thresh_max_rel = quantS_thresh_max_rel
        self.cS_thresh_max_rel = classS_thresh_max_rel
        if self.QuantStore.backend=='statevector_simulator':
            self.damp_sigma*=2


    def _run_qc_acse(self):
        '''
        Quantum Psi, Classical S
         1. find S classically,
         2. prepare ansatz for euler or newton
         3. run the ansatz, give best guess
        '''
        testS = classS.findSPairs(self.Store)
        self._check_norm(testS)
        if self.method=='qc-acse': #eulers methods
            self.delta=0.5
            self.__euler_qc_acse(testS) 
        elif self.method=='qc-acse2': #newtons methods
            self.delta = 0.25
            self.__newton_qc_acse(testS)
        elif self.method=='qc-acse-opt':
            self.delta=1.0
            self.__opt_acse(testS)

    def _run_qq_acse(self):
        '''
        # 1. find S quantumly,
        # 2. prepare ansatz for euler or newton
        # 3. run the ansatz, give best guess
        '''
        testS = quantS.findSPairsQuantum(self.Store,self.QuantStore,
                qS_thresh_max_rel=self.qS_thresh_max_rel,
                trotter_steps=self.N_trotter,
                verbose=True)
        self._check_norm(testS)
        if self.method=='qq-acse':
            self.delta = 0.5
            self.__euler_qc_acse(testS)
        elif self.method=='qq-acse2':
            self.delta = 0.25
            self.__newton_qc_acse(testS)
        elif self.method=='qq-acse-opt':
            self.delta=1.0
            self.__opt_acse(testS)

    def _check_norm(self,testS):
        '''
        evaluate norm of S calculation
        '''
        self.norm = 0
        for item in testS:
            self.norm+= item.norm
        self.norm = self.norm**(0.5)


    def __euler_qc_acse(self,testS):
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
        f = partial(self.__optimization_function,testS=testS)
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

    def __optimization_function(self,parameter,testS=None):
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
        en =  np.real(self.Store.evaluate_temp_energy(tempPsi.rdm2))
        return en

    def __newton_qc_acse(self,testS):
        # So, we make different Euler steps 
        # 1. Evaluate x
        max_val = 0
        for s in testS:
            if abs(s.c)>max_val:
                max_val = copy(s.c)
            s.qCo*=self.delta
            s.c*=self.delta
        print('Maximum value: {}'.format(max_val))
        self.Store.build_trial_ansatz(testS)
        Psi1e = Ansatz(self.Store,self.QuantStore,trialAnsatz=True,
                **self.QuantStore.reTomo_kw)
        Psi1e.build_tomography()
        Psi1e.run_circuit()
        Psi1e.construct_rdm()
        # 2. Evaluate 2x
        for s in testS:
            s.qCo*=self.d
            s.c*=self.d
        self.Store.build_trial_ansatz(testS)
        Psi2e = Ansatz(self.Store,self.QuantStore,trialAnsatz=True,
                **self.QuantStore.reTomo_kw)
        Psi2e.build_tomography()
        Psi2e.run_circuit()
        Psi2e.construct_rdm()
        # evaluate energies
        if self.d==1:
            sys.exit('b cannot be 1!')
        for s in testS:
            s.qCo*=(1/(self.delta*self.d))
            s.c*=(1/(self.delta*self.d))
        e1 = self.Store.evaluate_temp_energy(Psi1e.rdm2)
        e2 = self.Store.evaluate_temp_energy(Psi2e.rdm2)
        try:
            self.e0
        except AttributeError:
            self.e0 = self.Store.evaluate_energy()
        g1,g2= e1-self.e0,e2-self.e0

        d2D = (2*g2-2*self.d*g1)/(self.d*self.delta*self.delta*(self.d-1))
        d1D = (g1*self.d**2-g2)/(self.d*self.delta*(self.d-1))
        if abs(d2D)<1e-16:
            d2D=1e-16
        #
        # now, update for the Newton step
        #
        print('')
        print('--- Newton Step --- ')
        print('dE(d1): {:.10f},  dE(d2): {:.10f}'.format(
            np.real(g1),np.real(g2)))

        def damping(x):
            if self.damp_sigma==0:
                return 1
            else:
                return np.exp(-(x**2)/((self.damp_sigma)**2))

        damp = damping(max_val*(d1D/d2D))
        print('dE\'(0): {:.10f}, dE\'\'(0): {:.10f}'.format(
            np.real(d1D),np.real(d2D)))
        print('Step: {:.6f}, Largest: {:.6f}, Damping Factor: {:.6f}'.format(
            np.real(d1D/d2D),
            np.real(max_val*d1D/d2D),
            np.real(damp)))
        if d2D>0:
            if abs((d1D/d2D)*damp)<(self.delta*self.d):
                for f in testS:
                    f.qCo*= self.delta*self.d
                    f.c*= self.delta*self.d
            else:
                for f in testS:
                    f.qCo*= -(d1D/d2D)*damp
                    f.c*= -(d1D/d2D)*damp
            self.Store.update_ansatz(testS)
            Psi = Ansatz(self.Store,self.QuantStore,
                    **self.QuantStore.reTomo_kw)
            Psi.build_tomography()
            Psi.run_circuit()
            Psi.construct_rdm(variance=True)
            self.Store.rdm2=Psi.rdm2
            if abs(Psi.rdm2.trace()-2)>1e-3:
                print('Trace of 2-RDM: {}'.format(Psi.rdm2.trace()))
            if self.total.iter%3==0:
                self._calc_variance(Psi.rdm2_var,Psi)
        else:
            print('Hessian non-positive. Taking Euler step.')
            if g2<g1:
                for f in testS:
                    f.qCo*= self.delta*self.d
                    f.c*= self.delta*self.d
                self.Store.update_ansatz(testS)
                self.Store.rdm2 = Psi2e.rdm2
            else:
                for f in testS:
                    f.qCo*= 1/self.d
                    f.c*= 1/self.d
                self.Store.update_ansatz(testS)
                self.Store.rdm2 = Psi1e.rdm2


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
            if 'qc' in self.method:
                while not self.total.done:
                    self._run_qc_acse()
                    self._check()
            elif 'qq' in self.method:
                self.Store._get_HamiltonianOperators(full=True)
                while not self.total.done:
                    self._run_qq_acse()
                    self._check()
            elif 'cc' in self.method:
                while not self.total.done:
                    self._run_cc_ACSE()
                    self._check()
            elif 'aq' in self.method or 'ad' in self.method:
                while not self.total.done:
                    print('Time step: {}'.format(self.Store.t))
                    self._run_adiabatic_acse()
                    self._check()
            print('E, scf: {:.9f} H'.format(self.Store.hf.e_tot))
            print('E, run: {:.9f} H'.format(self.best))
            try:
                diff = 1000*(self.best-self.Store.e_casci)
                print('E, fci: {:.9f} H'.format(self.Store.e_casci))
                print('Energy difference from FCI: {:.8f} mH'.format(diff))
            except KeyError:
                pass
            rdm1 = self.Store.rdm2.reduce_order()
            print('Occupations of the 1-RDM:')
            print(np.real(np.diag(rdm1.rdm)))

    def _check(self):
        '''
        Internal check on the energy as well as norm of the S matrix
        '''
        # need to find energy
        if 'opt' in self.method:
            en = copy(self.Run.opt.best_f)
        else:
            en = self.Store.evaluate_energy()
        if self.total.iter==0:
            self.old = self.Store.hf.e_tot
        self.total.iter+=1
        print('---------------------------------------------')
        print('Step {:02}, Energy: {:.10f}, S: {:.10f}'.format(
            self.total.iter,
            np.real(en),
            np.real(self.norm)))
        if self.method in ['ac-acse','aq-acse']:
            if self.Store.t==float(1):
                self.total.done=True
        else:
            if self.total.iter==self.max_iter:
                self.total.done=True
        try:
            self.old
        except AttributeError:
            self.old = en
        if 'opt' in self.method:
            if self.old-en<self.crit:
                self.total.done=True
            else:
                print('Difference in energy: {:+.8f}'.format(self.old-en))
            self.old = copy(en)
            self.best = copy(e  n)
        else:
            if en<=self.old:
                self.old = en
            self.log_E.append(en)
            self.log_S.append(self.norm)
            i = 1
            temp_std_En = []
            temp_std_S = []
            std_En = 1
            std_S = 1
            avg_S = 1
            while i<=5 and self.total.iter>5:
                temp_std_En.append(self.log_E[-i])
                temp_std_S.append(self.log_S[-i])
                i+=1
            if self.total.iter>5:
                avg_En = np.real(np.average(np.asarray(temp_std_En)))
                avg_S =  np.real(np.average(np.asarray(temp_std_S)))
                std_En = np.real(np.std(np.asarray(temp_std_En)))
                std_S  = np.real(np.std(np.asarray(temp_std_S)))
                print('Standard deviation in energy: {:+.8f}'.format(std_En))
                print('Average energy: {:+.8f}'.format(avg_En))
                print('Standard deviation in S: {:.8f}'.format(std_S))
                print('Average S: {:.8f}'.format(avg_S))
                if self.QuantStore.backend=='statevector_simulator':
                    if en>self.best:
                        self.total.done=True
                    else:
                        self.best=np.real(en)
                else:
                    self.best = avg_En
            else:
                if en<self.best:
                    self.best = np.real(copy(en))
                else:
                    self.total.done=True

            print('---------------------------------------------')
            # implementing dynamic stopping criteria 
            if 'qq' in self.method or 'qc' in self.method:
                if std_En<self.crit and self.norm<0.05:
                    self.total.done=True
            self.e0 = en

    def save(self,
            name
            ):
        np.savetxt(name,np.asarray(
            [self.log_E,self.log_S]))


