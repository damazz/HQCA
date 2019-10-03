import pickle
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
from hqca.quantum import QuantumFunctions as qf
from hqca.quantum import NoiseSimulator as ns
from hqca.tools.util import Errors
from functools import reduce
import datetime
import sys
from hqca.tools import Preset as pre
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
        self.total=Cache()

    def build(self):
        QuantumRun._build_quantum(self)
        self.method = self.QuantStore.method # set method
        self.Store.method = self.method
        self.built=True

    def update_var(self,**kw):
        QuantumRun.update_var(self,**kw)
        self.Store.pr_m = self.kw['pr_m']

    def _run_qc_acse(self):
        testS = classS.findSPairs(self.Store)
        # 1. find S classically,
        # 2. prepare ansatz for euler or newton
        # 3. run the ansatz, give best guess
        if self.method=='qc-acse':
            #print('Eulers!')
            self.delta=0.1
            self.__euler_qc_acse(testS)
        elif self.method=='qc-acse2':
            self.delta = 0.25
            self.__newton_qc_acse(testS)

    def __euler_qc_acse(self,testS):
        '''
        function of Store.build_trial_ansatz
        '''
        # where is step size from? 
        # test procedure
        for s in testS:
            s.qCo*=self.delta
        self.Store.update_ansatz(testS)
        Psi = Ansatz(self.Store,self.QuantStore)
        Psi.build_tomography()
        Psi.run_circuit()
        Psi.construct_rdm()
        self.Store.rdm2=Psi.rdm2
        #Psi.rdm2.switch()
        #print(np.real(Psi.rdm2.rdm))
        #Psi.rdm2.switch()
        #print(self.Store.rdm1.rdm)


    def __newton_qc_acse(self,testS):
        # So, we make different Euler steps 
        # 1. Evaluate x
        hold = testS[:]
        for s in testS:
            s.qCo*=self.delta
            s.c*=self.delta
        self.Store.build_trial_ansatz(testS)
        Psi1e = Ansatz(self.Store,self.QuantStore,trialAnsatz=True)
        Psi1e.build_tomography()
        Psi1e.run_circuit()
        Psi1e.construct_rdm()
        # 2. Evaluate 2x
        d = 2
        for s in testS:
            s.qCo*=d
            s.c*=d
        self.Store.build_trial_ansatz(testS)
        Psi2e = Ansatz(self.Store,self.QuantStore,trialAnsatz=True)
        Psi2e.build_tomography()
        Psi2e.run_circuit()
        Psi2e.construct_rdm()
        # evaluate energies
        if d==1:
            sys.exit('b cannot be 1!')
        e1 = self.Store.evaluate_temp_energy(Psi1e.rdm2)
        e2 = self.Store.evaluate_temp_energy(Psi2e.rdm2)
        d2D = (e2-d*e1)/(self.delta*d*(d-1))
        d1D = e1-self.delta*d2D
        # now, update for the Newton step
        for s in hold:
            s.qCo*= -d1D/d2D
            s.c *= -d1D/d2D
        self.dx = abs(d1D/d2D)
        self.Store.update_ansatz(hold)
        Psi = Ansatz(self.Store,self.QuantStore)
        Psi.build_tomography()
        Psi.run_circuit()
        Psi.construct_rdm()
        self.Store.rdm2=Psi.rdm2
        #test = np.nonzero(Psi.rdm2.rdm)
        #for i,j,k,l in zip(test[0],test[1],test[2],test[3]):
        #    print(i,j,k,l,Psi.rdm2.rdm[i,j,k,l])
        #print('Trace : {}'.format(Psi.rdm2.trace()))
        #sys.exit()

    def _run_qq_acse(self):
        #self.Store._get_HamiltonianOperators(full=True)
        testS = quantS.findSPairsQuantum(self.Store,self.QuantStore)
        self.delta = 0.25
        # 1. find S quantumly,
        # 2. prepare ansatz for euler or newton
        # 3. run the ansatz, give best guess
        if self.method=='qq-acse':
            self.__euler_qc_acse(testS)
        elif self.method=='qq-acse2':
            self.__newton_qc_acse(testS)


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
            if self.method in ['qc-acse','qc-acse2']:
                while not self.total.done:
                    self._run_qc_acse()
                    self._check()
            elif self.method in ['qq-acse','qq-acse2']:
                self.Store._get_HamiltonianOperators(full=True)
                while not self.total.done:
                    self._run_qq_acse()
                    self._check()
            elif self.method in ['acse','cc-acse']:
                while not self.total.done:
                    self._run_cc_ACSE()
                    self._check()
            elif self.method in ['ac-acse','aq-acse','ac-acse2','aq-acse2']:
                while not self.total.done:
                    print('Time step: {}'.format(self.Store.t))
                    self._run_adiabatic_acse()
                    self._check()

    def _check(self):
        # need to find energy
        en = self.Store.evaluate_energy()
        self.total.iter+=1
        print('Step {:02}, Energy: {}'.format(self.total.iter,en))
        if self.method in ['ac-acse','aq-acse']:
            if self.Store.t==float(1):
                self.total.done=True
        else:
            if self.total.iter==self.Store.max_iter:
                self.total.done=True
            else:
                try:
                    self.old
                    #if self.old<en-0.005:
                    #    self.total.done=True
                except Exception:
                    self.old = en
        if en<self.old:
            self.old = en




