import numpy as np
import sys
np.set_printoptions(linewidth=200)
from hqca.tools.EnergyFunctions import Storage
from hqca.quantum.QuantumFunctions import QuantumStorage
from hqca.tools import Functions as fx
from hqca.tools import Chem as chem
from hqca.tools.RDM import RDMs
from functools import reduce
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver,UnitsType



class ModStorageACSE(Storage):
    '''
    modified Storage object, more well suited for containing the ACSE related
    objets, such as the 2S matrix
    '''
    def __init__(self,**kwargs):
        Storage.__init__(self,**kwargs)
        Storage.gas(self)
        Storage.gsm(self)
        self.modified='ACSE'
        self.r = self.No_tot*2 # spin orbitals
        # if hartree fock
        self.max_iter = self.kw['max_iter']
        #self.time = self.kw['time']
        self.dt = 0.1
        self.t = 0
        self.rdm2 = RDMs(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='hf',
                Ne=self.Ne_as,
                S=self.Ne_alp-self.Ne_bet)
        self.rdm3 = RDMs(
                order=3,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='hf',
                Ne=self.Ne_as,
                S=self.Ne_alp-self.Ne_bet)
        self.rdm1 = self.rdm2.reduce_order()
        self.update_full_ints()
        self.get_qiskit_ints2e()
        self.ints_2e = fx.expand(self.ints_2e)
        self.K2 = fx.contract(self.ints_2e)
        self.K1 = self.ints_1e
        self.nonH2 = np.nonzero(self.ints_2e)
        self.zipH2 = list(zip(
                self.nonH2[0],self.nonH2[1],
                self.nonH2[2],self.nonH2[3]))
        self.nonH1 = np.nonzero(self.ints_1e)
        self.zipH1 = list(zip(self.nonH1[0],self.nonH1[1]))
        self.active_rdm2e = list(np.nonzero(self.rdm2.rdm))
        self.active_rdm3e = list(np.nonzero(self.rdm3.rdm))
        self.F = chem.gen_spin_1ei(
                self.f.copy(),
                #np.identity(2),
                #np.identity(2),
                self.T_alpha.T,
                self.T_beta.T,
                self.alpha_mo,
                self.beta_mo,
                region='full',
                spin2spac=self.s2s
                )
        self.nonF = np.nonzero(self.F)
        self.zipF = list(zip(self.nonF[0],self.nonF[1]))
        self.ansatz = []

    def get_qiskit_ints2e(self):
        self.ints_2e_qiskit = chem.gen_spin_2ei_QISKit(
                self.ints_2e_ao,
                self.T_alpha.T,
                self.T_beta.T,
                self.alpha_mo,
                self.beta_mo,
                region='full',
                spin2spac=self.s2s)

    def evaluate_energy(self):
        self.rdm1 = self.rdm2.reduce_order()
        e_h1 = reduce(np.dot, (self.K1,self.rdm1.rdm)).trace()
        self.rdm2.switch()
        e_h2 = reduce(np.dot, (self.K2,self.rdm2.rdm)).trace()
        self.rdm2.switch()
        return e_h1 + 0.5*e_h2 + self.E_ne

    def evaluate_temp_energy(self,rdm2):
        rdm1 = rdm2.reduce_order()
        e_h1 = reduce(np.dot, (self.K1,rdm1.rdm)).trace()
        rdm2.switch()
        e_h2 = reduce(np.dot, (self.K2,rdm2.rdm)).trace()
        return e_h1 + 0.5*e_h2 + self.E_ne


    def update_ansatz(self,newS):
        '''
        takes given S and actually adds it onto the new ansatz:

        NEEDS WORK. DRASTICALLY
        '''
        #for fermi in newS:
        #    if len(self.ansatz)==0:
        #        self.ansatz.append(fermi)
        #    elif fermi.hasSameInd(self.ansatz[-1]):
        #        self.ansatz[-1].qCo+= fermi.qCo
        #        self.ansatz[-1].c+=fermi.c
        #    else:
        #        self.ansatz.append(fermi)
        for fermi in newS:
            new = True
            for old in self.ansatz:
                if fermi.hasSameInd(old):
                    old.qCo+= fermi.qCo
                    old.c += fermi.c
                    new = False
            if new:
                self.ansatz.append(fermi)


    def build_trial_ansatz(self,testS):
        '''
        takes terms from newS and adds them to ansatz as a trial
        '''
        self.tempAnsatz = self.ansatz[:]
        for fermi in testS:
            self.tempAnsatz.append(fermi)

    def _check_commutivity(self):
        '''
        will check if terms in S commute and can be applied to earlier instances
        on the ansatz
        '''
        pass

    def _get_HamiltonianOperators(self,full=True):
        test = np.nonzero(self.ints_2e_qiskit)
        if full:
            ferOp = FermionicOperator(h1=self.ints_1e,
                    h2=0.5*self.ints_2e_qiskit)
            qubOp = ferOp.mapping('JORDAN_WIGNER')
        else:
            sys.exit('Havent configured potential Hamiltonian V yet.')
            ferOp = FermionicOperator(h1=self.ints_1e,
                    h2=self.ints_2e)
            qubOp = ferOp.mapping('JORDAN_WIGNER')
        #print(self.ints_1e)
        #print(self.ints_2e_qiskit)
        print('------------------------------------------')
        print('Here is the qubit Hamiltonian: ')
        print('------------------------------------------')
        print(qubOp.print_operators())
        print('------------------------------------------')
        self.qubOp = qubOp.paulis


