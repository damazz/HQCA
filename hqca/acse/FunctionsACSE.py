import numpy as np
from collections import Counter
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
from hqca.tools.Fermi import FermiOperator as Fermi



class ModStorageACSE(Storage):
    '''
    modified Storage object, more well suited for containing the ACSE related
    objets, such as the 2S matrix
    '''
    def __init__(self,
            qubit_hamiltonian='local',
            **kwargs):
        Storage.__init__(self,**kwargs)
        Storage.gas(self)
        Storage.gsm(self)
        self.modified='ACSE'
        self.r = self.No_tot*2 # spin orbitals
        self.max_iter = self.kw['max_iter']
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
        self.qubitH = qubit_hamiltonian
        self.zipH1 = list(zip(self.nonH1[0],self.nonH1[1]))
        self.active_rdm2e = list(np.nonzero(self.rdm2.rdm))
        self.active_rdm3e = list(np.nonzero(self.rdm3.rdm))
        self.F = chem.gen_spin_1ei(
                self.f.copy(),
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
        print(self.rdm1.rdm)
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
        print('Old S:')
        for fermi in self.ansatz:
            print(fermi.ind,fermi.c)
        for fermi in newS:
            new = True
            for old in self.ansatz:
                if fermi.isSame(old) or fermi.isHermitian(old):
                    old.qCo = fermi.qCo+old.qCo
                    old.c  = fermi.c+old.c
                    new = False
            if new:
                self.ansatz.append(fermi)
        print('New S:')
        for fermi in self.ansatz:
            print(fermi.ind,fermi.c)


    def build_trial_ansatz(self,testS):
        '''
        takes terms from newS and adds them to ansatz as a trial
        '''
        self.tempAnsatz = self.ansatz[:]
        #for fermi in testS:
        #    self.tempAnsatz.append(fermi)
        for fermi in testS:
            new = True
            for old in self.tempAnsatz:
                if fermi.isSame(old) or fermi.isHermitian(old):
                    old.qCo = fermi.qCo+old.qCo
                    old.c  = fermi.c+old.c
                    new = False
            if new:
                self.tempAnsatz.append(fermi)
        print('New S: ')
        for fermi in self.tempAnsatz:
            print(fermi.ind,fermi.c)

    def _check_commutivity(self):
        '''
        will check if terms in S commute and can be applied to earlier instances
        on the ansatz
        '''
        pass


    def _get_HamiltonianOperators(self,full=True):
        if self.qubitH=='qiskit':
            if full:
                ferOp=FermionicOperator(h1=self.ints_1e,
                        h2=-0.5*self.ints_2e_qiskit)
                qubOp=ferOp.mapping('JORDAN_WIGNER')
            else:
                sys.exit('Havent configured potential Hamiltonian V yet.')
                ferOp = FermionicOperator(h1=self.ints_1e,
                        h2=self.ints_2e)
                qubOp = ferOp.mapping('JORDAN_WIGNER')
            print('------------------------------------------')
            print('Here is the qubit Hamiltonian: ')
            print('------------------------------------------')
            print(qubOp.print_operators())
            print('------------------------------------------')
            self.qubOp = qubOp.paulis
        elif self.qubitH=='local':
            alp = self.alpha_mo['active']
            bet = self.beta_mo['active']
            blocks = [
                    [alp,bet],
                    [alp,bet]
                    ]
            spins = [['a','b'],['a','b']]
            operators = {
                    'no':[],
                    'se':[],
                    'nn':[],
                    'ne':[],
                    'de':[]}
            for ze in range(len(blocks[0])):
                for p in blocks[0][ze]:
                    for q in blocks[1][ze]:
                        if p>q:
                            continue
                        if abs(self.ints_1e[p,q])<1e-10:
                            continue
                        spin = ''.join([spins[0][ze],spins[1][ze]])
                        newOp = Fermi(
                                coeff=self.ints_1e[p,q],
                                indices=[p,q],
                                sqOp='+-',
                                spin=spin
                                )
                        if p==q:
                            operators['no'].append(newOp)
                        else:
                            operators['se'].append(newOp)
            for p in alp+bet:
                for q in alp+bet:
                    if p>=q:
                        continue
                    term = self.ints_2e[p,q,p,q]
                    term-= self.ints_2e[p,q,q,p]
                    if abs(term)>1e-10:
                        c1 =  (p in self.alpha_mo['active'])
                        c2 =  (q in self.alpha_mo['active'])
                        spin = '{}{}{}{}'.format(
                                c1*'a'+(1-c1)*'b',
                                c2*'a'+(1-c2)*'b',
                                c2*'a'+(1-c2)*'b',
                                c1*'a'+(1-c1)*'b',
                                )
                        newOp = Fermi(
                                coeff=term,
                                indices=[p,q,q,p],
                                sqOp='++--',
                                spin=spin)
                        operators['nn'].append(newOp)
            for p in alp+bet:
                for q in alp+bet:
                    for r in alp+bet:
                        if p>=q:
                            continue
                        if r==p or r==q:
                            continue
                        ops = '++--'
                        term = self.ints_2e[p,r,q,r]
                        c1 =  (p in alp)
                        c2 =  (r in alp)
                        spin = '{}{}{}{}'.format(
                                c1*'a'+(1-c1)*'b',c2*'a'+(1-c2)*'b',
                                c2*'a'+(1-c2)*'b',c1*'a'+(1-c1)*'b')
                        if abs(term)>1e-10:
                            newOp = Fermi(
                                    coeff=term,
                                    indices=[p,r,r,q],
                                    sqOp='++--',
                                    spin=spin
                                    )
                            operators['ne'].append(newOp)
            for p in alp+bet:
                for q in alp+bet:
                    if p>=q:
                        continue
                    for r in alp+bet:
                        if p>=r:
                            continue
                        for s in alp+bet:
                            if r>=s:
                                continue
                            elif q==r or q==s:
                                continue
                            c1 =  (p in alp)
                            c2 =  (r in alp)
                            c3 =  (s in alp)
                            c4 =  (q in alp)
                            spin = '{}{}{}{}'.format(
                                    c1*'a'+(1-c1)*'b',c2*'a'+(1-c2)*'b',
                                    c3*'a'+(1-c3)*'b',c4*'a'+(1-c4)*'b')
                            term1 = self.ints_2e[p,r,q,s]
                            term2 = self.ints_2e[p,s,q,r]
                            if abs(term1)>1e-10:
                                newOp = Fermi(
                                        coeff=term1,
                                        indices=[p,r,s,q],
                                        sqOp='++--',
                                        spin=spin
                                        )
                                operators['de'].append(newOp)
                            if abs(term2)>1e-10:
                                newOp = Fermi(
                                        coeff=term2,
                                        indices=[p,s,r,q],
                                        sqOp='++--',
                                        spin=spin
                                        )
                                operators['de'].append(newOp)
            for k,v in operators.items():
                print('k: {}'.format(k))
                for item in v:
                    print(item.qInd,item.qOp,item.qSp,item.ind,item.qCo)
            for key, item in operators.items():
                for op in item:
                    op.generateHermitianExcitationOperators(Nq=2*self.No_as)

            def simplify(tp,tc):
                done = False
                def check_duplicate(paulis):
                    for n in range(len(paulis)):
                        for m in range(n):
                            if paulis[n]==paulis[m]:
                                return m,n
                    return False,False
                while not done:
                    n,m = check_duplicate(tp)
                    if n==False and m==False:
                        done=True
                    else:
                        tc[n]+= tc.pop(m)
                        tp.pop(m)
                for i in reversed(range(len(tp))):
                    if abs(tc[i])<1e-10:
                        tp.pop(i)
                        tc.pop(i)
                return tp,tc
            # 
            # now, start simplification in subgroups
            #
            pauli = []
            coeff = []
            tp,tc = [],[]
            for op in operators['no']:
                for p,c in zip(op.pauliExp,op.pauliCoeff):
                    tp.append(p)
                    tc.append(c)
            for op in operators['nn']:
                for p,c in zip(op.pauliExp,op.pauliCoeff):
                    tp.append(p)
                    tc.append(c)
            tp,tc = simplify(tp,tc)
            pauli += tp[:]
            coeff += tc[:]
            added = []
            op_list = []
            for item1 in operators['se']:
                op_list.append(item1)
                for n,item2 in enumerate(operators['ne']):
                    inds = [item2.qInd[0]==item2.qInd[1],
                            item2.qInd[1]==item2.qInd[2],
                            item2.qInd[2]==item2.qInd[3]]
                    c1 = item1.qInd[0]==item2.qInd[0]
                    c2 = item1.qInd[1]==item2.qInd[3]
                    if inds[0] and item1.qInd==item2.qInd[2:]:
                        op_list.append(item2)
                        added.append(n)
                    elif inds[2] and item1.qInd==item2.qInd[0:2]:
                        op_list.append(item2)
                        added.append(n)
                    elif inds[1] and c1 and c2:
                        op_list.append(item2)
                        added.append(n)

            tp ,tc = [],[]
            for item in op_list:
                for p,c in zip(item.pauliExp,item.pauliCoeff):
                    tp.append(p)
                    tc.append(c)
            tp,tc = simplify(tp,tc)
            pauli += tp[:]
            coeff += tc[:]



            tp ,tc = [],[]
            for op in operators['ne']:
                for p,c in zip(op.pauliExp,op.pauliCoeff):
                    tp.append(p)
                    tc.append(c)
            tp,tc = simplify(tp,tc)
            pauli += tp[:]
            coeff += tc[:]
            


            # simplification procedure for double excitations
            tp,tc = [],[]
            for item in operators['de']:
                for p,c in zip(item.pauliExp,item.pauliCoeff):
                    tp.append(p)
                    tc.append(c)
            tp,tc = simplify(tp,tc)
            pauli += tp[:]
            coeff += tc[:]

            #  # # # done
            self.qubOp = [pauli,coeff]
            for p,c in zip(self.qubOp[0],self.qubOp[1]):
                print(p,c)





