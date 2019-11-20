import numpy as np
from copy import deepcopy as copy
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
from hqca.tools import RDMFunctions as rdmf


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
        self.get_HF_rdm()
        self.get_FCI_rdm()
        self._set_overlap()

    def _set_overlap(self):
        self.d_hf_fci =  self.hf_rdm2.get_overlap(self.fci_rdm2)
        print('Distance between HF, FCI: {:.8f}'.format(
            np.real(self.d_hf_fci)))

    def get_HF_rdm(self):
        self.hf_rdm2 = RDMs(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='hf',
                Ne=self.Ne_as,
                )


    def get_FCI_rdm(self):
        d1,d2 = self.mc.fcisolver.make_rdm12s(
                self.mc.ci,self.No_as,self.Ne_as)
        fci_rdm2 = np.zeros((
            self.No_as*2,self.No_as*2,self.No_as*2,self.No_as*2))
        for i in self.alpha_mo['active']:
            for k in self.alpha_mo['active']:
                for l in self.alpha_mo['active']:
                    for j in self.alpha_mo['active']:
                        p,q = self.s2s[i],self.s2s[j]
                        r,s = self.s2s[k],self.s2s[l]
                        fci_rdm2[i,k,j,l] = d2[0][p,q,r,s]
        for i in self.alpha_mo['active']:
            for k in self.beta_mo['active']:
                for l in self.beta_mo['active']:
                    for j in self.alpha_mo['active']:
                        #if i>=j and k>=l:
                        #    continue
                        p,q = self.s2s[i],self.s2s[j]
                        r,s = self.s2s[k],self.s2s[l]
                        fci_rdm2[i,k,j,l]+= d2[1][p,q,r,s]
                        fci_rdm2[k,i,j,l]+= -d2[1][p,q,r,s]
                        fci_rdm2[i,k,l,j]+= -d2[1][p,q,r,s]
                        fci_rdm2[k,i,l,j]+= d2[1][p,q,r,s]

        for i in self.beta_mo['active']:
            for k in self.beta_mo['active']:
                for l in self.beta_mo['active']:
                    for j in self.beta_mo['active']:
                        p,q = self.s2s[i],self.s2s[j]
                        r,s = self.s2s[k],self.s2s[l]
                        fci_rdm2[i,k,j,l] = d2[2][p,q,r,s]
        self.fci_rdm2 = RDMs(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='given',
                rdm = fci_rdm2,
                Ne=self.Ne_as,
                )
        self.fci_rdm2.contract()

    def get_qiskit_ints2e(self):
        self.ints_2e_qiskit = chem.gen_spin_2ei_QISKit(
                self.ints_2e_ao[:],
                self.T_alpha.T,
                self.T_beta.T,
                self.alpha_mo,
                self.beta_mo,
                region='full',
                spin2spac=self.s2s)
        self.ints_2e = fx.expand(self.ints_2e)

    def evaluate_energy(self):
        self.rdm1 = self.rdm2.reduce_order()
        eig,eiv = np.linalg.eig(self.rdm1.rdm)
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
        rdm2.switch()
        return e_h1 + 0.5*e_h2 + self.E_ne

    def update_ansatz(self,newS):
        '''
        takes given S and actually adds it onto the new ansatz:

        NEEDS WORK. DRASTICALLY
        '''
        print('Current S:')
        for fermi in self.ansatz:
            print('{}: {:+.10f}'.format(
                fermi.ind,
                np.real(fermi.c))
                )
            fermi.clear()
        for fermi in newS:
            new = True
            for old in self.ansatz:
                if fermi.isSame(old):
                    old.qCo = fermi.qCo+old.qCo
                    old.c  = fermi.c+old.c
                    new = False
                    break
                elif fermi.isHermitian(old):
                    old.qCo = fermi.qCo-old.qCo
                    old.c  = fermi.c-old.c
                    new = False
                    break
            if new:
                self.ansatz.append(fermi)
        print('Updated New S:')
        for fermi in self.ansatz:
            print('{}: {:+.10f}  ({:+.5f})'.format(
                fermi.ind,
                np.real(fermi.c),
                np.real(fermi.qCo))
                )
            fermi.generateAntiHermitianExcitationOperators()

    def acse_analysis(self):
        print('  --  --  --  --  --  --  -- ')
        print('--  --  --  --  --  --  --  --')
        self.rdm2.get_spin_properties()
        print('Sz: {:.8f}'.format(np.real(self.rdm2.sz)))
        print('S2: {:.8f}'.format(np.real(self.rdm2.s2)))
        ovlp = self.rdm2.get_overlap(self.fci_rdm2)
        print('Distance from FCI RDM: {:.8f}'.format(
            ovlp))
        print('Normalized distance: {:.8f}'.format(
            ovlp/self.d_hf_fci))

    def build_trial_ansatz(self,testS):
        '''
        takes terms from newS and adds them to ansatz as a trial
        '''
        self.tempAnsatz = copy(self.ansatz)
        for fermi in self.tempAnsatz:
            fermi.clear()
        for fermi in testS:
            new = True
            for old in self.tempAnsatz:
                if fermi.isSame(old):
                    old.qCo = fermi.qCo+old.qCo
                    old.c  = fermi.c+old.c
                    new = False
                    break
                elif fermi.isHermitian(old):
                    old.qCo = fermi.qCo-old.qCo
                    old.c  = fermi.c-old.c
                    new = False
                    break
            if new:
                self.tempAnsatz.append(fermi)
        print('Temporary New S: ')
        for fermi in self.tempAnsatz:
            print('{}: {:+.10f}'.format(
                fermi.ind,
                np.real(fermi.c))
                )
            fermi.generateAntiHermitianExcitationOperators()

    def _check_commutivity(self):
        '''
        will check if terms in S commute and can be applied to earlier instances
        on the ansatz
        '''
        pass


    def _get_HamiltonianOperators(self,full=True,int_thresh=1e-14):
        if self.qubitH=='qiskit':
            if full:
                ferOp=FermionicOperator(h1=1.0*self.ints_1e,
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
        elif self.qubitH in ['local','split-local']:
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
                        if abs(self.ints_1e[p,q])<int_thresh:
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
                    term_pqqp = self.ints_2e[p,q,p,q]
                    term_pqqp-= self.ints_2e[p,q,q,p]
                    if abs(term_pqqp)>int_thresh:
                        c1 =  (p in self.alpha_mo['active'])
                        c2 =  (q in self.alpha_mo['active'])
                        spin = '{}{}{}{}'.format(
                                c1*'a'+(1-c1)*'b',
                                c2*'a'+(1-c2)*'b',
                                c2*'a'+(1-c2)*'b',
                                c1*'a'+(1-c1)*'b',
                                )
                        newOp = Fermi(
                                coeff=term_pqqp,
                                indices=[p,q,q,p],
                                sqOp='++--',
                                spin=spin)
                        operators['nn'].append(newOp)
            #
            # prrq operators
            #
            for p in alp+bet:
                for q in alp+bet:
                    for r in alp+bet:
                        if p>=q or ((p in alp) and (q in bet)):
                            continue
                        if r==p or r==q:
                            continue
                        ops = '++--'
                        term_pqrr = self.ints_2e[p,r,q,r] #aaab
                        term_pqrr-= self.ints_2e[p,r,r,q]
                        c1 =  (p in alp)
                        c2 =  (r in alp)
                        spin = '{}{}{}{}'.format(
                                c1*'a'+(1-c1)*'b',c2*'a'+(1-c2)*'b',
                                c2*'a'+(1-c2)*'b',c1*'a'+(1-c1)*'b')
                        if abs(term_pqrr)>int_thresh:
                            newOp = Fermi(
                                    coeff=term_pqrr,
                                    indices=[p,r,r,q],
                                    sqOp='++--',
                                    spin=spin
                                    )
                            operators['ne'].append(newOp)
            # prsq operators
            for p in alp:
                for r in alp:
                    if p>=r:
                        continue
                    for s in alp:
                        if r>=s:
                            continue
                        for q in alp:
                            if s>=q:
                                continue
                            term1 = self.ints_2e[p,r,q,s]-self.ints_2e[p,r,s,q]
                            term2 = self.ints_2e[p,s,q,r]-self.ints_2e[p,s,r,q]
                            term3 = self.ints_2e[p,q,r,s]-self.ints_2e[p,q,s,r]
                            if abs(term1)>int_thresh:
                                newOp = Fermi(
                                        coeff=term1,
                                        indices=[p,r,s,q],
                                        sqOp='++--',
                                        spin='aaaa'
                                        )
                                operators['de'].append(newOp)
                            if abs(term2)>int_thresh:
                                newOp = Fermi(
                                        coeff=term2,
                                        indices=[p,s,r,q],
                                        sqOp='++--',
                                        spin='aaaa'
                                        )
                                operators['de'].append(newOp)
                            if abs(term3)>int_thresh:
                                newOp = Fermi(
                                        coeff=term3,
                                        indices=[p,q,s,r],
                                        sqOp='++--',
                                        spin='aaaa'
                                        )
                                operators['de'].append(newOp)
            for p in bet:
                for r in bet:
                    if p>=r:
                        continue
                    for s in bet:
                        if r>=s:
                            continue
                        for q in bet:
                            if s>=q:
                                continue
                            term1 = self.ints_2e[p,r,q,s]-self.ints_2e[p,r,s,q]
                            term2 = self.ints_2e[p,s,q,r]-self.ints_2e[p,s,r,q]
                            term3 = self.ints_2e[p,q,r,s]-self.ints_2e[p,q,s,r]
                            if abs(term1)>int_thresh:
                                newOp = Fermi(
                                        coeff=term1,
                                        indices=[p,r,s,q],
                                        sqOp='++--',
                                        spin='bbbb'
                                        )
                                operators['de'].append(newOp)
                            if abs(term2)>int_thresh:
                                newOp = Fermi(
                                        coeff=term2,
                                        indices=[p,s,r,q],
                                        sqOp='++--',
                                        spin='bbbb',
                                        )
                                operators['de'].append(newOp)
                            if abs(term3)>int_thresh:
                                newOp = Fermi(
                                        coeff=term3,
                                        indices=[p,q,s,r],
                                        sqOp='++--',
                                        spin='bbbb'
                                        )
                                operators['de'].append(newOp)
            for p in alp:
                for r in bet:
                    for s in bet:
                        if r>=s:
                            continue
                        for q in alp:
                            if p>=q:
                                continue
                            print(p,r,s,q)
                            term1 = self.ints_2e[p,r,q,s]
                            term2 = self.ints_2e[p,s,q,r]
                            if abs(term1)>int_thresh:
                                newOp = Fermi(
                                        coeff=term1,
                                        indices=[p,r,s,q],
                                        sqOp='++--',
                                        spin='abba'
                                        )
                                operators['de'].append(newOp)
                            if abs(term2)>int_thresh:
                                newOp = Fermi(
                                        coeff=term2,
                                        indices=[p,s,r,q],
                                        sqOp='++--',
                                        spin='abba',
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
                    if abs(tc[i])<int_thresh:
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
            op_list = []
            for item1 in operators['se']:
                op_list.append(item1)
            for n,item2 in enumerate(operators['ne']):
                op_list.append(item2)
            tp ,tc = [],[]
            for item in op_list:
                for p,c in zip(item.pauliExp,item.pauliCoeff):
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
                print('P: {}, Value: {:.6f}'.format(p,c))
