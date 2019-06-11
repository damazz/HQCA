'''

./tools/BuildCircuit.py

Two main classes:
    - GenerateDirectCircuit
        - used in general mapping
    - GenerateCompactCircuit
        - used in special mapping cases

Interfaces with qiskit to actually build the quantum circuits. Only handles the
construction. Also loads different gates from the algorithms folder

'''
from qiskit import execute
from qiskit.circuit import Parameter
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit
import hqca.quantum.algorithms._UCC as ucc
import hqca.quantum.algorithms._ECC as ecc
import hqca.quantum.algorithms._Entanglers as ent
import hqca.quantum.algorithms._Tomo as tomo
from math import pi
import traceback
import sys
tf_ibm_qx2 = {'01':True,'02':True, '12':True, '10':False,'20':False, '21':False}
tf_ibm_qx4 = {'01':False,'02':False, '12':False, '10':True,'20':True, '21':True}
# note, qx4 is 'raven', or 'tenerife'

def read_qasm(input_qasm):
    pass

class GenerateDirectCircuit:
    def __init__(
            self,
            QuantStore,
            _name=False,
            _flag_sign=False,
            _flag_tomo=False,
            ):
        '''
        Note is self.ents has np>1, then it needs to be changed in
        QuantumFunctions

        What is behavior we want:
            if ec_type=='ent', then we want it to go look for the different
            entangling type?
        '''
        self.ents = {
                'Ry_cN':{
                    'f':ent._ent1_Ry_cN,
                    'np':1,
                    'pre':False},
                'Uent1_cN':{
                    'f':ent._Uent1_cN,
                    'np':2,
                    'pre':False},
                'UCC1':{
                    'f':ucc._UCC1,
                    'np':1,
                    'pre':False},
                'UCC2':{
                    'f':ucc._UCC2_full,
                    'np':3,
                    'pre':False},
                'UCC2_1s':{
                    'f':ucc._UCC2_1s,
                    'np':1},
                'UCC2_1s+ec':{
                    'f':ucc._UCC2_1s,
                    'np':1,
                    'anc':1},
                'UCC2_2s':{
                    'f':ucc._UCC2_2s,
                    'np':1},
                'UCC2_4s':{
                    'f':ucc._UCC2_4s,
                    'np':1},
                'phase':{
                    'f':ent._phase,
                    'np':2}, #c12 - ++-- + +-+-
                }
        self.ec = {
                'parity':{
                    'f':ecc._ec_ucc2_parity_single,
                    'anc':1
                    },
                'pauli_UCC2_test':{
                    'f':ecc._ec_ancilla_UCC2_test_1s,
                    'anc':1,
                    'np':1,
                    },
                'ancilla_sign':{
                    'f':ecc._ec_ancilla_sign,
                    'anc':1,
                    },
                }
        self.para = QuantStore.parameters
        self._sign = _flag_sign
        self._tomo = _flag_tomo
        self.Np = len(self.para)
        self.qs = QuantStore
        self.Nq = QuantStore.Nq_tot
        self.q = QuantumRegister(self.Nq,name='q')
        #self.c = ClassicalRegister(4,name='c')
        self.c = ClassicalRegister(self.Nq,name='c')
        self.Ne = QuantStore.Ne
        if _name==False:
            self.qc = QuantumCircuit(self.q,self.c)
        else:
            self.qc = QuantumCircuit(self.q,self.c,name=_name)
        try:
            self.ent_p =  self.ents[self.qs.ent_circ_p]
            self.ent_q =  self.ents[self.qs.ent_circ_q]
        except KeyError:
            print('')
            print('Incorrect entangling gate.')
            print('Supported gates are: ')
            print(list(self.ents.keys()))
            sys.exit()
        self.ent_p =  self.ents[self.qs.ent_circ_p]['f']
        self.ent_Np = self.ents[self.qs.ent_circ_p]['np']
        self.ent_q =  self.ents[self.qs.ent_circ_q]['f']
        self.ent_Nq = self.ents[self.qs.ent_circ_q]['np']
        self.map = QuantStore.rdm_to_qubit
        if self.qs.ec_syndrome or self.qs.ec_comp_ent:
            self._gen_ecc_circuit()
        else:
            self._gen_circuit()

    def _initialize(self):
        self.Ne_alp = self.qs.Ne_alp
        self.Ne_bet = self.qs.Ne-self.qs.Ne_alp
        for i in range(0,self.Ne_alp):
            targ = self.qs.alpha_qb[i]
            self.qc.x(self.q[targ])
        for i in range(0,self.Ne_bet):
            targ = self.qs.beta_qb[i]
            self.qc.x(self.q[targ])


    def _gen_circuit(self):
        self._initialize()
        hp = 0
        for d in range(0,self.qs.depth):
            for n,pair in enumerate(self.qs.pair_list):
                a = self.ent_Np
                temp = self.para[hp:hp+a]
                self.ent_p(*temp,i=self.map[pair[0]],k=self.map[pair[1]])
                hp+=self.ent_Np
            for n,quad in enumerate(self.qs.quad_list):
                p,q,r,s,sign = quad[0],quad[1],quad[2],quad[3],quad[4]
                spin = quad[5]
                a = self.ent_Nq
                temp = self.para[hp:hp+a]
                if self.qs.ent_circ_q=='UCC2_2s' and h==0:
                    ucc._UCC2_1s(self,*temp,i=p,j=q,k=r,l=s,
                            operator=sign,
                            spin=spin)
                else:
                    self.ent_q(self,*temp,i=p,j=q,k=r,l=s,
                            operator=sign,
                            spin=spin)
                hp+= self.ent_Nq

    def _gen_ecc_circuit(self):
        self._initialize()
        h = 0 #h keeps track of parameters
        anc = 0
        for d in range(0,self.qs.depth):
            for n,pair in enumerate(self.qs.pair_list):
                a = self.ent_Np
                temp = self.para[h:h+a]
                self.ent_p(*temp,i=self.map[pair[0]],k=self.map[pair[1]])
                h+=self.ent_Np
            for n,quad in enumerate(self.qs.quad_list):
                p,q,r,s,sign = quad[0],quad[1],quad[2],quad[3],quad[4]
                spin = quad[5]
                a = self.ent_Nq
                temp = self.para[h:h+a]
                # first, check if we are using a composite circuit
                if self.qs.ec_comp_ent: 
                    temp_rep = self.qs.ec_replace_quad[n]
                    if temp_rep['replace'] in [False,0]:
                        if self.qs.ent_circ_q=='UCC2_2s' and h==0:
                            ucc._UCC2_1s(self,*temp,i=p,j=q,k=r,l=s,
                                    operator=sign,
                                    spin=spin)
                        else:
                            self.ent_q(self,*temp,i=p,j=q,k=r,l=s,
                                    operator=sign,
                                    spin=spin)
                    else:
                        if temp_rep['use']=='sign' and not self._sign:
                            if self.qs.ent_circ_q=='UCC2_2s' and h==0:
                                ucc._UCC2_1s(self,*temp,i=p,j=q,k=r,l=s,
                                        operator=sign,
                                        spin=spin)
                            else:
                                self.ent_q(self,*temp,i=p,j=q,k=r,l=s,
                                        operator=sign,
                                        spin=spin)
                        else:
                            temp_Cq=self.ec[temp_rep['circ']]['f']
                            temp_Np=self.ec[temp_rep['circ']]['np']
                            temp_Na=self.ec[temp_rep['circ']]['anc']
                            temp_kw=temp_rep['kw']
                            temp_anc = temp_rep['ancilla']
                            if not temp_Np==a:
                                print('See BuildCircuit.py')
                                sys.exit('Parameter mismatch quad gates.')
                            elif not temp_Na==len(temp_anc):
                                print('See BuildCircuit.py')
                                text = 'Mismatch between circuit and ancilla.'
                                sys.exit(text)
                            else:
                                temp_Cq(self,*temp,
                                        i=p,j=q,k=r,l=s,
                                        anc=temp_anc,
                                        operator=sign,spin=spin,
                                        **temp_kw
                                        )
                else:
                    if self.qs.ent_circ_q=='UCC2_2s' and h==0:
                        ucc._UCC2_1s(self,*temp,i=p,j=q,k=r,l=s,
                                operator=sign,
                                spin=spin)
                    else:
                        self.ent_q(self,*temp,i=p,j=q,k=r,l=s,
                                operator=sign,
                                spin=spin)
                if self.qs.ec_syndrome:
                    for synd,locations in self.qs.syndromes.items():
                        m = n+len(self.qs.pair_list)
                        temp = locations[m]
                        if temp['use'] in [False,0]:
                            pass
                        else:
                            if temp['use']=='sign' and (not self._sign):
                                pass
                            else:
                                temp_anc = temp['ancilla']
                                temp_Cq = self.ec[temp['circ']]['f']
                                temp_Na = self.ec[temp['circ']]['anc']
                                temp_kw = temp['kw']
                                if not temp_Na==len(temp_anc):
                                    print('See BuildCircuit.py')
                                    text = 'Mismatch between circuit and ancilla.'
                                    sys.exit(text)
                                temp_Cq(self,i=p,j=q,k=r,l=s,
                                        anc=temp_anc,
                                        **temp_kw)
                h+= self.ent_Nq

class GenerateCompactCircuit:
    '''
    Class for initializing quantum algorithms. Has the quantum circuit
    attribute 
    '''
    def __init__(
            self,
            para,
            algorithm,
            order='default',
            _name=False,
            pr_q=0,
            **kwargs
            ):
        self.qa = algorithm
        self.Nq = algorithm_tomography[self.qa]['Nq']
        self.q = QuantumRegister(self.Nq,name='q')
        self.c = ClassicalRegister(self.Nq,name='c')
        if _name==False:
            self.qc = QuantumCircuit(self.q,self.c)
        else:
            self.qc = QuantumCircuit(self.q,self.c,name=_name)
        self.p  = []
        for i in para:
            self.p.append(i*pi/90)
        self.apply_algorithm(order)

    def apply_algorithm(self,order='default'):
        '''
        List of algorithms for use with quantum computer. Reminder to read
        the top of this module before adding or removing algorithms.
        '''
        if self.qa=='affine_2p_curved_tenerife':
            if order=='default':
                o = [0,2,1,0]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.ry(self.p[1],self.q[o[2]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
        elif self.qa=='affine_2p_flat_tenerife':
            if order=='default':
                o = [2,1,1,0]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.ry(self.p[1],self.q[o[2]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
        elif self.qa=='affine_2p_flatfish_tenerife':
            if order=='default':
                o = [0,2,2,1]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.ry(self.p[1],self.q[o[1]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
        elif self.qa=='3qtest':
            if order=='default':
                o = [0,2,0,1,2,1]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.ry(self.p[1],self.q[o[1]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.ry(self.p[2],self.q[o[2]])
            self.qc.ry(self.p[3],self.q[o[3]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
            self.qc.ry(self.p[4],self.q[o[4]])
            self.qc.ry(self.p[5],self.q[o[5]])
            self.qc.cx(self.q[o[4]],self.q[o[5]])
        else:
            pass
