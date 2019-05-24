#from qiskit import register, available_backends, get_backend
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
'''

./tools/GenerateCircuit.py

Two main classes:
    - GenerateDirectCircuit
        - used in general mapping
    - GenerateCompactCircuit
        - used in special mapping cases

'''
tf_ibm_qx2 = {'01':True,'02':True, '12':True, '10':False,'20':False, '21':False}
tf_ibm_qx4 = {'01':False,'02':False, '12':False, '10':True,'20':True, '21':True}
# note, qx4 is 'raven', or 'tenerife'

def read_qasm(input_qasm):
    pass

class GenerateDirectCircuit:
    def __init__(
            self,
            QuantStore,
            _name=False
            ):
        '''
        Want to do a different approach than previously. Want to make a
        simulated one that has variable size constraints.

        Note is self.ents has np>1, then it needs to be changed in
        QuantumFunctions
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
                    }
                }
        self.para = QuantStore.parameters
        self.Np = len(self.para)
        self.para = [Parameter('p{}'.format(i)) for i in range(self.Np)]
        self.qs = QuantStore
        self.Nq = QuantStore.Nq_tot
        self.q = QuantumRegister(self.Nq,name='q')
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
        self.anc_qb = self.qs.ancilla_qb
        if self.qs.ec:
            self.ec_q = self.ec[self.qs.ec_circ]['f']
            self.ec_Nq = self.ec[self.qs.ec_circ]['anc']
        self.map = QuantStore.rdm_to_qubit
        self.cg = 0
        self.sg = 0
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
        self.sg+= self.Ne

    def _gen_circuit(self):
        self._initialize()
        h,n = 0,0
        for d in range(0,self.qs.depth):
            for pair in self.qs.pair_list:
                a = self.ent_Np
                temp = self.para[h:h+a]
                self.ent_p(*temp,i=self.map[pair[0]],k=self.map[pair[1]])
                h+=self.ent_Np
                n+=1 
            for quad in (self.qs.qc_quad_list):
                p,q,r,s,sign = quad[0],quad[1],quad[2],quad[3],quad[4]
                spin = quad[5]
                a = self.ent_Nq
                temp = self.para[h:h+a]
                if self.qs.ent_circ_q=='UCC2_2s' and h==0:
                    ucc._UCC2_1s(self,*temp,
                            i=p,
                            j=q,
                            k=r,
                            l=s,
                            operator=sign,
                            spin=spin)
                else:
                    self.ent_q(self,*temp,
                            i=p,
                            j=q,
                            k=r,
                            l=s,
                            operator=sign,
                            spin=spin)
                if self.qs.ec and ('s' in self.qs.ec_type):
                    anc = self.qs.ancilla_qb[n:n+self.ec_Nq]
                    if self.qs.ec_ent_list[n]==1:
                        self.ec_q(self,
                                i=p,
                                j=q,
                                k=r,
                                l=s,
                                an=anc
                                )
                    elif not self.qs.ec_ent_list[n] in [0,None,False]:
                        sys.exit('Build Circuit: not configured yet.')
                h+= self.ent_Nq
                if self.qs.ec:
                    n+= self.ec_Nq


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
