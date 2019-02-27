#from qiskit import register, available_backends, get_backend
from qiskit import execute
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit
from math import pi
import traceback

'''

./tools/QuantumAlgorithms.py

Two main classes:
    - GenerateDirectCircuit
        - used in general mapping
    - GenerateCompactCircuit
        - used in special mapping cases
    - 

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
        '''
        self.ents = {
                'Ry_cN':{
                    'f':self._ent1_Ry_cN,
                    'np':1,
                    'pre':False},
                'Uent1_cN':{
                    'f':self._Uent1_cN,
                    'np':2,
                    'pre':False},
                'UCC1':{
                    'f':self._UCC1,
                    'np':1,
                    'pre':False},
                'UCC2':{
                    'f':self._UCC2_full,
                    'np':3,
                    'pre':False} ,
                'UCC2s':{
                    'f':self._UCC2_1p,
                    'np':1},
                'UCC2c':{
                    'f':self._UCC2_con,
                    'np':1
                    }
                }
        self.para = QuantStore.parameters
        self.qs = QuantStore
        self.Nq = QuantStore.Nq
        self.q = QuantumRegister(self.Nq,name='q')
        self.c = ClassicalRegister(self.Nq,name='c')
        self.Ne = QuantStore.Ne
        if _name==False:
            self.qc = QuantumCircuit(self.q,self.c)
        else:
            self.qc = QuantumCircuit(self.q,self.c,name=_name)
        self.ent_p = self.ents[self.qs.ent_circ_p]['f']
        self.ent_Np = self.ents[self.qs.ent_circ_p]['np']
        self.ent_q = self.ents[self.qs.ent_circ_q]['f']
        self.ent_Nq = self.ents[self.qs.ent_circ_q]['np']
        self.cg = 0
        self.sg = 0
        self._gen_circuit()

    def _initialize(self):
        self.Ne_alp = self.qs.Ne_alp
        self.Ne_bet = self.qs.Ne-self.qs.Ne_alp
        if self.qs.init=='default':
            for i in range(0,self.Ne_alp):
                targ = self.qs.rdm_to_backend[self.qs.alpha['active'][i]]
                self.qc.x(self.q[targ])
            for i in range(0,self.Ne_bet):
                targ = self.qs.rdm_to_backend[self.qs.beta['active'][i]]
                self.qc.x(self.q[targ])
        self.sg+= self.Ne

    def _gen_circuit(self):
        self._initialize()
        h = 0 
        for d in range(0,self.qs.depth):
            for quad in (self.qs.quad_list):
                a = self.ent_Nq
                temp = self.para[h:h+a]
                self.ent_q(*temp,
                        i=quad[0],
                        j=quad[1],
                        k=quad[2],
                        l=quad[3])
                h+= self.ent_Nq
            for pair in self.qs.pair_list:
                a = self.ent_Np
                temp = self.para[h:h+a]
                self.ent_p(*temp,i=pair[0],k=pair[1])
                h+=self.ent_Np


    def _ent1_Ry_cN(self,phi,i,k,ddphi=False):
        if not ddphi:
            self.qc.cx(self.q[k],self.q[i])
            self.qc.x(self.q[k])
            self.qc.ry(phi/2,self.q[k])
            self.qc.cx(self.q[i],self.q[k])
            self.qc.ry(-phi/2,self.q[k])
            self.qc.cx(self.q[i],self.q[k])
            self.qc.x(self.q[k])
            self.qc.cx(self.q[k],self.q[i])
            #for s in range(i,k):
                #self.qc.cz(self.q[k],self.q[s])
                #self.qc.z(self.q[s])
            self.cg+= 4
            self.sg+= 4

    def _Uent1_cN(self,phi1,phi2,i,k):
        self.qc.cx(self.q[k],self.q[i])
        self.qc.x(self.q[k])
        self.qc.rz(phi1,self.q[k])
        self.qc.ry(-phi2,self.q[k])
        self.qc.cz(self.q[i],self.q[k])
        self.qc.ry(phi2,self.q[k])
        self.qc.rz(-phi1,self.q[k])
        self.qc.x(self.q[k])
        self.qc.cx(self.q[k],self.q[i])
        self.cg+= 3
        self.sg+= 4

    def _UCC1(self,phi,i,k):
        sequence = [['h','y'],
                ['y','h']]
        index = [i,k]
        for nt,term in enumerate(sequence):
            ind=0
            for item in term:
                if item=='h':
                    self.qc.h(self.q[index[ind]])
                elif item=='y':
                    self.qc.rx(-pi/2,self.q[index[ind]])
                ind+=1
                self.sg+=1
            for control in range(i,k):
                target = control+1
                self.qc.cx(self.q[control],self.q[target])
                self.cg+=1
            self.qc.rz(phi,self.q[k])
            for control in reversed(range(i,k)):
                target = control+1
                self.qc.cx(self.q[control],self.q[target])
                self.cg+=1
            ind = 0
            self.sg+=1
            for item in term:
                if item=='h':
                    self.qc.h(self.q[index[ind]])
                elif item=='y':
                    self.qc.rx(pi/2,self.q[index[ind]])
                ind+=1
                self.sg+=1

    def _UCC2_full(self,phi1,phi2,phi3,i,j,k,l):
        if phi1==0 and phi2==0 and phi3==0:
            pass
        else:
            sequence = [
                    ['h','h','h','y'],
                    ['h','h','y','h'],
                    ['h','y','h','h'],
                    ['h','y','y','y'],
                    ['y','h','h','h'],
                    ['y','h','y','y'],
                    ['y','y','h','y'],
                    ['y','y','y','h']
                ]
            var =  [
                    [+1,+1,-1],[+1,-1,+1],
                    [-1,+1,+1],[+1,+1,+1],
                    [-1,-1,-1],[+1,-1,-1],
                    [-1,+1,-1],[-1,-1,+1]]
            # seq 1
            index = [i,j,k,l]
            for nt,term in enumerate(sequence):
                theta = phi1*var[nt][0]+phi2*var[nt][1]+phi3*var[nt][2]
                ind=0
                for item in term:
                    if item=='h':
                        self.qc.h(self.q[index[ind]])
                    elif item=='y':
                        self.qc.rx(-pi/2,self.q[index[ind]])
                    ind+=1
                    self.sg+=1
                for control in range(i,l):
                    target = control+1
                    self.qc.cx(self.q[control],self.q[target])
                    self.cg+=1
                self.qc.rz(theta,self.q[l])
                self.sg+=1
                for control in reversed(range(i,l)):
                    target = control+1
                    self.qc.cx(self.q[control],self.q[target])
                    self.cg+=1
                ind =  0
                for item in term:
                    if item=='h':
                        self.qc.h(self.q[index[ind]])
                    elif item=='y':
                        self.qc.rx(pi/2,self.q[index[ind]])
                    self.sg+=1
                    ind+=1

    def _UCC2_1p(self,phi1,i,j,k,l):
        if phi1>-0.02 and phi1<0.02:
            pass
        else:
            sequence = [
                    ['h','h','h','y'],
                    ['h','h','y','h'],
                    ['h','y','h','h'],
                    ['h','y','y','y'],
                    ['y','h','h','h'],
                    ['y','h','y','y'],
                    ['y','y','h','y'],
                    ['y','y','y','h']
                ]
            var =  [
                    [+1,+1,-1],[+1,-1,+1],
                    [-1,+1,+1],[+1,+1,+1],
                    [-1,-1,-1],[+1,-1,-1],
                    [-1,+1,-1],[-1,-1,+1]]
            index = [i,j,k,l]
            for nt,term in enumerate(sequence):
                ind=0
                for item in term:
                    if item=='h':
                        self.qc.h(self.q[index[ind]])
                    elif item=='y':
                        self.qc.rx(-pi/2,self.q[index[ind]])
                    self.sg+=1
                    ind+=1
                for control in range(i,l):
                    target = control+1
                    self.qc.cx(self.q[control],self.q[target])
                    self.cg+=1
                self.qc.rz(phi1*var[nt][0],self.q[l])
                self.sg+=1
                for control in reversed(range(i,l)):
                    target = control+1
                    self.qc.cx(self.q[control],self.q[target])
                    self.cg+=1
                ind = 0 
                for item in term:
                    if item=='h':
                        self.qc.h(self.q[index[ind]])
                    elif item=='y':
                        self.qc.rx(pi/2,self.q[index[ind]])
                    self.sg+=1
                    ind+=1

    def _UCC2_con(self,phi1,i,j,k,l):
        if phi1>-0.02 and phi1<0.02:
            pass
        else:
            sequence = [
                    #['h','h','h','y'],
                    #['h','h','y','h'],
                    ['h','y','h','h'],
                    ['h','y','y','y'],
                    ['y','h','h','h'],
                    ['y','h','y','y']
                    #['y','y','h','y']
                    #['y','y','y','h']
                ]
            var =  [
                    #[+1,+1,-1],[+1,-1,+1],
                    [-1,+1,+1],[+1,+1,+1],
                    [-1,-1,-1],[+1,-1,-1]
                    #[-1,+1,-1]
                    #[-1,-1,+1]
                    ]
            index = [i,j,k,l]
            for nt,term in enumerate(sequence):
                ind=0
                for item in term:
                    if item=='h':
                        self.qc.h(self.q[index[ind]])
                    elif item=='y':
                        self.qc.rx(-pi/2,self.q[index[ind]])
                    self.sg+=1
                    ind+=1
                for control in range(i,l):
                    target = control+1
                    self.qc.cx(self.q[control],self.q[target])
                    self.cg+=1
                self.qc.rz(phi1*var[nt][0],self.q[l])
                self.sg+=1
                for control in reversed(range(i,l)):
                    target = control+1
                    self.qc.cx(self.q[control],self.q[target])
                    self.cg+=1
                ind = 0 
                for item in term:
                    if item=='h':
                        self.qc.h(self.q[index[ind]])
                    elif item=='y':
                        self.qc.rx(pi/2,self.q[index[ind]])
                    self.sg+=1
                    ind+=1
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
        elif self.qa=='4qtest':
            if order=='default':
                o = [0,1,0,2,0,3,1,2,1,3,2,3]
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
            self.qc.ry(self.p[6],self.q[o[6]])
            self.qc.ry(self.p[7],self.q[o[7]])
            self.qc.cx(self.q[o[6]],self.q[o[7]])
            self.qc.ry(self.p[8],self.q[o[8]])
            self.qc.ry(self.p[9],self.q[o[9]])
            self.qc.cx(self.q[o[8]],self.q[o[9]])
            self.qc.ry(self.p[10],self.q[o[10]])
            self.qc.ry(self.p[11],self.q[o[11]])
            self.qc.cx(self.q[o[10]],self.q[o[11]])
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

algorithm_tomography = {
        'affine_2p_curved_tenerife':{
            'tomo':'d1rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2]
            },
        'affine_2p_flat_tenerife':{
            'tomo':'1rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2]
            },
        'affine_2p_flatfish_tenerife':{
            'tomo':'1rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2]
            },
        '4qtest':{
            'tomo':'2rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2,3]
            },
        '3qtest':{
            'tomo':'2rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2]
            }
        }

