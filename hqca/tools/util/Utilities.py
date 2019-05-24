import pickle
import sys
import random
import numpy as np


class CircuitProperties:
    def __init__(self,filename):
        with open(filename,'rb') as fp:
            self.data=pickle.load(fp)
        self.coupling_map = self.data['config'].coupling_map
        self.qubits = self.data['properties'].qubits #list
        self.gates = self.data['properties'].gates
        self.Nq = len(self.qubits)
        self._neighbors()
        self._find_gates()
        self._find_qb()

    def more(self):
        self._more_neighbors()

    def _neighbors(self):
        self.neighbor={}
        for i in range(0,16):
            self.neighbor[i]=[]
        for pair in self.coupling_map:
            self.neighbor[pair[0]].append(pair[1])
            self.neighbor[pair[1]].append(pair[0])

    def _more_neighbors(self):
        self.two_steps = {}
        self.three_steps = {}
        for i in range(0,16):
            self.two_steps[i]=[]
            self.three_steps[i]=[]
        for k in self.neighbor.keys():
            for n1 in self.neighbor[k]:
                c1 = n1 ==k
                if not (c1):
                    for n2 in self.neighbor[n1]:
                        c2 = not n2==k
                        c3 = not n2 in self.neighbor[k]
                        c4 = not n2 in self.two_steps[k]
                        if c2 and c3 and c4:
                            self.two_steps[k].append(n2)
                            for n3 in self.neighbor[n2]:
                                c4 = not n3==k
                                c5 = not n3 in self.neighbor[k]
                                c6 = not n3 in self.two_steps[k]
                                c7 = not n3 in self.three_steps[k]
                                if c4 and c5 and c6 and c7:
                                    self.three_steps[k].append(n3)

    def _find_gates(self):
        self.g = {
                'cx':{},
                'u1':{},
                'u2':{},
                'u3':{},
                }
        for g in self.gates:
            if g.gate=='cx':
                t1,t2 = g.name[2:].split('_')
                tm1 = '{}-{}'.format(t1,t2)
                tm2 = '{}-{}'.format(t2,t1)
                self.g['cx'][tm1]=g.parameters[0].value
                self.g['cx'][tm2]=g.parameters[0].value
            elif g.gate in ['u1','u2','u3']:
                self.g[g.gate][g.qubits[0]]=g.parameters[0].value

    def _find_qb(self):
        self.qb = {}
        for n,q in enumerate(self.qubits):
            self.qb[n]={
                    'T1':q[0].value,
                    'T2':q[1].value,
                    'ro':q[3].value,
                    }

    def load_qasm(self,qasm):
        self.qasm = []
        with open(qasm,'r') as fp:
            for n,line in enumerate(fp):
                if n in [0,1,2,3]:
                    continue
                temp = line.split(' ')
                if temp[0]=='cx':
                    name = 'cx'
                    q1,q2 = temp[1][:-2].split(',')
                    q1 = q1[2:-1]
                    q2 = q2[2:-1]
                    q = '{}-{}'.format(q1,q2)
                elif temp[0][0:2]=='u2':
                    name = 'u2'
                    q = temp[-1][:-2]
                elif temp[0][0:2]=='u3':
                    name = 'u3'
                    q = temp[-1][:-2]
                elif temp[0][0:2]=='u1':
                    name = 'u1'
                    q = temp[-1][:-2]
                elif temp[0][0:2]=='me':
                    name = 'ro'
                    q = (temp[1])
                else:
                    continue
                if not name=='cx':
                    q = int(q[2:-1])
                self.qasm.append([name,q])

def parameter_transfer(current,switch):
    i = random.randint(0,len(current)-1)
    #if not switch[i] in current:
    current[i]=switch[i]
    return current


def no_parameter_transfer(current,switch):
    return current

def full_mutate_sequence(current,leader,mutant=[0.8,0.1,0.1],Circuit=None):
    temp = []
    for i in range(len(current)):
        r = random.random()
        a,b,c = mutant[0],mutant[0]+mutant[1],mutant[0]+mutant[1]+mutant[2]
        if r<=a:
            temp.append(current[i])
        elif r>a and r<=b:
            temp.append(leader[i])
        elif r>b and r<=c:
            temp.append(random.randint(0,Circuit.Nq-1))
    return temp


def generate_really_random(strand=None,N=4,Circuit=None):
    return random.sample(range(Circuit.Nq),N)

def generate_random_sequence(strand=None,N=4,Circuit=None):
    done = False
    if strand==None:
        strand=[]
    it=0
    start = True
    while not done:
        if it%100==0:
            strand=[]
        if len(strand)==0:
            strand.append(random.choice(range(Circuit.Nq)))
        if not start:
            neighbor = Circuit.neighbor[strand[-1]] #list
        else:
            neighbor = Circuit.neighbor[strand[0]] #list
        seq = random.sample(neighbor,len(neighbor))
        for n,s in enumerate(seq):
            if not (s in strand):
                if start:
                    strand.insert(0,s)
                else:
                    strand.append(s)
                break
            if n==len(seq)-1:
                if start:
                    start=False
                else:
                    start=True
                    del strand[-1]
        if len(strand)==N:
            done=True
        else:
            pass
        it+=1 
    return strand
        


def mutate_sequence(seq,target,mutant=None,**kw):
    '''
    huh?
    '''
    Ns = len(seq)
    r = random.randint(0,Ns)
    s = random.randint(r,Ns)
    return generate_random_sequence(seq[r:s],**kw)

def fitness_qubits(qubits,Circuit,verbose=False):
    '''
    input is a list of target qubits and the circuit properties

    assumption is that all qubits are connected... :( that is the problem
    only looks at readout and connectivity errors
    '''
    Nq = len(qubits)
    fro = 0
    fcx = 1
    for i in range(Nq-1):
        temp ='{}-{}'.format(str(qubits[i]),str(qubits[i+1]))
        fcx*= (1-Circuit.g['cx'][temp])
    for n in qubits:
        fro +=  (1-Circuit.qb[n]['ro'])
    fro*= (1/Nq)
    if verbose:
        print('Measure: {}, CNOT: {}'.format(fro,fcx))
    return 1-fcx*fro

def fitness_qasm(qubits,Circuit,verbose=False):
    # note, you should set the circuit qubit 
    q2q = {i:qubits[i] for i in range(len(qubits))} #qasm to qubits
    Nq = len(qubits)
    f_c = np.ones(Nq)
    for ins,n in Circuit.qasm:
        if ins=='ro':
            f_c[n] = f_c[n]*(1-Circuit.qb[q2q[n]]['ro'])
            #f_ro+= (1-Circuit.qb[q2q[n]]['ro'])
        elif ins=='cx':
            a,b = n.split('-')
            a,b = int(a),int(b)
            #f_c*= (1-Circuit.g[ins][n])
            f_c[a] = f_c[a]*(1-Circuit.g[ins][n])
            f_c[b] = f_c[b]*(1-Circuit.g[ins][n])
        else:
            #f_c*=  (1-Circuit.g[ins][q2q[n]])
            f_c[n] = f_c[n]*(1-Circuit.g[ins][q2q[n]])
    #f_ro*= (1/Nq)
    f_c = np.average(f_c)
    if verbose:
        print('Measure: {}, CNOT: {}, Metric: {}'.format(f_ro,f_c,1-f_c))
    return 1-f_c

def fitness_random_qasm(qubits,Circuit,verbose=False):
    unique = []
    for i in qubits:
        if not i in unique:
            unique.append(i)
    if not (len(unique)==len(qubits)):
        return 1
    f_l = 1
    for i in range(len(qubits)-1):
        q1 = qubits[i]
        q2 = qubits[i+1]
        if q2 in Circuit.neighbor[q1]:
            pass
        elif q2 in Circuit.two_steps[q1]:
            f_l*= (1/2)
        elif q2 in Circuit.three_steps[q1]:
            f_l*= (1/3)
        else:
            f_l=0 #f_l is low number, we want high fitness for 0!
    f_r = fitness_qasm(qubits,Circuit,verbose=verbose)
    return 0.5*(1-f_l)+f_r*0.5
    
    # check qubits are real




