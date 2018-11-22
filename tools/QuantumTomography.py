from hqca.tools.QuantumAlgorithms import GenerateCompactCircuit
from hqca.tools.QuantumAlgorithms import GenerateDirectCircuit
from hqca.tools.QuantumAlgorithms import algorithm_tomography
from hqca.tools import RDMFunctions as rdmf
from hqca.tools import Functions as fx
from hqca.tools import IBM_check

import sys,time
import timeit
import traceback
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit import QISKitError
import qiskit
from numpy import log10,floor
from numpy import zeros,multiply,real

SIM_EXEC = ('/usr/local/lib/python3.5/dist-packages'
            ' /qiskit/backends/qasm_simulator_cpp')


class GenerateDirectTomography:
    '''
    Class for carrying out direct measurements

    '''




class GenerateCompactTomography:
    ''' Class for carrying out compact measurements.'''

    def __init__(
            self,
            qc_backend,
            qc_provider,
            algorithm,
            tomography='default',
            _num_runs=2,
            _num_shots=1024,
            verbose=False,
            **kwargs):
        kwargs['verbose']=verbose
        tic = timeit.default_timer()
        self.Nq = algorithm_tomography[algorithm]['Nq']
        self.qb_orbs = algorithm_tomography[algorithm]['qb_to_orb']
        self.No = len(self.qb_orbs)
        self.backend = qc_backend
        self.verbose = verbose
        self.qc_kwargs = kwargs
        self.qc_kwargs['algorithm']=algorithm
        self.tomo_get_backend(
                qc_provider,
                self.backend)
        self.circuit_list = []
        self.circuits = []
        self.Q = []
        self.counts = {}
        self._Ns = _num_shots
        self._Nr = _num_runs
        t1 = timeit.default_timer()
        if tomography=='default':
            tomo = algorithm_tomography[algorithm]['tomo']
        else:
            tomo = tomography
        self.tomo = tomo
        if tomo=='d1rdm':
            self.tomo_d1rdm()
        elif tomo=='1rdm':
            self.tomo_1rdm()
        elif tomo=='2rdm':
            self.tomo_2rdm()
        toc = timeit.default_timer()
        if verbose:
            print('Time to get backend : {}'.format(t1-tic))
            print('Time to run tomography : {}'.format(toc-t1))
            print('Total time: {}'.format(toc-tic))


    def tomo_get_backend(self,
            provider,
            backend
            ):
        if provider=='Aer':
            self.prov=Aer
        elif provider=='IBMQ':
            self.prov=IBMQ
        try:
            self.b = self.prov.get_backend(backend)
        except Exception:
            pass


    def tomo_get_counts(self,qo):
        '''
        Either performs measurement on normal quantum computation, or takes the
        unitary simualtor and returns it in a counted format.

        Regardless, this is the final step. It takes the compiled quantum object
        and executes the counts. 
        '''
        if self.backend=='local_unitary_simulator':
            qr = self.b.run(qo,timeout=6000)
            for circuit in self.circuit_list:
                U = qr.result().get_data(circuit)['unitary']
                self.counts[circuit] = fx.UnitaryToCounts(U)
        elif self.backend=='qasm_simulator_cpp':
            pass
        else:
            try:
                #tic = timeit.default_timer()
                job = self.b.run(qo)
                #time.sleep(5)
                #while job.running:
                ##    time.sleep(5)
                #    toc = timeit.default_timer()
                #    print('Still runnning...{:8.1f} s'.format(toc-tic))
                #print(self.b,toc-tic)
                for circuit in self.circuit_list:
                    self.counts[circuit] = job.result().get_counts(circuit)
            except Exception as e:
                print(e)
        if self.verbose:
            print('Circuit counts:')
            for k,v in self.counts.items():
                    print('  {}:{}'.format(k,v))

    def get_qb_parity(self,
            qb_to_orb='default'
            ):
        '''
        Given a qubit to fermion mapping, which is specified below, returns the
        parity mapping for orbitals.

        qb_to_orb should be the array of qubits which map to the sequence of
        orbitals, from high to low. I.e., for a mapping where you have qubits
        0,1,and 2 mapping to 0/5, 1/4, and 2/3 respectively, specify qb_to_orb
        as [0,1,2].
        '''
        ind = 0
        self.qb_sign = {}
        for item in reversed(self.qb_orbs):
            self.qb_sign[item]=(-1)**ind
            ind+=1

    def tomo_d1rdm(self):
        self.type = '1RDM'
        self.get_qb_parity()
        for i in range(0,self._Nr):
            self.Q.append(
                    GenerateCompactCircuit(**self.qc_kwargs,_name='ii{}'.format(i))
                    )
            self.Q[i].qc.measure(self.Q[i].q,self.Q[i].c)
            self.circuits.append(self.Q[i].qc)
            self.circuit_list.append('ii{}'.format(i))
        self.qo = qiskit.compile(
                self.circuits,
                shots=self._Ns,
                backend=self.b
                )
        self.tomo_get_counts(self.qo)

    def tomo_1rdm(self):
        self.type='1RDM'
        self.get_qb_parity()
        for i in range(0,self._Nr):
            self.Q.append(
                    GenerateCircuit(**self.qc_kwargs,_name='ii{}'.format(i))
                    )
            self.Q[i].qc.measure(self.Q[i].q,self.Q[i].c)
            self.circuit_list.append('ii{}'.format(i))
            self.circuits.append(self.Q[i].qc)
        for i in range(self._Nr,2*self._Nr):
            self.Q.append(
                    GenerateCircuit(**self.qc_kwargs,_name='ij{}'.format(
                        i-self._Nr)
                        )
                    )
            for j in self.qb_orbs:
                if self.qb_sign[j]==-1:
                    self.Q[i].qc.z(self.Q[i].q[j])
                    self.Q[i].qc.h(self.Q[i].q[j])
                elif self.qb_sign[j]==1:
                    self.Q[i].qc.h(self.Q[i].q[j])
                    self.Q[i].qc.z(self.Q[i].q[j])
                else:
                    sys.exit('Error in performing 1RDM tomography.')
            self.Q[i].qc.measure(self.Q[i].q,self.Q[i].c)
            self.circuit_list.append('ij{}'.format(i-self._Nr))
            self.circuits.append(self.Q[i].qc)
        self.qo = qiskit.compile(
                self.circuits,
                shots=self._Ns,
                backend=self.b
                )
        self.tomo_get_counts(self.qo)


    def tomo_2rdm(self):
        self.type='2RDM'
        self.get_qb_parity()
        temp_qb_set = self.qb_orbs.copy()
        self.pairs = gen_2rdm_pairs(self.No)
        cc = -1
        for i in range(0,self._Nr): #note, only 1 place
            for k,p in self.pairs.items():

                #
                # Starting with unique iklj elements
                #
                self.Q.append(
                        GenerateCircuit(
                            **self.qc_kwargs,
                            _name='iklj{}{}'.format(
                                k,i
                                )
                            )
                        )
                cc+=1 
                pairs = p.split(',')
                Npair = len(pairs)
                for j in range(0,Npair):
                    a,b = int(pairs[j][0]),int(pairs[j][1])
                    sa,sb = self.qb_sign[a],self.qb_sign[b]

                    self.Q[cc].qc.cx(self.Q[cc].q[a],self.Q[cc].q[b])
                    if sa*sb==1:
                        self.Q[cc].qc.h(self.Q[cc].q[a])
                        self.Q[cc].qc.z(self.Q[cc].q[a])
                    else:
                        self.Q[cc].qc.z(self.Q[cc].q[a])
                        self.Q[cc].qc.h(self.Q[cc].q[a])
                    self.Q[cc].qc.cx(self.Q[cc].q[a],self.Q[cc].q[b])
                self.Q[cc].qc.measure(self.Q[cc].q,self.Q[cc].c)

                self.circuit_list.append('iklj{}{}'.format(
                                k,i
                                ))
                self.circuits.append(self.Q[cc].qc)

                #
                # Now adding the rotated off-pieces, ikli
                #

                self.Q.append(
                        GenerateCircuit(
                            **self.qc_kwargs,
                            _name='ikli{}{}'.format(
                                k,i
                                )
                            )
                        )
                cc+=1 
                temp = temp_qb_set.copy()
                for j in range(0,Npair):
                    a,b = int(pairs[j][0]),int(pairs[j][1])
                    temp.remove(a)
                    temp.remove(b)
                    sa,sb = self.qb_sign[a],self.qb_sign[b]
                    self.Q[cc].qc.cz(self.Q[cc].q[a],self.Q[cc].q[b])
                    if sb==1: #NOTE THIS IS REVERSED!!!! Important. 
                        self.Q[cc].qc.z(self.Q[cc].q[b])
                        self.Q[cc].qc.h(self.Q[cc].q[b])
                    else:
                        self.Q[cc].qc.h(self.Q[cc].q[b])
                        self.Q[cc].qc.z(self.Q[cc].q[b])
                    self.Q[cc].qc.cz(self.Q[cc].q[a],self.Q[cc].q[b])
                self.Q[cc].qc.measure(self.Q[cc].q,self.Q[cc].c)

                self.circuit_list.append('ikli{}{}'.format(
                                k,i
                                ))
                self.circuits.append(self.Q[cc].qc)

                #
                # Same as above, but now in reverse, stored as ikkj
                #

                self.Q.append(
                        GenerateCircuit(
                            **self.qc_kwargs,
                            _name='ikkj{}{}'.format(
                                k,i
                                )
                            )
                        )
                cc+=1 
                temp = temp_qb_set.copy()
                for j in range(0,Npair):
                    a,b = int(pairs[j][0]),int(pairs[j][1])
                    temp.remove(a)
                    temp.remove(b)
                    sa,sb = self.qb_sign[a],self.qb_sign[b]
                    self.Q[cc].qc.cz(self.Q[cc].q[b],self.Q[cc].q[a])
                    if sa==1:
                        self.Q[cc].qc.h(self.Q[cc].q[a])
                        self.Q[cc].qc.z(self.Q[cc].q[a])
                    else:
                        self.Q[cc].qc.z(self.Q[cc].q[a])
                        self.Q[cc].qc.h(self.Q[cc].q[a])
                    self.Q[cc].qc.cz(self.Q[cc].q[b],self.Q[cc].q[a])
                self.Q[cc].qc.measure(self.Q[cc].q,self.Q[cc].c)
                self.circuit_list.append('ikkj{}{}'.format(
                                k,i
                                ))
                self.circuits.append(self.Q[cc].qc)

                #
                # Finally, the identity elements. 
                #

                self.Q.append(
                        GenerateCircuit(
                            **self.qc_kwargs,
                            _name='ikki{}{}'.format(
                                k,i
                                )
                            )
                        )
                cc+=1 
                temp = temp_qb_set.copy()
                for j in range(0,Npair):
                    a,b = int(pairs[j][0]),int(pairs[j][1])
                    temp.remove(a)
                    temp.remove(b)
                    sa,sb = self.qb_sign[b],self.qb_sign[a]
                    #if sa*sb==1:
                    #    self.Q[cc].qc.cx(self.Q[cc].q[a],self.Q[cc].q[b])
                    #else:
                    #    self.Q[cc].qc.z(self.Q[cc].q[a])
                    self.Q[cc].qc.cx(self.Q[cc].q[a],self.Q[cc].q[b])
                self.Q[cc].qc.measure(self.Q[cc].q,self.Q[cc].c)
                self.circuit_list.append('ikki{}{}'.format(
                                k,i
                                ))
                self.circuits.append(self.Q[cc].qc)

            self.Q.append(
                   GenerateCircuit(
                       **self.qc_kwargs,
                       _name='ii{}'.format(
                           i
                           )
                       )
                   )
            cc+=1 
            self.Q[cc].qc.measure(self.Q[cc].q,self.Q[cc].c)
            self.circuit_list.append('ii{}'.format(
                            i
                            ))
            self.circuits.append(self.Q[cc].qc)


            self.Q.append(
                    GenerateCircuit(
                        **self.qc_kwargs,
                        _name='ij{}'.format(
                            i
                            )
                        )
                    )
            cc+=1 
            for j in self.qb_orbs:
                if self.qb_sign[j]==-1:
                    self.Q[cc].qc.z(self.Q[cc].q[j])
                    self.Q[cc].qc.h(self.Q[cc].q[j])
                elif self.qb_sign[j]==1:
                    self.Q[cc].qc.h(self.Q[cc].q[j])
                    self.Q[cc].qc.z(self.Q[cc].q[j])
            self.Q[cc].qc.measure(self.Q[cc].q,self.Q[cc].c)
            self.circuit_list.append('ij{}'.format(i))
            self.circuits.append(self.Q[cc].qc)

                #
                # And thats it! :) 
                # 

        self.qo = qiskit.compile(
                self.circuits,
                shots=self._Ns,
                backend=self.b
                )
        self.tomo_get_counts(self.qo)



tomo_circuits = {
        'd1rdm':['ii'],
        '1rdm' :['ii','ij']
        #'2rdm' :['iklj','ikli','kijk','ikki']
        }

def gen_2rdm_pairs(Nq):
    if Nq==3:
        pair_list = ['01','02','12']
    elif Nq==4:
        pair_list = ['01,23','02,13','03,12']
    elif Nq==5:
        pair_list = ['01,23','12,34','03,24','02,14','04,13']
    elif Nq==6:
        pair_list = ['01,23,45','05,12,34','02,14,35','04,13,25','03,15,24']
    else:
        print('Not a good qubit architecture. Not supported here.')
        print('Goodbye!')
        sys.exit()
    keys = ['{:0{}b}f'.format(
        i,
        int(
            floor(
                log10(len(pair_list)
                    )
                )+2
            )
        ) for i in range(0,len(pair_list))
        ]
    circuit_pair_map = {}
    for i in keys:
        circuit_pair_map[i]=pair_list.pop()
    print('Mapping: {}'.format(circuit_pair_map))
    return circuit_pair_map



def combine_dictionary(one,two):
    for key,val in two.items():
        try:
            one[key] = int(one[key]) + int(val)
        except:
            one[key] = int(val)
    return one


class ProcessToRDM:

    def __init__(
            self,
            combine=True
            ):
        # First, want to combine results
        self.data = {}

    def add_data(self,QuantTomoInst):
        self.type = QuantTomoInst.type
        self.circuits = QuantTomoInst.circuit_list
        self.runs = QuantTomoInst._Nr
        self.Nels = QuantTomoInst.No  # note No is spatial orbitals
        self.Norb = QuantTomoInst.No*2
        self.qb_sign = QuantTomoInst.qb_sign
        self.Nq = QuantTomoInst.Nq
        self.qb_orbs = QuantTomoInst.qb_orbs
        self.verbose = QuantTomoInst.verbose
        if self.type=='1RDM':
            for i in tomo_circuits[QuantTomoInst.tomo]:
                self.data[i]={}
            for circuit in self.circuits:
                self.data[circuit[:-1]] = combine_dictionary(
                        self.data[circuit[:-1]],
                        QuantTomoInst.counts[circuit]
                        )
        elif self.type=='2RDM':
            self.pair_map = QuantTomoInst.pairs
            for circuit in self.circuits:
                try:
                    self.data[circuit[:-1]] = combine_dictionary(
                        self.data[circuit[:-1]],
                        QuantTomoInst.counts[circuit]
                        )
                except KeyError:
                    self.data[circuit[:-1]] = QuantTomoInst.counts[circuit]
        if QuantTomoInst.verbose:
            print('Compiled circuit:')
            for k,v in self.data.items():
                print('  {}:{}'.format(k,v))

    def add_instance(self, QuantTomoInst):
        try:
            self.type
            self.circuits
            for circuit in self.circuits:
                self.data[circuit] = combine_dictionary(
                        self.data[circuit],
                        self.QuantTomoInst[circuit]
                        )
        except AttributeError: #i.e. on first instance, assign
            self.type = QuantTomoInst.type
            self.circuits = QuantTomoInst.circuit_list
            self.data = QuantTomoInst.counts
            #print(QuantTomoInst.counts)
        except Exception:
            traceback.print_exc()

    def build_rdm(self,**kwargs):
        if self.type=='1RDM':
            try:
                self.data['ij']
                use_err=True
            except KeyError:
                self.data['ij']={}
                use_err=False
            self.on, self.rdm1 = fx.counts_to_1rdm(
                self.data['ii'],
                self.data['ij'],
                use_err=use_err,
                **kwargs
                )
        elif self.type=='2RDM':
            self.rdm2 = self.assemble_2rdm()
            self.rdm1 = rdmf.check_2rdm(self.rdm2)
            self.rdm1 = real(self.rdm1)
            self.rdm2trace = rdmf.trace_2rdm(self.rdm2)
            temp = fx.get_trace(self.Nq,self.qb_orbs)
            #print(fx.filt(self.data['ij'],temp))
            self.rdm1c, self.on, self.rdm1ev = fx.construct_rdm(
                    fx.rdm(
                        fx.filt(self.data['ii'],temp)
                        ),
                    fx.rdm(
                        fx.filt(self.data['ij'],temp)
                        )
                    )
            if self.verbose:
                print('Trace of 2-RDM: {}'.format(self.rdm2trace))

    def assemble_2rdm(self):
        '''
        Method (somewhat general, as much as it can be)
        to generate a 2RDM from what is given to it....
        we will see how this goes. 
        '''
        def measure(data,reverse=True):
            '''
            To measure a counts instance. If reverse is true, then it will
            output in the reversed order. Why is this important? Not sure. It
            has to be reversed at some point.
            '''
            unit = list(data.keys())
            total_count=0
            r = zeros(len(unit[0]))
            for qubit, count in data.items():
                total_count += count
                n_qb = len(qubit)
                for i in range(0,n_qb):
                    if qubit[n_qb-1-i]=='0':
                        if reverse:
                            r[i]+= count
                        else:
                            r[n_qb-1-i]+= count
            r = multiply(r,total_count**-1)
            return r # len in how many qubits

        def rdm_update(
                rdm,element,
                i,j,k,l,
                t1,t2,t3,t4,
                s1,s2,
                spin_restrict=True,
                e1='alpha',
                e2='alpha'
                ):
            if element=='iklj':
                rdm[i,k,l,j]+= t1*s1*s2
                rdm[k,i,j,l]+= t1*s1*s2
                rdm[j,l,k,i]+= t1*s1*s2
                rdm[l,j,i,k]+= t1*s1*s2
                rdm[i,l,k,j]+= t2*s1*s2
                rdm[l,i,j,k]+= t2*s1*s2
                rdm[j,k,l,i]+= t2*s1*s2
                rdm[k,j,i,l]+= t2*s1*s2
                if e1==e2:
                    rdm[k,i,l,j]-= t1*s1*s2
                    rdm[i,k,j,l]-= t1*s1*s2
                    rdm[l,j,k,i]-= t1*s1*s2
                    rdm[j,l,i,k]-= t1*s1*s2
                    rdm[l,i,k,j]-= t2*s1*s2
                    rdm[i,l,j,k]-= t2*s1*s2
                    rdm[k,j,l,i]-= t2*s1*s2
                    rdm[j,k,i,l]-= t2*s1*s2
            elif element=='ikli':
                rdm[i,k,l,i]+= t1*s2
                rdm[k,i,i,l]+= t1*s2
                rdm[i,l,k,i]+= t1*s2
                rdm[l,i,i,k]+= t1*s2
                rdm[j,k,l,j]+= t2*s2
                rdm[k,j,j,l]+= t2*s2
                rdm[j,l,k,j]+= t2*s2
                rdm[l,j,j,k]+= t2*s2
                if e1==e2:
                    rdm[k,i,l,i]-= t1*s2
                    rdm[i,k,i,l]-= t1*s2
                    rdm[l,i,k,i]-= t1*s2
                    rdm[i,l,i,k]-= t1*s2
                    rdm[j,k,j,l]-= t2*s2
                    rdm[k,j,l,j]-= t2*s2
                    rdm[j,l,j,k]-= t2*s2
                    rdm[l,j,k,j]-= t2*s2
            elif element=='ikki':
                rdm[j,l,l,j]+= t4
                rdm[l,j,j,l]+= t4
                rdm[j,k,k,j]+= t3
                rdm[k,j,j,k]+= t3
                rdm[i,l,l,i]+= t2
                rdm[l,i,i,l]+= t2
                rdm[k,i,i,k]+= t1
                rdm[i,k,k,i]+= t1
                if e1==e2:
                    rdm[j,l,j,l]-= t4
                    rdm[l,j,l,j]-= t4
                    rdm[j,k,j,k]-= t3
                    rdm[k,j,k,j]-= t3
                    rdm[i,l,i,l]-= t2
                    rdm[l,i,l,i]-= t2
                    rdm[k,i,k,i]-= t1
                    rdm[i,k,i,k]-= t1
            return rdm

        rdm2 = zeros((
            self.Norb,
            self.Norb,
            self.Norb,
            self.Norb)
            )
        temp_ij_rot = self.qb_orbs.copy()
        temp_ii_rot = self.qb_orbs.copy()
        #for k,v in self.data.items():
        #    print(k)
        for key,v in self.data.items():
            if key[0:4]=='iklj':
                pairs = self.pair_map[key[4:]]
                pair_list = pairs.split(',')
                temp = measure(v)
                for item in pair_list:
                    q1,q2 = int(item[0]),int(item[1])
                    alp1 = 1 - 2*temp[q1] # sum of terms 
                    alp2 = 1 - 2*temp[q2] # difference of terms 
                    temp1 = 0.25*(alp1+alp2)
                    temp2 = 0.25*(alp1-alp2)
                    i = self.Norb-1-q1
                    j = q1
                    k = self.Norb-1-q2
                    l = q2
                    s1 = self.qb_sign[q1]
                    s2 = self.qb_sign[q2]
                    rdm2 = rdm_update(
                        rdm2,'iklj',
                        i,j,k,l,
                        temp1,temp2,0,0,
                        s1,s2
                        )
            elif key[0:4]=='ikli' or key[0:4]=='ikkj':
                # akin to..something. 
                pairs = self.pair_map[key[4:]]
                pair_list = pairs.split(',')
                temp = measure(v)
                for item in pair_list:
                    q1,q2 = int(item[0]),int(item[1])
                    if key[0:4]=='ikkj':
                        q1,q2 = q2,q1
                    bet2 = 1 - 2*temp[q2] # difference of terms 

                    temp1 = -0.25*(+bet2)
                    temp2 = -0.25*(-bet2)

                    i,j = self.Norb-1-q1,q1
                    k,l = self.Norb-1-q2,q2
                    s1 = self.qb_sign[q1]
                    s2 = self.qb_sign[q2]
                    #print(item,temp1,temp2,q1,q2,i,k,l,j,s1,s2)
                    rdm2 = rdm_update(
                        rdm2,'ikli',
                        i,j,k,l,
                        temp1,temp2,0,0,
                        s1,s2
                        )
            elif key[0:4]=='ikki':
                pairs = self.pair_map[key[4:]]
                pair_list = pairs.split(',')
                temp = measure(v)
                for item in pair_list:
                    q1,q2 = int(item[0]),int(item[1])
    
                    i,j = self.Norb-1-q1,q1
                    k,l = self.Norb-1-q2,q2
                    s1 = 1#self.qb_sign[q1]
                    s2 = 1#self.qb_sign[q2]
    
                    m_q1 = temp[q1]
                    m_q2c= temp[q2]
                    
                    temp1 = 0.5*(- m_q1 + m_q2c) #ikki, delta
                    temp2 = 0.5*(- m_q1 - m_q2c) #illi, gamma
                    temp3 = 0.5*(+ m_q1 - m_q2c) #jkkj, beta
                    temp4 = 0.5*(+ m_q1 + m_q2c) #jllj, alpha

                    rdm2 = rdm_update(
                        rdm2,'ikki',
                        i,j,k,l,
                        temp1,temp2,temp3,temp4,
                        s1,s2
                        )
            elif key[0:2]=='ii':
                temp = measure(v)
                for k2,v2 in self.pair_map.items():
                    pair_list = v2.split(',')
                    for item in pair_list:
                        #print(item)
                        q1,q2 = int(item[0]),int(item[1])
                        i,j = self.Norb-1-q1,q1
                        k,l = self.Norb-1-q2,q2
                        s1 = self.qb_sign[q1]
                        s2 = self.qb_sign[q2]
                        m_q2 = temp[q2]
                        temp1 = 0.5*(+ 1 - m_q2) #ikki, delta
                        temp2 = 0.5*(+ 1 + m_q2) #illi, gamma
                        temp3 = 0.5*(+ 1 - m_q2) #jkkj, beta
                        temp4 = 0.5*(- 1 + m_q2) #jllj, alpha
                        rdm2 = rdm_update(
                            rdm2,'ikki',
                            i,j,k,l,
                            temp1,temp2,temp3,temp4,
                            s1,s2
                            )
            elif key[0:2]=='ij':
                temp = measure(v)
                for k2,v2 in self.pair_map.items():
                    pair_list = v2.split(',')
                    for item in pair_list:
                        q1,q2 = int(item[0]),int(item[1])
                        i,j = self.Norb-1-q1,q1
                        k,l = self.Norb-1-q2,q2
                        s1 = self.qb_sign[q1]
                        s2 = self.qb_sign[q2]
                        bet1 = -0.25*(1-2*temp[q2])
                        bet2 = -0.25*(1-2*temp[q1])
                        #print(item,bet1,bet2,q1,q2,i,k,l,j,s1,s2)
                        rdm2 = rdm_update(
                            rdm2,'ikli',
                            i,j,k,l,
                            bet1,bet1,0,0,
                            s1,s2
                            )
                        rdm2 = rdm_update(
                            rdm2,'ikli',
                            k,l,i,j,
                            bet2,bet2,0,0,
                            s2,s1
                            )
        if self.verbose:
            print('Done with 2-RDM! yay! whew.')
        return rdm2

