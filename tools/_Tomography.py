'''
/tools/_Tomography.py

File needed for executing and processing the tomography. 
'''

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

def combine_dictionary(one,two):
    for key,val in two.items():
        try:
            one[key] = int(one[key]) + int(val)
        except:
            one[key] = int(val)
    return one

class Process:
    def __init__(self,
            output,
            tomo_rdm,
            tomo_basis,
            tomo_extra,
            qa_fermion,
            qb2so,
            actqb2beqb='default',
            **kw
            ):
        # First, want to combine results
        self.data = {'ii':{},'ij':{},'ijkl':{}}
        self.tomo_rdm = tomo_rdm
        self.tomo_basis = tomo_basis
        self.tomo_extra = tomo_extra
        self.fermi = qa_fermion
        self.add_data(output)
        self.occ_qb = []
        self.kw = kw
        for v,k in qb2so:
            self.occ_qb.append(v)
        self.Nq_act = len(self.occ_qb)
        if actqb2beqb=='default':
            self.a2b={i:i for i in range(0,self.Nq_act)}
        else:
            self.a2b= actqb2beqb


    def add_data(self,output):
        i=0
        for name,counts in output:
            if i==0:
                self.Nq_tot = len(list(counts.items())[0])
            prbdis = self.proc_counts(counts)
            if name[0]=='ii':
                self.data['ii']['counts']=counts
                self.data['ii']['pd']=prbdis
            elif name[0]=='ij':
                self.data['ij'][i]={}
                self.data['ij'][i]['qb']=name[1:]
                self.data['ij'][i]['counts']=counts
                self.data['ij'][i]['pd']=prbdis
                i+=1
            else:
                pass

    def proc_counts(self,counts):
        Nc = 0
        r  = zeros(self.Nq_tot)
        for qb_state,outcome in counts.items():
            for i in range(0,self.Nq_tot):
                if qb_state[i]=='1':
                    r[self.Nq-1-i]+=outcome
            Nc += outcome
        r = r*(1/Nc)
        return r

    def build_rdm(self):
        if self.fermi=='direct':
            self._build_direct_rdm()
        elif self.fermi=='compact':
            self._build_compact_rdm(**self.kw)


    def _build_direct_rdm(self):
        if self.tomo_basis=='bch':
            self.rdm = zeros((self.Nq_act,self.Nq_act))
            for actqb in self.occ_qb:
                ind = self.a2b[actqb]
                temp = self.data['ii']['pd'][ind]
                self.rdm[actqb,actqb]=temp
            for item in self.data['ij']:
                for pair in item['qb']:
                    i1,i2 = int(pair[0]),int(pair[1])
                    i1,i2 = self.a2b[i1],self.a2b[i2]
                    val = item['pd'][i1]
                    val -= 0.5*(self.rdm[i1,i1]+self.rdm[i1,i1])
                    self.rdm[i1,i2]=val
                    self.rdm[i2,i1]=val


    def _build_compact_rdm(self,**kwargs):
        if self.tomo_rdm=='1rdm':
            try:
                self.data['ij']['counts']
                use_err=True
            except KeyError:
                self.data['ij']={'counts':None}
                use_err=False
            for k,v in kwargs.items():
                print(k,v)
            self.on, self.rdm = fx.counts_to_1rdm(
                self.data['ii']['counts'],
                self.data['ij']['counts'],
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


efficient_qubit_tomo_pairs = {
        3:[
            ['01'],['02'],['12']],
        4:[
            ['01','23'],['12','03'],
            ['02','13']],
        5:[
            ['01','23'],['04','12'],
            ['02','34'],['13','24'],
            ['03','14']],
        6:[
            ['01','23','45'],
            ['02','14','35'],
            ['13','04','25'],
            ['03','24','15'],
            ['12','34','05']],
        7:[
            ['01','23','46'],
            ['02','13','56'],
            ['03','14','25'],
            ['04','15','26'],
            ['05','16','34'],
            ['06','24','35'],
            ['12','36','45']],
        8:[
            ['01','24','35','67'],
            ['02','14','36','57'],
            ['03','15','26','47'],
            ['04','12','56','37'],
            ['05','13','27','46'],
            ['06','17','23','45'],
            ['07','16','25','34']]

        }

