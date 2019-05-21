'''
/tools/_Tomography.py

File needed for executing and processing the tomography. 
'''

from hqca.tools.QuantumAlgorithms import GenerateCompactCircuit
from hqca.tools.QuantumAlgorithms import GenerateDirectCircuit
from hqca.tools.QuantumAlgorithms import algorithm_tomography
from hqca.tools import RDMFunctions as rdmf
from hqca.tools import Functions as fx
from functools import reduce
import sys,time
import timeit
import traceback
import qiskit
from numpy import log10,floor,complex_
from numpy import log10,floor
from numpy import zeros,multiply,real

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
            QuantStore,
            ):
        # First, want to combine results
        self.data = {
                'ii':{},'ij':[],'ijkl':[],
                'ijR':[],'ijI':[],'ij':[],
                'sign':[]
                }
        self.tomo_rdm = QuantStore.tomo_rdm
        self.tomo_basis=QuantStore.tomo_bas
        self.qs = QuantStore
        self.Nq_act = QuantStore.Nq
        self.Nq_tot = QuantStore.Nq_tot
        self.add_data(output)
        self.occ_qb = []
        for k,v in QuantStore.qubit_to_rdm.items():
            self.occ_qb.append(k)

    def add_data(self,output):
        i,k,j,l,s=0,0,0,0,0
        if self.qs.pr_q>2:
            print('Circuit, counts:')
        for name,counts in output:
            prbdis = self.proc_counts(counts)
            if self.qs.pr_q>2:
                print('Circuit: {}'.format(name))
                print('Counts : {}'.format(counts))
                print('Avg    : {}'.format(prbdis))
            if name[0]=='ii':
                self.data['ii']['counts']=counts
                self.data['ii']['pd']=prbdis
            elif name[0][0:3]=='ijR':
                self.data['ijR'].append({})
                self.data['ijR'][l]['qb']=name[1:]
                self.data['ijR'][l]['counts']=counts
                self.data['ijR'][l]['pd']=prbdis
                l+=1
            elif name[0][0:3]=='ijI':
                self.data['ijI'].append({})
                self.data['ijI'][j]['qb']=name[1:]
                self.data['ijI'][j]['counts']=counts
                self.data['ijI'][j]['pd']=prbdis
                j+=1
            elif name[0][0:2]=='ij' and (not name[0][0:4]=='ijkl'):
                self.data['ij'].append({})
                self.data['ij'][k]['qb']=name[1:]
                self.data['ij'][k]['counts']=counts
                self.data['ij'][k]['pd']=prbdis
                k+=1
            elif name[0][0:4]=='sign':
                self.data['sign'].append({})
                self.data['sign'][s]['name']=name[0][4:]
                self.data['sign'][s]['counts']=counts
                self.data['sign'][s]['pd']=prbdis
                s+=1 
            else:
                pass

    def sign_from_2rdm(self):
        self.sign = [1]
        if self.qs.tomo_ext=='sign_2e':
            for s,item in enumerate(self.data['sign']):
                i,j,k,l = item['name'].split('-')
                i,j,k,l = int(i),int(j),int(k),int(l)
                # note, now we are constructing 2rdm elements
                t1 = 0.25*(item['pd'][i]-item['pd'][j])
                #print(i,j,k,l,t1
                if t1>0:
                    self.sign.append(1)
                else:
                    self.sign.append(-1)
            if self.qs.pr_q>2:
                print('rdm2 sign element: {}'.format(t1))
        elif self.qs.tomo_ext=='sign_2e_pauli':
            hold = [] #stores....stuff? 
            holding = {}
            nsign = 1
            for s,item in enumerate(self.data['sign']):
                self._measure_z_rdm2(s)
                if item['qbs'] in hold:
                    holding[item['qbs']][item['pauli']]=item['z']
                else:
                    hold.append(item['qbs'])
                    holding[item['qbs']]={
                        item['pauli']:item['z']
                        }
                    holding[item['qbs']]['n']=item['n']
                    nsign+=1
            self.sign=[1]*nsign
            for quad in holding.keys():
                dat =  holding[quad]
                rdme = {
                        '++--':0,
                        '+-+-':0,
                        '+--+':0,
                        'fo':0,
                        'so':0,
                        }
                for seq,val in dat.items():
                    if seq=='n':
                        n = int(val)
                        continue
                    a,b,c,d,e=1/8,1/8,1/8,0,0
                    if seq in ['xxxx','yyyy']:
                        d,e = 1/2,1/4
                    elif seq in ['xxyy','yyxx']:
                        a*=-1
                        e=1/4
                    elif seq in ['xyyx','yxxy']:
                        c*=-1
                    elif seq in ['xyxy','yxyx']:
                        b*=-1
                    rdme['++--']+= a*val
                    rdme['+-+-']+= b*val
                    rdme['+--+']+= c*val
                    rdme['fo']+= d*val
                    rdme['so']+=(e*val)
                holding[quad]['rdme']=rdme
                if self.qs.tomo_approx=='full':
                    test = rdme['++--']
                elif self.qs.tomo_approx=='fo':
                    test = rdme['fo']
                elif self.qs.tomo_approx=='so':
                    test = rdme['so']
                if test<0:
                    self.sign[n+1]=-1
                if self.qs.pr_q>2:
                    print('rdm2 sign elements: {}'.format(rdme))
            self.holding = holding
        else:
            self.sign=[1]*10
            self.holding = {}

    def _measure_z_rdm2(self,s):
        '''
        function to analyze counts for some double excitaiton pauli operators
        measurement is made in the z basis
        indices should be sequenctial, i.e. - i<j<k<l
        '''
        total = 0
        val = 0 
        i,j,k,l,n,pauli = self.data['sign'][s]['name'].split('-')
        self.data['sign'][s]['qbs']='-'.join((i,j,k,l))
        self.data['sign'][s]['n']=n
        i,j,k,l = int(i),int(j),int(k),int(l)
        self.data['sign'][s]['pauli']=pauli
        self.ns = 0
        for det,count in self.data['sign'][s]['counts'].items():
            total += count
            temp = 1
            for p in range(i,l+1):
                if p<k and p>j:
                    pass
                else:
                    if det[self.Nq_tot-1-p]=='1':
                        temp = temp*(-1)
            val += temp*count
        self.data['sign'][s]['z']=val/total

        if self.qs.pr_q>2:
            print('Pauli:',pauli,val/total)

    def proc_counts(self,counts):
        Nc = 0
        r  = zeros(self.Nq_tot)
        for qb_state,outcome in counts.items():
            for i in range(0,self.Nq_tot):
                if qb_state[i]=='1':
                    r[self.Nq_tot-1-i]+=outcome
            Nc += outcome
        r = r*(1/Nc)
        return r

    def build_rdm(self):
        if self.qs.fermion_mapping=='jordan-wigner':
            self._build_direct_rdm()
        elif self.qs.fermion_mapping=='compact':
            self._build_compact_rdm(**self.kw)


    def _build_direct_rdm(self):
        '''
        now, we have the qubit result. we just need to ocnvert this back to RDM
        speak. 
        '''
        self.r2q = self.qs.rdm_to_qubit
        self.q2r = self.qs.qubit_to_rdm
        if self.qs.tomo_bas=='bch':
            self.rdm = zeros((self.Nq_act,self.Nq_act))
            for actqb in self.occ_qb:
                ind = self.q2r[actqb]
                self.rdm[ind,ind]=self.data['ii']['pd'][actqb]
            for item in self.data['iIj']:
                for pair in item['qb']:
                    i1,i2 = int(pair[0]),int(pair[1])
                    i1,i2 = self.a2b[i1],self.a2b[i2]
                    val = item['pd'][i1]*0.5
                    val -= 0.25*(self.rdm[i2,i2]+self.rdm[i1,i1])
                    self.rdm[i2,i1]+=val
                    self.rdm[i1,i2]+=val
            for item in self.data['iZj']:
                for pair in item['qb']:
                    i1,i2 = int(pair[0]),int(pair[1])
                    i1,i2 = self.a2b[i1],self.a2b[i2]
                    val = item['pd'][i1]*0.5
                    val -= 0.25*(self.rdm[i2,i2]+self.rdm[i1,i1])
                    self.rdm[i1,i2]+=val
                    self.rdm[i2,i1]+=val
        elif self.tomo_basis=='hada':
            self.rdm = zeros((self.Nq_act,self.Nq_act))
            for actqb in self.occ_qb:
                ind = self.a2b[actqb]
                temp = self.data['ii']['pd'][ind]
                self.rdm[actqb,actqb]=temp
            for item in self.data['ijR']:
                for pair in item['qb']:
                    i1,i2 = int(pair[0]),int(pair[1])
                    i1,i2 = self.a2b[i1],self.a2b[i2]
                    val = 0.5*(item['pd'][i2]-item['pd'][i1])
                    self.rdm[i2,i1]+=val
                    self.rdm[i1,i2]+=val
        elif self.tomo_basis=='hada+imag':
            self.rdm = zeros((self.Nq_act,self.Nq_act),dtype=complex_)
            for actqb in self.occ_qb:
                ind = self.a2b[actqb]
                temp = self.data['ii']['pd'][ind]
                self.rdm[actqb,actqb]=temp
            for item in self.data['ijR']:
                for pair in item['qb']:
                    i1,i2 = int(pair[0]),int(pair[1])
                    i1,i2 = self.a2b[i1],self.a2b[i2]
                    val = 0.5*(item['pd'][i2]-item['pd'][i1])
                    self.rdm[i2,i1]+=val
                    self.rdm[i1,i2]+=val
            for item in self.data['ijI']:
                for pair in item['qb']:
                    i1,i2 = int(pair[0]),int(pair[1])
                    i1,i2 = self.a2b[i1],self.a2b[i2]
                    val = 0.5*(item['pd'][i2]-item['pd'][i1])
                    self.rdm[i2,i1]+=val*(-1j)
                    self.rdm[i1,i2]+=val*(1j)
        elif self.tomo_basis=='no':
            self.rdm = zeros((self.Nq_act,self.Nq_act))
            for actqb in self.occ_qb:
                ind = self.q2r[actqb]
                self.rdm[ind,ind]=self.data['ii']['pd'][actqb]

    def _build_compact_rdm(self,**kwargs):
        if self.qs.tomo_rdm=='1rdm':
            try:
                self.data['ij']['counts']
                use_err=True
            except KeyError:
                self.data['ij']={'counts':None}
                use_err=False
            except TypeError:
                self.data['ij']={'counts':None}
                use_err=False
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
            self.rdm1c, self.on, self.rdm1ev = fx.construct_rdm(
                    fx.rdm(
                        fx.filt(self.data['ii'],temp)
                        ),
                    fx.rdm(
                        fx.filt(self.data['ij'],temp)
                        )
                    )
            if self.pr_q>2:
                print('Trace of 2-RDM: {}'.format(self.rdm2trace))

