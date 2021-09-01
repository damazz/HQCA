from hqca.core import *
from functools import partial
from hqca.tools import *
from hqca.operators import *
import numpy as np
import sys

class StabilizerProcess(Process):
    def __init__(self,
            stabilizer=None,
            **kw):
        self.kw = kw
        if stabilizer in ['simple','parity']:
            self.stable = self._parity_check_simple
            self.measure = self.__measure_z_string
        elif stabilizer in ['spin']:
            self.stable = self._parity_check_spin
            self.measure = self.__measure_z_string
        elif stabilizer in ['custom']:
            self.measure = self.__measure_z_string
        elif stabilizer in ['encoded','symmetry']:
            self.stable = self._filter_encoded_symmetries
            self.measure = self._measure_stabilizer_z_string
        elif stabilizer in ['filter_diagonal']:
            self.stable = self._filter_diagonal_symmetries
            self.measure = self.__measure_z_string
        else:
            self.measure = self.__measure_z_string
   
    def _parity_check_simple(self,
            counts,
            quantstore,
            **kw):
        if not quantstore.Nq_anc==1:
            print('Error in number of ancilla needed for parity check.')
            print('Need 1 ancilla.')
            sys.exit()
        Nanc = quantstore.Nq_tot-1-quantstore.Nq
        Ne = quantstore.Ne
        new_counts = {}
        for k,v in counts.items():
            if k[Nanc]==str(Ne%2):
                new_counts[k]=v
        return new_counts

    def _filter_diagonal_symmetries(self,
            original,
            counts,quantstore,
            **kw):
        #print(original)
        if original=='Z'*quantstore.Nq:
            #print('Filtering...')
            #print('Original counts: ')
            #print(counts)
            # then, need to do N filtering
            No = quantstore.No_as*2
            N = Operator()
            Sz= Operator()
            for i in range(No):
                ph = (-1)**(i//(No/2))
                N  += FermiString(coeff=1,indices=[i,i],ops='+-',N=No)
                Sz += FermiString(coeff=ph,indices=[i,i],ops='+-',N=No)
            N = N.transform(quantstore.transform)
            Sz = Sz.transform(quantstore.transform)
            new = {}
            for k,v in counts.items():
                n=0
                sz=0
                for op in N:
                    n+= op.c*self.__measure_z_string(
                            counts={k:v},
                            pauli_string=op.s,
                            Nq=quantstore.Nq,
                            )
                for op in Sz:
                    sz+= op.c*self.__measure_z_string(
                            counts={k:v},
                            pauli_string=op.s,
                            Nq=quantstore.Nq,
                            )
                c1 = (round(n,5)==quantstore.Ne)
                c2 = (round(sz,5)==(quantstore.Ne_alp-quantstore.Ne_bet))
                if c1 and c2:
                    new[k]=v
            #print('Filtered Counts: ')
            #print(new)
            return new
        else:
            return counts


    def _filter_encoded_symmetries(self,
            original,
            #pauli_string,
            counts,quantstore,
            **kw):
        print('Original {} counts: '.format(original))
        print(counts)
        if original=='Z'*quantstore.Nq:
            # then, need to do N filtering
            No = quantstore.No_as*2
            N = Operator()
            Sz= Operator()
            for i in range(No):
                ph = (-1)**(i//(No/2))
                N+=FermiString(coeff=1,indices=[i,i],ops='+-',N=No)
                Sz+=FermiString(coeff=ph,indices=[i,i],ops='+-',N=No)
            N = N.transform(quantstore.transform)
            Sz = Sz.transform(quantstore.transform)
            new = {}
            for k,v in counts.items():
                n=0
                sz=0
                for op in N:
                    n+= op.c*self.__measure_z_string(
                            counts={k:v},
                            pauli_string=op.s,
                            Nq=quantstore.Nq,
                            )
                for op in Sz:
                    sz+= op.c*self.__measure_z_string(
                            counts={k:v},
                            pauli_string=op.s,
                            Nq=quantstore.Nq,
                            )
                c1 = (round(n,5)==quantstore.Ne)
                c2 = (round(sz,5)==(quantstore.Ne_alp-quantstore.Ne_bet))
                if c1 and c2:
                    new[k]=v
            print('Filtered {} counts: '.format(original))
            print(new)
            return new
        # get encoded symmetries
        # first....get stabilized circuit
        stab_circ = quantstore.stabilizer_map[original]
        n_zz = len(stab_circ.zz)
        #print('ZZ Symmetries')
        #for i in stab_circ.zz:
        #    print(i)
        if n_zz==1:
            total = 1
            prod  =-1
        elif n_zz==2:
            total = 2
            prod  = 1
        elif n_zz==3:
            total =1
            prod  = 1
        elif n_zz==0:
            return counts
        new  = {}
        for k,v in counts.items():
            # 
            c_prod=1 
            c_total=0
            for zz in stab_circ.zz:
                native_z  = stab_circ.T_M[zz.s] # GET THE MATCHING Z SYMM
                sgn = self.__measure_z_string(
                        counts={k:v},
                        pauli_string=native_z,
                        Nq=quantstore.Nq,
                        )
                c_prod*=sgn
                c_total+=sgn
            if c_prod==prod and total==abs(c_total):
                new[k]=v
        print('Filtered {} counts: '.format(original))
        print(new)
        return new

    def _measure_stabilizer_z_string(self,
            counts,
            original,
            pauli_string,
            quantstore,
            **kw
            ):
        # target is pauli string
        stab_circ = quantstore.stabilizer_map[original]
        print(original,pauli_string)
        print(stab_circ.paulis)
        generators = stab_circ.paulis[pauli_string]
        msr = 1
        print('Generators')
        print(generators)
        print(stab_circ.T_M)
        fetch = Operator()+PauliString('I'*quantstore.Nq,1)
        for g in generators[0]:
            fetch*= PauliString(stab_circ.T_M[g],1)
        print('Gen: ',g)
        print('Fetch: ')
        print(fetch)
        msr*=self.__measure_z_string(
                counts=counts,
                pauli_string=fetch[0].s,
                Nq=quantstore.Nq,
                )
        return msr*generators[1]



    def _parity_check_spin(self,
            counts,
            quantstore,
            **kw):
        if not quantstore.Nq_anc==2:
            print('Error in number of ancilla needed for parity check.')
            print('Need 2 ancilla.')
            sys.exit()
        Nan1 = quantstore.Nq_tot-quantstore.Nq-1
        Nan2 = quantstore.Nq_tot-quantstore.Nq-2
        Na = quantstore.Ne_alp
        Nb = quantstore.Ne_bet
        new_counts = {}
        for k,v in counts.items():
            if k[Nan1]==str(Na%2) and k[Nan2]==str(Nb%2):
                new_counts[k]=v
        print(counts,new_counts)
        return new_counts


    def process(self,**kw):
        # passing counts through a stabilizer
        kw['counts']=self.stable(**kw)
        return self.measure(**kw)

    def __measure_z_string(self,counts,
            pauli_string,
            backend='statvector_simulator',
            Nq='default',
            **kw):
        if backend in ['statevector_simulator']:
            val = 0
            N = 2**Nq
            test = ['{:0{}b}'.format(
                i,Nq)[::1] for i in range(0,N)]
            for n,b in enumerate(test):
                if abs(counts[n])<1e-14:
                    continue
                sgn = 1
                for i in range(len(b)):
                    if pauli_string[i]=='I':
                        pass
                    else:
                        if b[Nq-i-1]=='1':
                            sgn*=-1
                val+= np.real(counts[n]*np.conj(counts[n])*sgn)
        else:
            val,total= 0,0
            for det,n in counts.items():
                ph=1
                for i,z in enumerate(pauli_string):
                    if z in ['I','i']:
                        continue
                    if det[Nq-i-1]=='1':
                        ph*=-1
                val+= n*ph
                total+=n
            val = val/total
        return val


