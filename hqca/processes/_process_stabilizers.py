from hqca.core import *
import numpy as np
import sys

class StabilizerProcess(Process):
    def __init__(self,
            stabilizer=None,
            **kw):
        if stabilizer in ['simple','parity']:
            self.stable = self._parity_check_simple
        elif stabilizer in ['spin']:
            self.stable = self._parity_check_spin
        pass

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
        kw['counts']=self.stable(**kw)
        return self.__measure_z_string(**kw)

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
            #print(Nq)
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
