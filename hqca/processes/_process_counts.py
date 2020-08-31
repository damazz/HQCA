from hqca.core import Process
import numpy as np

class StandardProcess(Process):
    def __init__(self):
        '''
        takes a Pauli measurement, measures it 
        '''
        pass

    def process(self,**kw):
        return self.__measure_z_string(**kw)

    def __measure_z_string(self,counts,
            pauli_string,
            backend='qasm_simulator',
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
