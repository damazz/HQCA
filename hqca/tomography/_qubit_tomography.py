import numpy as np
from functools import reduce,partial
from scipy import stats
from copy import deepcopy as copy
import sys
import traceback
from timeit import default_timer as dt
from hqca.core import *
from hqca.tools import *
from hqca.operators import *
from hqca.tomography._reduce_circuit import simplify_tomography
from hqca.tomography._reduce_circuit import compare_tomography
from hqca.processes import *
from hqca.tomography._tomography import StandardTomography
from hqca.core.primitives import *
import multiprocessing as mp
from hqca.maple import *
from qiskit.transpiler import Layout
from qiskit import transpile,assemble,execute,schedule
import pickle
import hqca.config as config

class RDMElement:
    def __init__(self,op,qubOp,ind=None,**kw):
        self.rdmOp = op
        self.qubOp = qubOp
        self.ind = ind

def generate_qrdme(
        ind,
        real=True,
        imag=False,
        transform=None,
        alpha=None,
        beta=None,
        ):
    c1,c2 = real/2+imag/2,real/2-imag/2
    if not (real+imag):
        raise TomographyError('Need real and/or real imaginary tomography.')
    op = Operator()
    N = len(alpha+beta)
    n= len(ind)//2
    op+= QubitString(
            coeff=c1,
            indices=ind,
            ops='+'*n+'-'*n,
            N=N,
            )
    op+= QubitString(
            coeff=c2,
            indices=ind[::-1],
            ops='+'*n+'-'*n,
            N=N,
            )
    qubOp = op.transform(transform)
    return RDMElement(op,qubOp,ind=ind)



class QubitTomography(StandardTomography):
    '''
    Tomography

    generate (tomo) 
    set (circuits)
    simulate (circuits)
    construct (object)
    '''
    def __init__(self,*args,tomo_type='rdm',**kwargs):
        self.tomo_type = tomo_type
        StandardTomography.__init__(self,*args,**kwargs)

    def save(self,name):
        temp = [self.op,self.mapping,self.rdme,self.real,self.imag]
        with open(name+'.rto','wb') as fp: 
            pickle.dump(temp,fp)

    def load(self,tomo_object):
        with open(tomo_object,'rb') as fp:
            dat = pickle.load(fp)
        self.op = dat[0]
        self.mapping = dat[1]
        self.rdme = dat[2]
        self.real = dat[3]
        self.imag = dat[4]

    def _preset_configuration(self,
            Tomo=None,
            **kw
            ):
        self.grouping=True
        self.mapping = Tomo.mapping
        self.op = Tomo.op
        self.tomo_type = Tomo.tomo_type
        self.rdme = Tomo.rdme
        self.real = Tomo.real
        self.imag = Tomo.imag
        try:
            self.p = Tomo.p
        except Exception:
            pass

    def set(self,Instruct):
        i=0
        for circ in self.op:
            self.circuit_list.append(circ)
            Q = GenericCircuit(
                    QuantStore=self.qs,
                    _name=circ,
                    )
            init = Operator()
            init+= PauliString('I'*self.qs.Nq,1)
            for n,item in enumerate(self.qs.initial):
                tem = Operator()
                op1 = QubitString(1,
                        indices=[item],
                        ops='+',
                        N=self.qs.dim)
                op2 = QubitString( -1,
                        indices=[item],
                        ops='-',
                        N=self.qs.dim,
                        )
                tem+=op1
                tem+=op2
                try:
                    new = tem.transform(self.qs.initial_transform)
                    init*= new
                except AttributeError:
                    new = tem.transform(self.qs.transform)
                    init*= new
            try:
                U = self.qs.initial_clifford
                apply_clifford_operation(Q,U)
            except AttributeError as e:
                pass
                #print(e)
            except Exception as e:
                print('Error in applying initial clifford transformation.')
                sys.exit(e)
            for s in init:
                apply_pauli_string(Q,s)
            Q.apply(Instruct=Instruct)
            if self.method=='local':
                for n,q in enumerate(circ):
                    pauliOp(Q,n,q)
                    if not self.qs.be_type=='sv':
                        Q.qc.measure(Q.q[n],Q.c[n])
            elif self.method=='stabilizer':
                self._stabilizer(Q)
            else:
                sys.exit('Need to specify method')
            self.circuits.append(Q.qc)
            self.qr.append(Q.q)
            self.cr.append(Q.c)

    def construct(self,compact=False,
            **kwargs):

        '''
        build the RDM or qubit-RDM 

        use keywords from quantstore (self.qs) for error mitigation, etc. 
        '''
        try:
            self.rdme
        except Exception:
            sys.exit('Have not specified the rdme elements for tomography.')
        try:
            self.counts
        except AttributeError:
            sys.exit('Did you forget to run the circuit? No counts available.')
        if self.tomo_type=='pauli':
            self._build_generic_pauli(**kwargs)
        elif self.tomo_type=='rdm':
            if compact:
                self._build_compact_qubit_RDM(**kwargs)
            else:
                self._build_qubit_RDM(**kwargs)

    def _build_generic_pauli(self,
            processor=None,
            variance=False,
            vector=False,
            **kw):
        if type(processor)==type(None):
            processor=StandardProcess()
        if vector:
            result = []
            for op in self.rdme:
                get = self.mapping[op] #self.mapping gets appropriate pauli
                #
                result.append((1j**self.imag)*processor.process(
                        counts=self.counts[get],
                        pauli_string=op,
                        quantstore=self.qs,
                        backend=self.qs.backend,
                        original=get,
                        Nq=self.qs.Nq_tot)
                                     )
            result = np.asarray(result)
        else:
            result = Operator()
            for op in self.rdme:
                get = self.mapping[op] #self.mapping gets appropriate pauli
                #
                result+= PauliString(op,(1j**self.imag)*processor.process(
                        counts=self.counts[get],
                        pauli_string=op,
                        quantstore=self.qs,
                        backend=self.qs.backend,
                        original=get,
                        Nq=self.qs.Nq_tot)
                                     )
        self.result = result

    def _build_qubit_RDM(self,
            processor=None,
            variance=False,**kw):
        if type(processor)==type(None):
            processor=StandardProcess()
        nRDM = np.zeros(self.dim,dtype=np.complex_)
        for r in self.rdme:
            temp = 0
            for op in r.qubOp:
                if op.s=='I'*len(op.s):
                    temp+= op.c
                    continue
                get = self.mapping[op.s] #self.mapping gets appropriate pauli
                # property to get the right pauli
                zMeas = processor.process(
                        counts=self.counts[get],
                        pauli_string=op.s,
                        quantstore=self.qs,
                        backend=self.qs.backend,
                        original=get,
                        Nq=self.qs.Nq_tot)
                temp+= zMeas*op.c
            #
            if self.p==2:
                opAnn = r.ind[2:][::-1]
                opCre = r.ind[0:2]
                reAnn = Recursive(choices=opAnn)
                reCre = Recursive(choices=opCre)
                reAnn.unordered_permute()
                reCre.unordered_permute()
                for i in reAnn.total:
                    for j in reCre.total:
                        ind1 = tuple(j[:self.p]+i[:self.p])
                        nRDM[ind1]+=temp #factor of 2 is for double counting
                        #print(ind1,s)
                        if not set(i[:2])==set(j[:2]):
                            ind2 = tuple(i[:self.p]+j[:self.p])
                            nRDM[ind2]+=np.conj(temp)
                            #print(ind2)
            elif self.p==1:
                nRDM[tuple(r.ind)]+=temp
                if len(set(r.ind))==len(r.ind):
                    nRDM[tuple(r.ind[::-1])]+=np.conj(temp)
        self.rdm = RDM(
                order=self.p,
                alpha=self.qs.groups[0],
                beta=self.qs.groups[1],
                rdm=nRDM,
                Ne=self.qs.Ne,
                )

    def _build_compact_qubit_RDM(self,
            processor=None,
            **kw):
        """
        Generates a compact representation of the RDM, given in terms of the
        unique RDM elements.

        :param processor: processes count and matrix results;
        default is StandarProcess()
        :param kw:
        :return:
        """
        if type(processor)==type(None):
            processor=StandardProcess()
        nRDM = []
        for r in self.rdme:
            temp = 0
            for op in r.qubOp:
                if op.s=='I'*len(op.s):
                    temp+= op.c
                    continue
                get = self.mapping[op.s]
                # self.mapping gets appropriate pauli
                # property to get the right pauli
                zMeas = processor.process(
                        counts=self.counts[get],
                        pauli_string=op.s,
                        quantstore=self.qs,
                        backend=self.qs.backend,
                        original=get,
                        Nq=self.qs.Nq_tot)
                temp+= zMeas*op.c
            nRDM.append(temp)
        self.rdm = np.asarray(nRDM)


    def generate(self,**kw):
        if self.tomo_type=='rdm':
            if self.p==2:
                self._generate_2rdme(**kw)
            elif self.p==1:
                self._generate_1rdme(**kw)
            self._generate_pauli_from_qrdm(**kw)
        elif self.tomo_type=='pauli':
            self._generate_pauli_set(**kw)

    def _generate_pauli_set(self,
                            real=False,
                            imag=True,
                            paulis=None,
                            simplify=True,
                            **kw
                            ):
        self.real = real
        self.imag = imag
        self.rdme = paulis
        if simplify==True:
            self.op,self.mapping = simplify_tomography(
                    paulis,
                    **kw)
        elif simplify=='comparison':
            self.op,self.mapping = compare_tomography(
                    paulis,
                    **kw)
        else:
            self.op = paulis
            self.mapping = {p:p for p in paulis}

    def _generate_pauli_from_qrdm(self,
            transform=None,
            simplify=True,
            **kw):
        paulis = []
        alpha = self.qs.alpha['qubit']
        beta = self.qs.beta['qubit']
        partial_generate_rdme = partial(generate_qrdme,
                           # *(self.real,self.imag,
                           #    transform,
                           #    alpha,
                           #    beta)
                           **{
                               'real': self.real,
                               'imag': self.imag,
                               'transform': transform,
                               'alpha': alpha,
                               'beta': beta,

                           }
                           )
        if config._use_multiprocessing:
            pool = mp.Pool(mp.cpu_count())
            self.rdme = pool.map(partial_generate_rdme, self.rdme)
            pool.close()
        else:
            self.rdme = [partial_generate_rdme(i) for i in self.rdme]
        self.rdme_keys = [i.ind for i in self.rdme]

        for fermi in self.rdme:
            for j in fermi.qubOp:
                if j.s in paulis:
                    pass
                else:
                    paulis.append(j.s)
        if simplify==True:
            self.op,self.mapping = simplify_tomography(
                    paulis,
                    **kw)
        elif simplify=='comparison':
            self.op,self.mapping = compare_tomography(
                    paulis,
                    **kw)
        else:
            self.op = paulis
            self.mapping = {p:p for p in paulis}

    def _transform_q2r(self,rdme):
        nrdme = []
        for i in rdme:
            nrdme.append(self.qs.qubit_to_rdm[i])
        return nrdme
