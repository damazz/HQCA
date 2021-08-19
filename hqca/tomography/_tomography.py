"""
hqca/tomography/_tomography.py

Contains the StandardTomography object, which can be used to generate 1- and 2-RDMs.



"""

import numpy as np
from scipy import stats
import sys
import traceback
from copy import deepcopy as copy
from timeit import default_timer as dt
from functools import partial
from hqca.core import *
from hqca.tools import *
from hqca.operators import *
from hqca.tomography._reduce_circuit import simplify_tomography
from hqca.tomography._reduce_circuit import compare_tomography
from hqca.processes import *
from hqca.core.primitives import *
from hqca.maple import *
from qiskit.transpiler import Layout
from qiskit import transpile,assemble,QuantumRegister,QuantumCircuit,ClassicalRegister
from qiskit import Aer,execute
import pickle
import multiprocessing as mp
import hqca.config as config

class RDMElement:
    def __init__(self,op,qubOp,ind=None,**kw):
        self.rdmOp = op
        self.qubOp = qubOp
        self.ind = ind

def generate_rdme(
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
    op+= FermiString(
            coeff=c1,
            indices=ind,
            ops='+'*n+'-'*n,
            N=N,
            )
    op+= FermiString(
            coeff=c2,
            indices=ind[::-1],
            ops='+'*n+'-'*n,
            N=N,
            )
    qubOp = op.transform(transform)
    return RDMElement(op,qubOp,ind=ind)


class StandardTomography(Tomography):
    '''

    basic instructionsL

    tomo = StandardTomography(QuantStore,**kwargs)
    tomo.generate(real,imag,transform)
    tomo.set()
    tomo.simulate()
    tomo.construct()

    then, you can access tomo.rdm and obtain an RDM object
    '''
    def __init__(self,
            QuantStore,
            preset=False,
            verbose=True,
            Nq=None,
            order=None,
            method='local',
            **kw):
        self.grouping = False
        self.run = False
        if type(QuantStore)==type(None):
            self.Nq= Nq
            self.Nq_tot = Nq
            self.p = order
        else:
            self.Nq = QuantStore.Nq
            self.Nq_tot = QuantStore.Nq_tot
            self.qs = QuantStore
            if type(order)==type(None):
                self.p = QuantStore.p
            else:
                self.p = order
        if preset:
            self._preset_configuration(**kw)
        self.method=method
        self.dim = tuple([
            self.qs.dim for i in range(2*self.p)])
        self.circuits = []
        self.qr = []
        self.cr = []
        self.circuit_list = []
        self.verbose=verbose
        self.op_type = QuantStore.op_type

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
        self.p = len(self.rdme[0].ind)//2
        self.imag = dat[4]

    def _preset_configuration(self,
            Tomo=None,
            **kw
            ):
        self.grouping=True
        self.mapping = Tomo.mapping
        self.op = Tomo.op
        self.rdme = Tomo.rdme
        self.real = Tomo.real
        self.imag = Tomo.imag
        try:
            self.p = Tomo.p
        except Exception:
            pass

    def set(self,Instruct):
        '''
        Setting the instructings, generating circuits.
        '''
        i=0
        t0 = dt()
        for circ in self.op:
            self.circuit_list.append(circ)
            if self.qs.be_type=='sv' and i>0:
                continue
            Q = GenericCircuit(
                    QuantStore=self.qs,
                    _name=circ,
                    )
            init = Operator()
            init+= PauliString('I'*self.qs.Nq,1)
            for n,item in enumerate(self.qs.initial):
                tem = Operator()
                op1 = FermiString(1,
                        indices=[item],
                        ops='+',
                        N=self.qs.dim)
                op2 = FermiString( -1,
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
            i+=1

    def construct(self,
                  compact=False,
            **kwargs):
        '''
        build the RDM or qubit-RDM from simulate

        use keywords from quantstore (self.qs) for error mitigation, etc.
        '''
        try:
            self.rdme[0]
        except Exception:
            sys.exit('Have not specified the rdme elements for tomography.')
        try:
            self.counts
        except AttributeError:
            sys.exit('Did you forget to run the circuit? No counts available.')
        if self.op_type=='fermionic':
            if compact:
                self._build_compact_RDM(**kwargs)
            else:
                self._build_fermionic_RDM(**kwargs)
            #
            # here, we an implement some post processing
            #
            if self.qs.post:
                if 'shift' in self.qs.method:
                    if type(self.qs.Gamma)==type(None):
                        pass
                    else:
                        self.rdm+= self.qs.Gamma*self.qs.Gam_coeff
                if 'sdp' in self.qs.method:
                    try:
                        if type(self.qs.Gamma)==type(None):
                            pass
                        else:
                            self.rdm = purify(self.rdm,self.qs)
                    except Exception as e:
                        self.rdm = purify(self.rdm,self.qs)
        elif self.op_type=='qubit':
            raise TomographyError


    def _build_fermionic_RDM(self,
            processor=None,
            antisymmetry=False,
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
            if self.p==2:
                opAnn = r.ind[2:][::-1]
                opCre = r.ind[0:2]
                reAnn = Recursive(choices=opAnn)
                reCre = Recursive(choices=opCre)
                reAnn.unordered_permute()
                reCre.unordered_permute()
                #print('Hrm.')
                for i in reAnn.total:
                    for j in reCre.total:
                        ind1 = tuple(j[:self.p]+i[:self.p])
                        s = i[self.p]*j[self.p]
                        nRDM[ind1]+=temp*s #factor of 2 is for double counting
                        if not set(i[:2])==set(j[:2]):
                            ind2 = tuple(i[:self.p]+j[:self.p])
                            nRDM[ind2]+=np.conj(temp)*s
            elif self.p==1:
                nRDM[tuple(r.ind)]+=temp
                if len(set(r.ind))==len(r.ind):
                    nRDM[tuple(r.ind[::-1])]+=np.conj(temp)
            elif self.p>2 and self.p<5:
                p = self.p
                opAnn = r.ind[p:][::-1]
                opCre = r.ind[0:p]
                reAnn = Recursive(choices=opAnn)
                reCre = Recursive(choices=opCre)
                reAnn.unordered_permute()
                reCre.unordered_permute()
                # print('Hrm.')
                for i in reAnn.total:
                    for j in reCre.total:
                        ind1 = tuple(j[:self.p] + i[:self.p])
                        s = i[self.p] * j[self.p]
                        nRDM[ind1] += temp * s  # factor of 2 is for double counting
                        if not set(i[:self.p]) == set(j[:self.p]):
                            ind2 = tuple(i[:self.p] + j[:self.p])
                            nRDM[ind2] += np.conj(temp) * s
        self.rdm = RDM(
                order=self.p,
                alpha=self.qs.groups[0],
                beta=self.qs.groups[1],
                rdm=nRDM,
                Ne=self.qs.Ne,
                )


    def _build_compact_RDM(self,
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
            nRDM.append(temp)
        self.rdm = np.asarray(nRDM)

    def generate(self,**kw):
        if self.p==2:
            self._generate_2rdme(**kw)
        elif self.p==1:
            self._generate_1rdme(**kw)
        elif self.p==3:
            self._generate_3rdme(**kw)
        elif self.p == 4:
            self._generate_4rdme(**kw)

        self._generate_pauli_measurements(**kw)

    def _generate_1rdme(self,
            real=True,
            imag=False,
            verbose=False,**kw):
        self.real=real
        kw['verbose']=verbose
        self.imag=imag
        if not self.grouping:
            alp = self.qs.groups[0]
            Na = len(alp)
            rdme = []
            bet = self.qs.groups[1]
            S = []
            if verbose:
                print('Generating alpha-alpha block of 2-RDM')
            for i in alp:
                for j in alp:
                    if i>j:
                        continue
                    if (imag and not real) and i==j:
                        continue
                    rdme.append([i,j])
            if verbose:
                print('Generating beta-beta block of 2-RDM')
            for i in bet:
                for j in bet:
                    if i>j:
                        continue
                    if (imag and not real) and i==j:
                        continue
                    rdme.append([i,j])
            self.rdme = rdme
        else:
            raise TomographyError

    def _generate_2rdme(self,real=True,imag=False,verbose=False,
            **kw):
        self.real=real
        kw['verbose']=verbose
        self.imag=imag
        if not self.grouping:
            alp = self.qs.groups[0]
            Na = len(alp)
            rdme = []
            bet = self.qs.groups[1]
            S = []
            if verbose:
                print('Generating alpha-alpha block of 2-RDM')
            for i in alp:
                for k in alp:
                    if i>=k:
                        continue
                    for l in alp:
                        for j in alp:
                            if j>=l or i*Na+k>j*Na+l:
                                continue
                            if imag and not real and i*Na+k==j*Na+l:
                                continue
                            rdme.append([i,k,l,j])
            if verbose:
                print('Generating beta-beta block of 2-RDM')
            for i in bet:
                for k in bet:
                    if i>=k:
                        continue
                    for l in bet:
                        for j in bet:
                            if j>=l or i*Na+k>j*Na+l:
                                continue
                            if imag and not real and i*Na+k==j*Na+l:
                                continue
                            rdme.append([i,k,l,j])
            if verbose:
                print('Generating alpha-beta block of 2-RDM')
            for i in alp:
                for k in bet:
                    for l in bet:
                        for j in alp:
                            if i*Na+k>j*Na+l:
                                continue
                            if imag and not real and i*Na+k==j*Na+l:
                                continue
                            rdme.append([i,k,l,j])
            self.rdme = rdme
        else:
            raise TomographyError('No grouping?')


    def _generate_3rdme(self,real=True,imag=False,verbose=False,
            **kw):
        """

        :param real:  specify real portion of the 3-RDM
        :param imag:  specify imaginary portion of the 3-RDM
        :param verbose:
        :param kw:
        :return:

        needs aaa, aab, abb, bbb
        """
        self.real=real
        kw['verbose']=verbose
        self.imag=imag
        if not self.grouping:
            alp = self.qs.groups[0]
            N = len(alp)
            self.rdme = []
            bet = self.qs.groups[1]
            S = []
            if verbose:
                print('Generating alpha-alpha-alpha block of 3-RDM')
            def generate_indices(g1,g2,g3):
                temp = []
                for i in g1:
                    for k in g2:
                        if i>=k:
                            continue
                        for m in g3:
                            if k>=m:
                                continue
                            c1 = i*N**2 +k*N+m
                            for n in g3:
                                for l in g2:
                                    if l>=n:
                                        continue
                                    for j in g1:
                                        c2 = j*N**2+l*N+n
                                        if c1>c2 or j>=l:
                                            continue
                                        if imag and not real and c1==c2:
                                            continue

                                        temp.append([i,k,m,n,l,j])
                return temp
            self.rdme+= generate_indices(alp,alp,alp)
            self.rdme+= generate_indices(alp,alp,bet)
            self.rdme+= generate_indices(alp,bet,bet)
            self.rdme+= generate_indices(bet,bet,bet)
        else:
            raise TomographyError('No grouping?')



    def _generate_4rdme(self,real=True,imag=False,verbose=False,
            **kw):
        """

        :param real:  specify real portion of the 3-RDM
        :param imag:  specify imaginary portion of the 3-RDM
        :param verbose:
        :param kw:
        :return:

        needs aaa, aab, abb, bbb
        """
        self.real=real
        kw['verbose']=verbose
        self.imag=imag
        if not self.grouping:
            alp = self.qs.groups[0]
            N = len(alp)
            self.rdme = []
            bet = self.qs.groups[1]
            S = []
            if verbose:
                print('Generating alpha-alpha-alpha block of 3-RDM')
            def generate_indices(g1,g2,g3,g4):
                temp = []
                for i in g1:
                    for k in g2:
                        if i>=k:
                            continue
                        for m in g3:
                            if k>=m:
                                continue
                            for o in g4:
                                if o>=m:
                                    continue
                                c1 = i*N**3+k*N**2+m*N+o
                                for p in g4:
                                    for n in g3:
                                        if n>=p:
                                            continue
                                        for l in g2:
                                            if l>=n:
                                                continue
                                            for j in g1:
                                                c2 = j*N**3+l*N**2+n*N+p
                                                if c1>c2 or j>=l:
                                                    continue
                                                if imag and not real and c1==c2:
                                                    continue
                                                temp.append([i,k,m,o,p,n,l,j])
                return temp
            self.rdme+= generate_indices(alp,alp,alp,alp)
            self.rdme+= generate_indices(alp,alp,alp,bet)
            self.rdme+= generate_indices(alp,alp,bet,bet)
            self.rdme+= generate_indices(alp,bet,bet,bet)
            self.rdme+= generate_indices(bet,bet,bet,bet)
        else:
            raise TomographyError('No grouping?')



    def _generate_pauli_measurements(self,
            real=True,
            imag=False,
            transform=None,
            simplify=True,
            symmetries=None,
            **kw):
        paulis = []
        alpha = self.qs.alpha['qubit']
        beta = self.qs.beta['qubit']
        partial_generate_rdme = partial(generate_rdme,
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
            if self.imag:
                rz=False
            else:
                rz=True
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


    def simulate(self,verbose=False):
        t0 = dt()
        beo = self.qs.beo
        backend_options = {}
        counts = []
        if self.qs.use_noise:
            backend_options['noise_model']=self.qs.noise_model
            backend_options['basis_gates']=self.qs.noise_model.basis_gates
            coupling = self.qs.noise_model.coupling_map
        else:
            if self.qs.be_file in [None,False]:
                if self.qs.be_coupling in [None,False]:
                    if self.qs.backend=='qasm_simulator':
                        coupling=None
                    else:
                        coupling = beo.configuration().coupling_map
                else:
                    coupling = self.qs.be_coupling
            else:
                raise DeviceConfigurationError
                #try:
                #    coupling = NoiseSimulator.get_coupling_map(
                #            device=self.qs.backend,
                #            saved=self.qs.be_file
                #            )
                #except Exception as e:
                #    print(e)
                #    sys.exit()
        #print('-- -- init: {}'.format(dt()-t0))
        t0 = dt()
        if self.qs.transpile=='default':
            if self.qs.be_type=='sv'  and self.method=='local':
                self.qs.Ns = 1


                circuits = []
                m = 0 
                c = self.circuits[0]
                lo = Layout()
                for n,i in enumerate(self.qs.be_initial):
                    lo.add(self.qr[m][n],i)
                layout = lo
                transpile_kw = copy(self.qs.transpiler_keywords)
                circuits.append(transpile(
                    circuits=c,
                    backend=beo,
                    coupling_map=coupling,
                    initial_layout=layout,
                    **transpile_kw
                    ))
                if self.qs.get_gate_count:
                    pseudo = copy(self.circuits[0])
                    pseudo = transpile(
                        circuits=pseudo,
                        backend=beo,
                        initial_layout=layout,
                        optimization_level=2,
                    )
                    self.operator_count = pseudo.count_ops()
                    #print('Psuedo counts (transpiled)')
                    #print(
                    #    self.operator_count)
                    #print('Default counts')
                    #print(circuits[0].count_ops())
                else:
                    self.operator_count = circuits[0].count_ops()

            else:
                circuits = []
                for m,c in enumerate(self.circuits):
                    lo = Layout()
                    for n,i in enumerate(self.qs.be_initial):
                        lo.add(self.qr[m][n],i)
                    #layout = {self.qr[m][n]:i for n,i in
                    #        enumerate(self.qs.be_initial)}
                    layout = lo
                    circuits.append(transpile(
                        circuits=c,
                        backend=beo,
                        coupling_map=coupling,
                        #initial_layout=self.qs.be_initial,
                        initial_layout=layout,
                        **self.qs.transpiler_keywords
                        ))
                self.operator_count = circuits[0].count_ops()

        else:
            sys.exit('Configure pass manager.')
        #print('-- -- transpile: {}'.format(dt()-t0))
        t0 = dt()
        #print(dir(circuits[0]))
        #print(circuits[0]._layout)
        #print(circuits[0])
        #for d in circuits[0].data:
        #    print(d)
        #print(circuits[0].data[-6:])
        #for i in range(len(self.circuit_list[0])):
        #    circuits[0].data.pop(-1)
        qo = assemble(
                circuits,
                shots=self.qs.Ns
                )

        #sys.exit()

        #print('-- -- assemble: {}'.format(dt()-t0))
        #t0 = dt()
        #qo = schedule(qo,beo)
        if self.qs.backend=='unitary_simulator':
            job = beo.run(qo)
            for circuit in self.circuit_list:
                counts.append(job.result().get_counts(circuit))
        elif self.qs.be_type=='sv' and self.method=='local':
            #
            # #
            # # interestingly, running the circuits is faster, but the real time save is 
            # # in the transpilation and assembly steps, which are faster for a single circuit
            # # 
            #
            # for a local tomography, we will try to simply invert the measurement
            # so that we dont have to run the circuit over and over
            # 
            #a = dt()
            #print('Running simulation....')
            job = beo.run(qo)
            #b = dt()
            #print(b-a)
            #print('Running circuit: ')
            #print(b-a)

            psi = np.reshape(job.result().get_statevector(self.circuit_list[0]),(2**self.qs.Nq,1))
            c0 = self.circuit_list[0]

            counts.append(psi[:,0])
            #print('Running tomography')
            unitary_circs = []
            if len(self.circuit_list)>1:
                for ni,c in enumerate(self.circuit_list):
                    qr = QuantumRegister(self.qs.Nq)
                    cr = ClassicalRegister(self.qs.Nq)
                    qc = QuantumCircuit(qr,cr)
                    for n,i in enumerate(c0):
                       if i == 'X':
                           qc.h(qr[n])
                       elif i == 'Y':
                           qc.h(qr[n])
                           qc.s(qr[n])
                    if ni>0:
                        for n, i in enumerate(c):
                            if i == 'X':
                                qc.h(qr[n])
                            elif i == 'Y':
                                qc.sdg(qr[n])
                                qc.h(qr[n])
                    unitary_circs.append(qc)
                us = Aer.get_backend('unitary_simulator')
                job = execute(unitary_circs,us,shots=1).result().results
                for n,c in enumerate(self.circuit_list[:]):
                    U = job[n].data.unitary
                    nPsi = np.dot(U,psi)
                    #print(c)
                    #print(nPsi.T)
                    if n==0:
                        self.psi = nPsi
                    else:
                        counts.append(nPsi[:,0])
            #a = dt()
            #print(b-a)
        elif self.qs.be_type=='sv' and not self.method=='local':
            job = beo.run(qo)
            for n,circuit in enumerate(self.circuit_list):
                counts.append(job.result().get_statevector(circuit))
        elif self.qs.use_noise:
            try:
                job = beo.run(
                        qo,
                        backend_options=backend_options,
                        noise_model=self.qs.noise_model,
                        )
            except Exception as e:
                traceback.print_exc()
                raise TomographyError
            for circuit in self.circuit_list:
                name = circuit
                counts.append(job.result().get_counts(name))
        else:
            try:
                job = beo.run(qo)
                for circuit in self.circuit_list:
                    name = circuit
                    counts.append(job.result().get_counts(name))
                for circ in self.circuits:
                    if circ.name=='Z':
                        print(circ)
                    elif circ.name=='ZZ':
                        print(circ)
                    elif circ.name=='XY':
                        print(circ)
            except Exception as e:
                print('Error: ')
                print(e)
                traceback.print_exc()
        if self.qs.use_meas_filter:
            self.counts  = {}
            for i,j in zip(self.circuit_list,counts):
                c = self.qs.meas_filter.apply(
                    j,
                    method='least_squares'
                    )
                self.counts[i]=c
        else:
            self.counts = {i:j for i,j in zip(self.circuit_list,counts)}
        if self.verbose:
            for i,j in self.counts.items():
                if self.qs.be_type=='sv':
                    pass
                else:
                    print(i,j)

    def evaluate_error(
            self,
            numberOfSamples=256, # of times to repeat
            sample_size=1024, # number of counts in sample
            ci=0.90, #target CI#,
            f=None,
            replace=False,
            spin_alt=False
            ):
        print('Samples: {}'.format(numberOfSamples))
        print('Sample size: {}'.format(sample_size))
        count_list = []
        N = self.qs.Ns
        if sample_size>=N*8:
            sample_size=int(N/8)
        samplesSD = []
        sample_means = []
        counts_list = {}
        for pauli,counts in self.counts.items():
            count_list = []
            for k,v in counts.items():
                count_list = count_list+[k]*v
            counts_list[pauli]=count_list
        for t in range(numberOfSamples):
            t1 = dt()
            sample_mean  = f(
                    self.getRandomRDMFromCounts(
                        counts_list,sample_size
                        )
                    )
            if np.isnan(sample_mean):
                continue
            else:
                sample_means.append(sample_mean)
            t2 = dt()
            #print('Time: {}'.format(t2-t1))
        t = stats.t.ppf(ci,N)
        std_err = np.std(np.asarray(sample_means),axis=0) #standard error of mean
        ci = std_err*np.sqrt(sample_size/N)*t
        return ci

    def getRandomRDMFromCounts(self,counts_list,length):
        random_counts = {}
        for pauli,clist in counts_list.items():
            random_counts[pauli]={}
            sample_list = np.random.choice(clist,length,replace=False)
            for j in sample_list:
                try:
                    random_counts[pauli][j]+=1
                except KeyError:
                    random_counts[pauli][j]=1
        #print('Build random list: {}'.format(t3-t5))
        del self.rdm
        self.counts = random_counts
        self.construct()
        #new = self._build_mod_2RDM(random_counts)
        #print('Build 2rdm: {}'.format(t4-t3))
        return self.rdm

    def _stabilizer(self,Q):
        # this applies the stabilizer circuit
        # self.op is list of measurements
        # self.mapping maps needed pauli measurements to elements of self.op
        # if we need A,B,C, mapped to B,C, self.mapping takes in A,B,C and will
        # output B,C

        stable = self.qs.stabilizer_map[Q.name] # this should 
        Q.apply(Instruct=stable)
        if not self.qs.be_type=='sv':
            Q.qc.measure(Q.q,Q.c)

    def build_stabilizer(self):
        circs = {k:[] for k in self.op}
        for k,v in self.mapping.items():
            circs[v].append(k)
        stabilizer_map = {}
        for k,v in circs.items():
            new = Operator()
            for j in v:
                if not j=='I'*self.Nq:
                    new+= PauliString(pauli=j,coeff=1)
            check = StabilizedCircuit(new,verbose=self.verbose)
            check.gaussian_elimination()
            check.find_symmetry_generators()
            check.construct_circuit()
            check.simplify()
            stabilizer_map[k]=check
        return stabilizer_map

def run_multiple(tomo_list,quantstore,verbose=False):
    """
    TODO: make sure this works....not sure if it is implemented for other things
    :param tomo_list:
    :param quantstore:
    :param verbose:
    :return:
    """
    new_circ = []
    new_circ_list = []
    for n,tomo in enumerate(tomo_list):
        for circ in tomo.circuits:
            new_name = circ.name+'{:02}'.format(int(n))
            new_circ_list.append(new_name)
    ##
    beo = quantstore.beo
    backend_options = quantstore.backend_options
    counts = []
    if quantstore.use_noise:
        backend_options['noise_model']=quantstore.noise_model
        backend_options['basis_gates']=quantstore.noise_model.basis_gates
        coupling = quantstore.noise_model.coupling_map
    else:
        if quantstore.be_file in [None,False]:
            if quantstore.be_coupling in [None,False]:
                if quantstore.backend=='qasm_simulator':
                    coupling=None
                else:
                    coupling = beo.configuration().coupling_map
            else:
                coupling = quantstore.be_coupling
        else:
            raise DeviceConfigurationError
            #try:
            #    coupling = NoiseSimulator.get_coupling_map(
            #            device=quantstore.backend,
            #            saved=quantstore.be_file
            #            )
            #except Exception as e:
            #    print(e)
            #    sys.exit()
    if quantstore.transpile=='default':
        circuits = []
        for z,t in enumerate(tomo_list):
            for m,c in enumerate(t.circuits):
                lo = Layout()
                for n,i in enumerate(quantstore.be_initial):
                    lo.add(t.qr[m][n],i)
                #layout = {self.qr[m][n]:i for n,i in
                #        enumerate(self.qs.be_initial)}
                c.name = c.name+'{:02}'.format(int(z))
                layout = lo
                circuits.append(transpile(
                    circuits=c,
                    backend=beo,
                    coupling_map=coupling,
                    #initial_layout=self.qs.be_initial,
                    initial_layout=layout,
                    **quantstore.transpiler_keywords
                    ))
    else:
        sys.exit('Configure pass manager.')

    qo = assemble(
            circuits,
            shots=quantstore.Ns
            )
    #qo = schedule(qo,beo)
    if quantstore.be_type=='sv':
        job = beo.run(qo,
                      backend_options=backend_options,
                      )
        for n,circuit in enumerate(new_circ_list):
            #if self.verbose and self.Nq<=4:
            #    print('Circuit: {}'.format(circuit))
            #    print(job.result().get_statevector(circuit))
            counts.append(job.result().get_statevector(circuit))
    elif quantstore.use_noise:
        try:
            job = beo.run(
                    qo,
                    backend_options=backend_options,
                    noise_model=quantstore.noise_model,
                    )
        except Exception as e:
            traceback.print_exc()
            raise TomographyError
        for circuit in new_circ_list:
            name = circuit
            counts.append(job.result().get_counts(name))
    else:
        try:
            job = beo.run(qo)
            for circuit in new_circ_list:
                name = circuit
                counts.append(job.result().get_counts(name))
            for circ in new_circ:
                if circ.name=='Z':
                    print('ZZ')
                    print(circ)
                elif circ.name=='ZZ':
                    print(circ)
                elif circ.name=='XY':
                    print(circ)
        except Exception as e:
            print('Error: ')
            print(e)
            traceback.print_exc()
    for t in tomo_list:
        t.counts  = {}
    if quantstore.use_meas_filter:
        new_counts = {}
        for i,j in zip(new_circ_list,counts):
            c = quantstore.meas_filter.apply(
                j,
                method='least_squares'
                )
            n = int(i[-2:])
            new_counts[n][i:-2]=c
    else:
        for i in range(len(new_circ_list)):
            name = new_circ_list[i]
            res = counts[i]
            n = int(name[-2:])
            ni = name[:-2]
            tomo_list[n].counts[ni] = res


