import numpy as np
from functools import reduce
from scipy import stats
from copy import deepcopy as copy
import sys
import traceback
from timeit import default_timer as dt
from hqca.core import *
from hqca.tools import *
from hqca.circuits import *
from hqca.state_tomography._reduce_circuit import simplify_tomography
from hqca.state_tomography._reduce_circuit import compare_tomography
from hqca.state_tomography._tomography import StandardTomography
from hqca.processes import *
from hqca.state_tomography.__constant_project import *
from hqca.state_tomography.__symmetry_project import *
from hqca.core.primitives import *
from qiskit import transpile,assemble,execute

class ConRDMElement:
    def __init__(self,op,qubOp,ind=None):
        self.rdmOp = op
        self.qubOp =  qubOp
        try:
            ind[0]
            self.ind=ind
        except Exception:
            self.ind = op.op[0].inds()

class ReducedTomography(StandardTomography):
    def _generate_1rdme(self,
            real=True,
            imag=False,
            **kw):
        raise AttributeError

    def _generate_2rdme(self,
            real=True,
            imag=False,
            verbose=False,
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

            def sub_rdme(i,k,l,j,transform,**kw):
                op = Operator()
                if self.real and self.imag:
                    c1,c2=1,0
                elif self.real and not self.imag:
                    c1,c2=0.5,0.5
                elif not self.real and self.imag:
                    c1,c2 = 0.5,-0.5
                test = FermiString(
                    coeff=c1,
                    indices=[i,k,l,j],
                    ops='++--',
                    N = self.qs.dim,
                    )
                op+=test
                test = FermiString(
                    coeff=c2,
                    indices=[j,l,k,i],
                    ops='++--',
                    N=self.qs.dim,
                    )
                op+= test
                #con = ConstantNumberProjection(op,transform)
                con = SymmetryProjection(op,transform,self.qs)
                #print('---')
                #print('Qubit op: ')
                #print(con.qubOp)
                #print('Fermi op: ')
                #print(op)
                #print('----')
                # op and qubOp
                return ConRDMElement(op,con.qubOp,ind=[i,k,l,j])
            
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
                            new= sub_rdme(i,k,l,j,**kw)
                            rdme.append(new)
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
                            new = sub_rdme(i,k,l,j,**kw)
                            rdme.append(new)
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
                            new = sub_rdme(i,k,l,j,**kw)
                            rdme.append(new)
            self.rdme = rdme
        '''
        if not self.grouping:
            alp = self.qs.groups[0]
            Na = len(alp)
            rdme = []
            bet = self.qs.groups[1]
            S = []

            def sub_rdme(i,k,l,j,spin):
                test = FermiString(
                    N = self.qs.dim,
                    coeff=1,
                    indices=[i,k,l,j],
                    ops='++--',
                    )
                idx = '-'.join([str(i) for i in test.qInd])
                if self.real and not self.imag:
                    try:
                        tomo = self.qubit_pairing[idx].real[test.qOp]
                    except Exception as e:
                        traceback.print_exc()
                        sys.exit()
                elif not self.real and self.imag:
                    try:
                        tomo = self.qubit_pairing[idx].imag[test.qOp]
                    except Exception:
                        pass
                elif self.real and self.imag:
                    tomo = self.qubit_pairing[idx].real[test.qOp]
                    tomoI = self.qubit_pairing[idx].imag[test.qOp]
                    for i in tomoI:
                        tomo.append(i)
                if test.qCo==-1:
                    for i in range(len(tomo)):
                        tomo[i][1] = tomo[i][1]*test.qCo
                return PseudoRDMElement(tomo,test.ind)

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
                            rdme.append(sub_rdme(i,k,l,j,'aaaa'))
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
                            rdme.append(sub_rdme(i,k,l,j,'bbbb'))
            for i in alp:
                for k in bet:
                    for l in bet:
                        for j in alp:
                            if i*Na+k>j*Na+l:
                                continue
                            if imag and not real and i*Na+k==j*Na+l:
                                continue
                            rdme.append(sub_rdme(i,k,l,j,'abba'))
        '''
        self.rdme = rdme

    def _generate_pauli_measurements(self,simplify=True,
            symmetries=[],**kw):
        paulis = []
        for fermi in self.rdme:
            for j in fermi.qubOp:
                if j.s in paulis:
                    pass
                else:
                    paulis.append(j.s)
        for i in symmetries:
            if j in paulis:
                pass
            else:
                paulis.append(j)
            for j in paulis:
                new = self._pauli_commutation(i,j)
                if new in paulis:
                    pass
                else:
                    paulis.append(new)
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

