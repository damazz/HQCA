"""
Symmetry projected tomography. Very similar to traditional tomography, just with a different generating function.

"""
import multiprocessing as mp
from functools import partial
from hqca.core import TomographyError
from hqca.tomography.__symmetry_project import *
from hqca.tomography._qubit_tomography import QubitTomography
from hqca.tomography._reduce_circuit import simplify_tomography, compare_tomography
from hqca.tomography._tomography import StandardTomography
import hqca.config as config

np.set_printoptions(linewidth=200, suppress=True, precision=4)


class ConRDMElement:
    """
    Dummy object for holding RDM elements, indices, and the related qubit operator
    """
    def __init__(self, op, qubOp, ind=None):
        self.rdmOp = op
        self.qubOp = qubOp
        self.ind = ind


def generate_sp_rdme(
        ind,
        real=True,
        imag=False,
        transform=None,
        alpha=None,
        beta=None,
        **kwargs,
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
    con = SymmetryProjection(op, transform, alpha, beta, **kwargs)
    return ConRDMElement(op, con.qubOp, ind=ind)


def generate_sp_qrdme(
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
    con = SymmetryProjection(op, transform, alpha, beta,local_tomography=True)
    return ConRDMElement(op, con.qubOp, ind=ind)


class ReducedTomography(StandardTomography):
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
        items = {
            'real': self.real,
            'imag': self.imag,
            'transform': transform,
            'alpha': alpha,
            'beta': beta,
            }
        for k,v in kw.items():
            items[k]=v
        partial_generate_sp_rdme = partial(generate_sp_rdme,
                                           # *(self.real,self.imag,
                                           #    transform,
                                           #    alpha,
                                           #    beta)
                                           **items,
                                           )
        if config._use_multiprocessing:
            pool = mp.Pool(mp.cpu_count())
            self.rdme = pool.map(partial_generate_sp_rdme, self.rdme)
            pool.close()
        else:
            self.rdme = [partial_generate_sp_rdme(i) for i in self.rdme]
        self.rdme_keys = [i.ind for i in self.rdme]

        for fermi in self.rdme:
            for j in fermi.qubOp:
                if j.s in paulis:
                    pass
                else:
                    paulis.append(j.s)

        def ztype(pauli_string):
            for p in pauli_string:
                if not p in ['I','Z']:
                    return False
            return True
        zpauli = []
        for n in reversed(range(len(paulis))):
            if ztype(paulis[n]):
                zpauli.append(paulis.pop(n))

        if simplify == True:
            if self.imag and not self.real:
                rz = False
            else:
                rz = True
            self.op, self.mapping = simplify_tomography(
                paulis,
                reassign_z=rz,
                **kw)
        elif simplify == 'comparison':
            self.op, self.mapping = compare_tomography(
                paulis,
                **kw)
        else:
            self.op = paulis
            self.mapping = {p: p for p in paulis}
        for z in zpauli:
            self.mapping[z]='Z'*self.qs.Nq
        self.op+= zpauli

class ReducedQubitTomography(QubitTomography):
    def _generate_pauli_from_qrdm(self,
                                     real=True,
                                     imag=False,
                                     transform=None,
                                     simplify=True,
                                     symmetries=None,
                                     **kw):
        paulis = []
        alpha = self.qs.alpha['qubit']
        beta = self.qs.beta['qubit']
        partial_generate_sp_qrdme = partial(generate_sp_qrdme,
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
            self.rdme = pool.map(partial_generate_sp_qrdme, self.rdme)
            pool.close()
        else:
            self.rdme = [partial_generate_sp_qrdme(i) for i in self.rdme]
        self.rdme_keys = [i.ind for i in self.rdme]
        for fermi in self.rdme:
            for j in fermi.qubOp:
                if j.s in paulis:
                    pass
                else:
                    paulis.append(j.s)
        if simplify == True:
            self.op, self.mapping = simplify_tomography(
                paulis,
                **kw)
        elif simplify == 'comparison':
            self.op, self.mapping = compare_tomography(
                paulis,
                **kw)
        else:
            self.op = paulis
            self.mapping = {p: p for p in paulis}
