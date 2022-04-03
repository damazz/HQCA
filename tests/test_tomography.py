from delayed_assert import delayed_assert as da
from hqca.tomography import *
from _generic import *
import numpy as np
import hqca.config as config
config._use_multiprocessing=False
from hqca.transforms import *
import pytest
from hqca.tomography._mixed_tomography import MixedTomography

@pytest.mark.skip('Mixed tomography likely not needed.')
def test_mixed():
    '''
    test mixed tomography for a fermionic qubit-particle matrix 
    '''
    #qs = tomo_quantumstorage()
    qs = generic_quantumstorage()
    tomoRe = MixedTomography(qs,
            lorder=1,
            rorder=1)
    tomoRe.generate(real=True,imag=False,transform=Qubit)
    print(tomoRe.op)

    tomo2 = ReducedMixedTomography(
            qs,
            lorder=2,
            rorder=2)
    tomo2.generate(real=False,imag=True,transform=Qubit)
    print(tomo2.op)

def test_tomography():
    qs = generic_quantumstorage()
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=JordanWigner)
    tomoIm = StandardTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=JordanWigner)
    print(tomoIm.op)
    temp = set([
        'XXXX','XXYY','XYYX','XYXY',
        'YYYY','YYXX','YXXY','YXYX',
        'XXZZ','YYZZ','ZZXX','ZZYY','ZZZZ'
        ])
    da.expect(set(tomoRe.op) == temp)
    tempI = set([
        'XYYY','YXYY','YYXY','YYYX',
        'XXXY','YXXX','XXYX','XYXX',
        'XYZZ','YXZZ','ZZXY','ZZYX',
        ])
    da.expect(set(tomoIm.op) == tempI)
    da.assert_expectations()

def test_reduced_tomography():
    qs = generic_quantumstorage()
    tomoRe = ReducedTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=JordanWigner)
    tempR = set([
        'XXZZ','YXYX','ZZXX','XXXX','ZZZZ'
        ])
    print(tomoRe.op)
    da.expect(set(tomoRe.op) == tempR)
    da.assert_expectations()


def test_compare_constructions():
    qs = generic_quantumstorage()
    coeff =  [np.random.random()*np.pi/2-np.pi/4 for i in range(3)]
    tomo0 = StandardTomography(qs,verbose=False)
    tomo0.generate(real=True,imag=True,
            simplify=True,transform=JordanWigner,
            method='gt',strategy='lf')
    ins = genericIns(coeff)
    tomo0.set(ins)
    tomo0.simulate(verbose=False)
    tomo0.construct()
    tomo1 = ReducedTomography(qs,verbose=False)
    tomo1.generate(real=True,imag=True,
            simplify=True,transform=JordanWigner,
            method='gt',strategy='lf')
    tomo1.set(ins)
    tomo1.simulate(verbose=False)
    tomo1.construct()
    d01 = tomo0.rdm-tomo1.rdm
    d01.contract()
    da.expect(abs(d01.rdm.trace())<1e-10)
    da.assert_expectations()



