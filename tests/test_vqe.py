'''
/hqca/tests/test_vqe.py

Test the variational quantum eigensolvers.
'''
from _generic import *
from hqca.vqe import *
from hqca.opts import Optimizer
from functools import partial
import hqca.config as config
config._use_multiprocessing=False

def test_generic():
    ham, st, qs, ins, proc, tomoRe = generic_vqe_objects()
    opt = partial(Optimizer,
                  optimizer='bfgs',
                  unity=np.pi/2,
                  verbose=True)
    vqe = RunVQE(st,opt,qs, ins,
                 tomo_Psi=tomoRe,
                 gradient=True,
                 )
    vqe.build()
    assert vqe.built
    vqe.run()
    assert abs(vqe.best-vqe.Store.H.ef)<1e-8

test_generic()
