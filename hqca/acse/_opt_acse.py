from copy import copy as copy
from functools import partial
from hqca.opts import *


def _opt_step(acse):
    '''
    '''
    testS = copy(acse.A)
    acse._opt_log = []
    acse._opt_en = []
    func = partial(acse._opt_acse_function,newS=testS)
    if acse._opt_thresh=='default':
        thresh = acse.delta/4
    else:
        thresh = acse._opt_thresh
    opt = Optimizer(acse._optimizer,
            function=func,
            verbose=True,
            shift= -1.01*acse.delta,
            initial_conditions='old',
            unity=acse.delta,
            conv_threshold=thresh,
            diagnostic=True,
            )
    opt.initialize([acse.delta])
    # use if nelder mead
    print('Initial Simplex: ')
    for x,e in zip(opt.opt.simp_x,opt.opt.simp_f):
        print(x,e)
    opt.run()
    # run optimization, then choose best run
    #
    #
    if abs(opt.opt.best_x[0])<0.00001:
        acse.accept_previous_step = False
        print('Rejecting optimization step.')
    else:
        acse.accept_previous_step = True
        for s in testS:
            s.c*=opt.opt.best_x[0]

        acse.S = acse.S + testS
        for r,e in zip(acse._opt_log,acse._opt_en):
            if abs(e-opt.opt.best_f)<=1e-8:
                # finding the best energy
                # 
                # 
                acse.Store.update(r.rdm)
        #print(acse.S)
