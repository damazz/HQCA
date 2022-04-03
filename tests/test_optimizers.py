from scipy.optimize import minimize
from hqca.opts.testing import *
from hqca.opts import *
import numpy as np
import scipy as sy


def test_nelder_mead():
    initial_conditions = [
            [0,0],
            [-1,1.1],
            [-1,1,2],
            [-4,+4,2],
            [-5,7,2],
            [5,-3,-1],
            ]
    problems = [
            Beale,
            Beale,
            Rosenbrock,
            Rosenbrock,
            ]
    for Problem,initial in zip(problems, initial_conditions):
        case = Problem()
        new =  Optimizer('nm',verbose=False,
                function=case.f,
                )
        new.initialize(initial)
        new.run()

        print(type(Problem()))
        assert abs(new.opt.best_f - case.fg)<1e-6
        assert np.linalg.norm(new.opt.best_x-case.xg)<1e-6
        print('Done with problem!')

def test_bfgs():
    initial_conditions = [
            [0,0],
            [-0.5,0.5],
            [-1,1,3],
            [-4,+4,2],
            [-5,7,2],
            [5,-3,-1],
            ]
    problems = [
            Beale,
            Beale,
            Rastrigin,
            Rastrigin,
            Rosenbrock,
            Rosenbrock,
            ]
    for Problem,initial in zip(problems, initial_conditions):
        case = Problem(array=True)
        
        sci= sy.optimize.minimize(method='BFGS',
                fun=case.f,
                x0=initial,
                jac=case.g,
                options={
                    'maxiter':1000,
                    },
                )
        print(sci.message)
        if not sci.success:
            print('Scipy not successful early -')
        print('Scipy solution: ')
        print(sci.x)
        print('Global: ')
        print(case.xg)
        #assert sci.success==True
        #assert np.linalg.norm(sci.x-case.xg)<1e-6
        case = Problem(array=False)
        new =  Optimizer('bfgs',verbose=False,
                function=case.f,
                gradient=case.g
                )


        new.initialize(initial)
        new.run()
        if not sci.success:
            assert abs(new.opt.best_f - case.fg)<1e-6
            assert np.linalg.norm(new.opt.best_x-case.xg)<1e-6
        print('Done with problem:')
        print(type(Problem()))
        print('---------------------')



test_bfgs()
