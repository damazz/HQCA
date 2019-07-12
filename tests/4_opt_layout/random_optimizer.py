import hqca.tools.util.Utilities as util
from functools import partial
import pickle
import sys
import timeit
from gloa.gloa import alg

# load file
filename = '071119_ibmq_16_melbourne'
qasm = 'testing.qasm'
mel = util.CircuitProperties(filename)
mel.more()
mel.load_qasm(qasm)

kwf = {'Circuit':mel}
kwr = {'Circuit':mel,'N':6}
kwm = {'Circuit':mel}

fitness = partial(util.fitness_random_qasm,**kwf)
random  = partial(util.generate_really_random,**kwr)
mutate  = partial(util.full_mutate_sequence,**kwm)
param   = util.parameter_transfer


test = alg.algorithm(
        fitness=fitness,
        rf=random,
        groups=15,
        members=30,
        mutation=[0.8,0.1,0.1],
        param_type='specified',
        evaluate=print,
        tolerance=0.05,
        mutation_function=mutate,
        parameter_transfer=param,
        )
test.start()
test.execute(250)


