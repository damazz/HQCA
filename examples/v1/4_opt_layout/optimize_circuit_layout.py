import hqca.tools.util.Utilities as util
from functools import partial
import pickle
import sys
import timeit
from gloa.gloa import alg

# load file
#filename = '20190514_ibmqx2'
#filename = '071119_ibmq_16_melbourne'
filename = '071419_ibmq_16_melbourne'
qasm = 'eight.qasm'
mel = util.CircuitProperties(filename)
mel.load_qasm(qasm)
kwf = {'Circuit':mel}
kwr = {'Circuit':mel,'N':8}
kwm = {'Circuit':mel,'N':8}

fitness = partial(util.fitness_qasm,**kwf)
random  = partial(util.generate_random_sequence,**kwr)
mutate  = partial(util.mutate_sequence,**kwm)


test = alg.algorithm(
        fitness=fitness,
        rf=random,
        groups=10,
        members=20,
        mutation=[0.8,0.1,0.1],
        param_type='specified',
        evaluate=print,
        tolerance=0.05,
        mutation_function=mutate,
        parameter_transfer=util.no_parameter_transfer
        )
test.start()
test.execute(250)


