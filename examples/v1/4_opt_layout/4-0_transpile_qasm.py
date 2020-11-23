from hqca.sub.Transpile import TranspileQASM
from os import getcwd

circ = TranspileQASM(getcwd()+'/rdme2.qasm')

#circ.get_backend('ibmq_16_melbourne')
circ.get_backend('ibmqx2')
circ.standard(3,12,500)

