from hqca.sub.Transpile import TranspileQASM
from os import getcwd

circ = TranspileQASM(getcwd()+'/test4.qasm')

circ.get_backend('ibmq_16_melbourne')
#circ.get_backend('ibmqx4')
circ.standard(3,5,500)

