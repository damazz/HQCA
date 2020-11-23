from hqca.tools import *
import sys
import numpy as np
from math import pi
from hqca.hamiltonian import *


h1,h2,h3 = 1,-1.0,-0.5
to = Operator()
to+= PauliOperator('ZII',-2)
to+= PauliOperator('IIZ',1)
to+= PauliOperator('IZI',-5)
#to+= PauliOperator('XXI',h1)
#to+= PauliOperator('YYI',h1)
#to+= PauliOperator('XIX',h2)
#to+= PauliOperator('YIY',h2)
#to+= PauliOperator('IXX',h3)
#to+= PauliOperator('IYY',h3)

H = QubitHamiltonian(
        3,
        operator='pauli',
        order=3,
        pauli=to
        )

tc = Circ(3) #test circuit
#a,b,c = 0.1,0.5,0.33
a,b,c = 0,0,0

ts = State(3)
ts*tc


rho = DensityMatrix(ts)

rho2 = qRDM(1,3)
rho2.from_density_matrix(rho)


en = np.dot(rho.rho,H.matrix[0]).trace()
print('Output energy: {}'.format(np.real(en)))

H = QubitHamiltonian(
        3,
        operator='pauli',
        order=1,
        pauli=to
        )

en = 0
print(rho2.rdm)
for a,b in zip(H.matrix,rho2.rdm):
    en+= np.dot(a,b).trace()
print('Output reduced energy: {}'.format(np.real(en)))
