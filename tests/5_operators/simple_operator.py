from hqca.tools import *
from hqca.transforms import *
import numpy as np

new = Operator()
p1 = PauliString('XY',1)
p2 = PauliString('ZZ',0.5)



p3 = FermiString(1,
        ops='++--',
        indices=[0,1,1,2],
        N=6,
        )
p4 = FermiString(0.5,
        ops='++--',
        indices=[2,3,3,4],
        N=6,
        )
p5 = FermiString(1,ops='+-',indices=[0,2],N=3)
p6 = FermiString(1,ops='-+',indices=[0,2],N=3)
f,g = Operator(),Operator()
f+= p5
f+= p6
pa = f.transform(JordanWigner)


U = Operator()
U+= PauliString('XII',1/np.sqrt(2))
U+= PauliString('ZIZ',1/np.sqrt(2))


x = change_basis(pa,U)
print(x)
x = trim_operator(x,
        qubits=[1],
        eigvals=[-1],
        paulis=['Z'],
        )
print(x)


p = Recursive(depth=4,choices=[['+','+','-','-']])
p.choose()
p.simplify()
