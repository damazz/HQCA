from hqca.tools import *
import numpy as np
from hqca.transforms import *

test = Recursive(choices=[0,1,2,3],depth=4)
test.permute()
print(test.total)
new = Operator()
new+= FermiString(1,ops='pp',indices=[0,1],N=4)
new+= FermiString(1,ops='pp',indices=[0,2],N=4)
new+= FermiString(1,ops='pp',indices=[0,3],N=4)
new+= FermiString(1,ops='pp',indices=[1,2],N=4)
new+= FermiString(1,ops='pp',indices=[1,3],N=4)
new+= FermiString(1,ops='pp',indices=[2,3],N=4)

#
#
#

a = Operator()
a+= FermiString(0.5,ops='++--',indices=[0,1,2,3],N=4)

qa = a.transform(JordanWigner)
print(qa)

qn = new.transform(JordanWigner)
print(qn)

temp = np.zeros((16,16),dtype=np.complex_)
for i in qn:
    Q = Circ(4)
    for n,j in enumerate(i.s):
        if j=='X':
            Q.x(n)
        elif j=='Y':
            Q.y(n)
        elif j=='Z':
            Q.z(n)
    temp+= Q.m*i.c

for i in qa:
    Q = Circ(4)
    for n,j in enumerate(i.s):
        if j=='X':
            Q.x(n)
        elif j=='Y':
            Q.y(n)
        elif j=='Z':
            Q.z(n)
    print(np.count_nonzero(
        np.dot(temp,np.dot(Q.m,temp))))
    print(temp,np.dot(Q.m,temp))
