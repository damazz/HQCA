from hqca.tools import *
import numpy as np

a = Circ(2)
a.Rz(1,np.pi/6)
a.Cx(0,1)
a.Rz(1,2-np.pi/6)
a.Cx(0,1)

print(a.m)


b = Circ(2)
b.Cx(0,1)
b.h(0)
b.z(0)
b.z(1)
b.h(0)
b.Cx(0,1)

print(b.m)

