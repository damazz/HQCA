from hqca.tools import *
import numpy as np

a = Circ(2)
a.s(0)
a.h(0)
a.si(0)
a.s(0)
a.h(0)
a.si(0)#
a.s(1)
a.h(1)
a.si(1)
a.s(1)
a.h(1)
a.si(1)#

a.si(1)
a.h(0)
a.h(1)
a.Cx(0,1)
a.Rz(1,np.pi/6)
a.Cx(0,1)
a.h(0)
a.h(1)
a.s(1)
a.si(0)
a.h(0)
a.h(1)
a.Cx(0,1)
a.Rz(1,-np.pi/6)
a.Cx(0,1)
a.h(0)
a.h(1)
a.s(0)

a.s(0)
a.h(0)
a.si(0)
a.s(0)
a.h(0)
a.si(0)#
a.s(1)
a.h(1)
a.si(1)
a.s(1)
a.h(1)
a.si(1)#

b = Circ(2)
b.h(1)
b.Cx(1,0)
b.Ry(0,-np.pi/6)
b.Ry(1,-np.pi/6)
b.Cx(1,0)
b.h(1)
print(b.m)
print(a.m-b.m)

c = Circ(1)
c.s(0)
c.h(0)
c.si(0)
c.h(0)
print(c.m)