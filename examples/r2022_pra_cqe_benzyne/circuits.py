from hqca.tools import *

a = Circ(2)
a.s(0)
a.h(0)
a.si(0)
a.x(0)
a.s(0)
a.h(0)
a.si(0)

a.s(1)
a.h(1)
a.si(1)
a.x(1)
a.s(1)
a.h(1)
a.si(1)
print(a.m)

