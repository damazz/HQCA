from hqca.tools import *
import sys
from math import pi

a = Circ(2)
v = State(2)

phi = pi/4

a.Ry(0,pi/4)


sys.exit()

a.Cx(3,1)
a.Ry(3,pi/4)
a.Cx(3,1)

a.Cx(2,0)
a.Ry(2,pi/4)
a.Cx(2,0)

a.Cx(2,1)
a.Ry(2,pi/4)
a.Cx(2,1)

a.Cx(3,0)
a.Ry(3,pi/4)
a.Cx(3,0)

# # # # #
a.Cx(5,3)
a.Ry(5,pi/6)
a.Cx(5,3)

a.Cx(4,2)
a.Ry(4,pi/6)
a.Cx(4,2)

a.Cx(4,3)
a.Ry(4,pi/6)
a.Cx(4,3)

a.Cx(5,2)
a.Ry(5,pi/6)
a.Cx(5,2)

v*a
rho = DensityMatrix(v)


for i in [0,1,2,3,4,5]:
    z = Circ(6)
    z.z(i)
    print('Z{}: {}'.format(i,rho.observable(z)))
print('')
for i in [0,2,4]:
    for j in [0,2,4]:
        if not i==j:
            z = Circ(6)
            z.z(i)
            z.z(j)
            print('Z{}{}: {}'.format(i,j,rho.observable(z)))
    print('')
for i in [0,2,4]:
    for j in[1,3,5]:
        z = Circ(6)
        z.z(i)
        z.z(j)
        print('Z{}{}: {}'.format(i,j,rho.observable(z)))
    print('')

z1234 = Circ(4)
z1234.z(0)
z1234.z(1)
z1234.z(2)
z1234.z(3)

z34=Circ(4)
z34.z(2)
z34.z(3)


