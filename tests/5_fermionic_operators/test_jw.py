from hqca.tools import *
import sys

op = FermionicOperator(
        coeff=1,
        indices=[0,1,2,3],
        sqOp='+--+',
        spin='aabb',
        add=True
        )
op.generateOperators(Nq=4,mapping='jw')
nop = op.formOperator()
print('')
print('Operator, ++--')
print(nop)
op = FermionicOperator(
        coeff=1,
        indices=[0,1,2,3],
        sqOp='-+++',
        spin='aabb',
        add=True
        )
op.generateOperators(Nq=4,mapping='jw')
nop+= op.formOperator()
print('')
print('Operator, +-+-')
print(nop)
#op = FermionicOperator(
#        coeff=1,
#        indices=[0,1,2,3],
#        sqOp='+--+',
#        spin='aabb',
#        add=True
#        )
#
#op.generateOperators(Nq=4,mapping='jw')
#nop = op.formOperator()
#print('')
#print('Operator, +--+')
#print(nop)
#

