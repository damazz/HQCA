from hqca.tools import *
import sys

new1 = FermionicOperator(
        coeff=1,
        indices=[5,2,0,5],
        sqOp='++--',
        spin='bbaa',
        add=True
        )
new2 = FermionicOperator(
        coeff=1,
        indices=[5,0,2,5],
        sqOp='++--',
        spin='baab',
        add=True
        )
#new3 = FermionicOperator(
#        coeff=1,
#        indices=[0,5,5,2],
#        sqOp='++--',
#        spin='abba',
#        add=True
#        )
#new4 = FermionicOperator(
#        coeff=1,
#        indices=[2,5,5,0],
#        sqOp='++--',
#        spin='abba',
#        add=True
#        )

new1.generateOperators(6)
new2.generateOperators(6)
#new4.generateOperators(6)

newA = new1.formOperator()
newA += new2.formOperator()
print(newA)
#newB = new3.formOperator()
#newB += new4.formOperator()
#print('------')
#print(newB)
#print(newA+newB)
