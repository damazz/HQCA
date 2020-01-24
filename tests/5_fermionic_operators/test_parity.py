from hqca.tools import *
import sys

#operators = [
#        '++++','+++-','++-+','++--',
#        '+-++','+-+-','+--+','+---',
#        '-+++','-++-','-+-+','-+--',
#        '--++','--+-','---+','----',
#        ]
operators = [
        '++','--','+-','-+']

for sop in operators:
    op = FermionicOperator(
            coeff=1,
            indices=[0,2,4,6],
            sqOp=sop,
            spin='ab',
            add=True
            )
    op.generateOperators(Nq=7,mapping='parity')
    nop = op.formOperator()
    print('')
    print('Operator, {}'.format(sop))
    print(nop)



