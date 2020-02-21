from hqca.tools import *
from hqca.tools.fermions import *
import sys

mapPar = ParitySet(8,Ne=[0,0],reduced=True)
operators = [
        '++++','+++-','++-+','++--',
        '+-++','+-+-','+--+','+---',
        '-+++','-++-','-+-+','-+--',
        '--++','--+-','---+','----',
        ]

for sop in operators:
    op = FermionicOperator(
            coeff=1,
            indices=[4,5,6,7],
            sqOp=sop,
            spin='bbbb',
            add=True
            )
    op.generateOperators(Nq=8,mapping='parity',MapSet=mapPar)
    nop = op.formOperator()
    print('')
    print('Operator, {}'.format(sop))
    print(nop)



