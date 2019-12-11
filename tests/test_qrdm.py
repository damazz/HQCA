from hqca.tools import *

new = QubitOperator(
        coeff=2,
        indices=[0,1,2,2],
        sqOp='+-+-'
        )
new = QubitOperator(
        coeff=2,
        indices=[0,1,4],
        sqOp='+-p',
        )

new.generateOperators(6,real=False,imag=True)

test = new.formOperator()
print(test)
