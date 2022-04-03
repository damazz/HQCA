"""

"""

from hqca.transforms import *
from hqca.operators import *

a = Operator(FermiString(coeff=1,indices=[0,4],ops='+-',N=8))
#

b = a.transform(PartialJordanWigner)

print(a)
print('')
print(b)
