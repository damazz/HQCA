import numpy as np
import math
e1 = 0.0625
e2 = 0.08594
e3 = 0.00391

l1 = 0.47025
l2 = 0.52975
l3 = 0.91701
l4 = 1-l3
l5 = 0.88086
l6 = 1-l5

x1 = 1/2 + np.sqrt(1-4*(l1*l2+e1**2))/2
x2 = 1-x1
x3 = 1/2 - np.sqrt(1-4*(l3*l4+e2**2))/2
x4 = 1-x3
x5 = 1/2 - np.sqrt(1-4*(l5*l6+e3**2))/2
x6 = 1-x5

print(x1,x2,x3,x4,x5,x6)
print(math.isnan(float(x1)))


