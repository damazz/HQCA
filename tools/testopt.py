import sys
from nevergrad.optimization import optimizerlib


from nevergrad.optimization import registry
#for i in sorted(registry.keys()):
#    print(i)
#    opt = registry[i]
#    print(opt(dimension=2,budget=1000))

def beale(p):
    x,y = p[0],p[1]
    a = (1.5- x + x*y)**2
    a+= (2.25- x + x*(y**2))**2
    a+= (2.625 - x + x*(y**3))**2
    return a

optimizer = optimizerlib.OnePlusOne(dimension=2,budget=1000)
#print(optimizer)
#recommendation = optimizer.optimize(beale)
#print(recommendation)

for y in range(optimizer.budget):
    x = optimizer.ask()
    value = beale(x)
    optimizer.tell(x,value)
    if y%100==0:
        print(x,value)
    print(optimizer.budget)
print(optimizer.provide_recommendation())






