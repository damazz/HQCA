from hqca.tools import *




x = PauliOperator('XX',1)
y = PauliOperator('YY',1)
z = PauliOperator('ZZ',1)
i = PauliOperator('II',1)


a = Operator()
b = Operator()
a+= x 
#a+= y
b+= z
b+= i
print(a)
print(b)
print('')
print(a*b)


