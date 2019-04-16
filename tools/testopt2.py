from nevergrad.optimization import optimizerlib
from nevergrad.optimization import registry
from nevergrad.optimization.recaster import _MessagingThread as mt
import threading

def beale2(p):
    x,y = p[0],p[1]
    return (1.5-x+x*y)**2+(2.25-x+x*(y**2))**2+(2.625-x+x*(y**3))**2


print('Initial Threading')
print(threading.enumerate())
a = registry['OnePlusOne'](2,budget=500)
print('Threading after getting 1+1')
print(threading.enumerate())

print('Threading after each step: ')
for i in range(3):
    x = a.ask()
    a.tell(x,beale2(*x.args))
    print(threading.enumerate())

print('Threading after getting Cobyla')
print(threading.enumerate())
b = registry['Cobyla'](2,budget=5)
print('Threading after each step: ')
for i in range(5):
    x = b.ask()
    b.tell(x,beale2(*x.args))
    print(threading.enumerate())
print('Done. Current threads: ')
print(threading.enumerate())
for t in reversed(threading.enumerate()):
    if type(t)==type(mt(None)):
        pass
        #t.stop()
    else:
        pass
print('Final.')
print(threading.enumerate())


