from scipy.optimize import minimize

def beale2(p):
    x,y = p[0],p[1]
    a = (1.5- x + x*y)**2
    a+= (2.25- x + x*(y**2))**2
    a+= (2.625 - x + x*(y**3))**2
    return a

test = minimize(
        beale2,
        [0,0],
        method='COBYLA',
        tol=1e-5
        )
print(test.x)

