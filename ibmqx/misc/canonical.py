import sympy as sy
import numpy as np
import numpy.linalg as LA
def canonical_3qb(wf):
    # takes a 3-qubit wafe function, turns it in to the canonical form 
    T_0 = sy.Matrix([[wf[0],wf[1]],[wf[2],wf[3]]])
    T_1 = sy.Matrix([[wf[4],wf[5]],[wf[6],wf[7]]])
    #sy.pprint(T_0)
    #sy.pprint(T_1)
    # find u_0 
    gam,bet = sy.symbols('gam,bet')
    mix = T_0 + (gam)*T_1
    det = mix.det()
    sol1 = sy.solve(det,gam)
    print('Quadratic solutions, ',sol1)
    try:
        gam = sol1[0]
        try:
            gam_imag = sy.ask(gam,sy.Q.imaginary)
        except:
            gam_imag = True
        alp = bet/gam
        if gam_imag:
            ans = sy.solve(alp*sy.conjugate(alp)+bet*sy.conjugate(bet)-1, bet)
            alp = alp.subs(bet,ans)
            bet = bet.subs(bet,ans)
            U1 = sy.Matrix([[alp,bet],[-sy.conjugate(bet),sy.conjugate(alp)]])
        else:
            ans = sy.solve(alp**2 + bet**2 -1, bet)
            print(ans)
            alp = alp.subs(bet,ans)
            bet = bet.subs(bet,ans)
            U1 = sy.Matrix([[alp,bet],[-bet,alp]])
    except Exception as e:
        print(e)
    T1_0 = alp*T_0 + bet*T_1
    sy.pprint(T1_0)
    U2, U3 = T1_0.diagonalize()
    return U1,U2,U3

def exp_round(exp,digits):
    for a in sy.preorder_traversal(exp):
        if isinstance(a,sy.Float):
            exp = exp.subs(a,round(a,digits))
    return exp

def canonical_3qb_val(wf): 
    T_0 = sy.Matrix([[wf[0],wf[1]],[wf[2],wf[3]]])
    T_1 = sy.Matrix([[wf[4],wf[5]],[wf[6],wf[7]]])
    x,y,z = sy.symbols('x,y,z')
    #sy.pprint(T_0)
    #sy.pprint(T_1)
    P = x*T_0 + y*T_1
    det_P = exp_round(P.det(),10)
    print('Determinant: ',det_P)
    if type(det_P)==sy.numbers.Zero:
        U1 = sy.eye(2)      
    else:
        # find unitarys
        P1 = det_P.subs([(x,1),(y,z)])
        sols1 = sy.solve(P1,z)
        try:
            gam =  sols1[0]
            print(sols1)
            sol2 = sy.solve(z*gam*sy.conjugate(z*gam)+z*sy.conjugate(z)-1,z)
            print(sol2)
            try:
                alp = sol2[0]
                bet = alp*gam
            except:
                print('Error with alpha')
            U1 = sy.Matrix([[alp,bet],[-sy.conjugate(bet),sy.conjugate(alp)]])
            #print(U1*U1.adjoint())
        except Exception as e:
            print(e)
            if det_P.subs(y,0)==0:
                U1 = sy.eye(2)
            elif det_P.subs(x,0)==0:
                U1 = sy.Matrix([[0,1],[1,0]])
    #print(U1, U1*U1.adjoint())
    P = P.subs([(x,U1[0,0]),(y,U1[0,1])])
    #print(P)
    try:
        P = np.matrix([[float(P[0,0]),float(P[0,1])],[float(P[1,0]),float(P[1,1])]])
    except:
        P = np.matrix([[np.complex_(P[0,0]),np.complex_(P[0,1])],[np.complex_(P[1,0]),np.complex_(P[1,1])]],dtype=np.complex_)
    print('Determinant:',P[0,0]*P[1,1]-P[0,1]*P[1,0])
    U2,D,U3 = LA.svd(P)
    #D,U3 = LA.eig(P)

    #print(U3*U3.getH())
    #print(D,U3)
    #U2 = U3.getH()
    print(U2*P*U3.getH())
    return U1,U2,U3.getH()

#a,b,c,d,e,f,g,h = sy.symbols('a,b,c,d,e,f,g,h')

#canonical_3qb([a,b,c,d,e,f,g,h])
