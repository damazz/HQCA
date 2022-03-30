'''
Linesearch that attempts to satisy the Wolfe conditions.

See Nocedal & Wright, chapter 3
'''

from copy import deepcopy as copy
import numpy as np
from hqca.operators import *
from hqca.acse._quant_S_acse import solveqACSE
from hqca.acse._class_S_acse import solvecACSE
from hqca.acse._qubit_A import solvepACSE
from hqca.acse._user_A import findUserA
from hqca.core import *
from warnings import warn

def _quadratic_interpolate(a_l,a_h,f_l,f_h,g_l):
    # quadratic interpolation
    # from f(p) = alp*(p-l)^2 + bet*(p-l)+gam
    #      g(p) = 2*alp*(p-l) + bet
    # uses f(l), g(l), f(h) to form interpolation
    Del = a_h-a_l
    alpha = (Del**-2)*(f_h - g_l*Del-f_l)
    beta = g_l
    gamma = f_l
    x = - g_l / (2*alpha) + a_l
    delta = a_l
    return np.real(x)

def _cubic_interpolate(a_l,a_h,a_m,f_l,f_h,f_m,g_l):
    # similar to quadratic interpolation except we  
    # now need to solve a linear system equations 
    # f(p) = alp*(p-l)^3 + bet*(p-l)^2 + gam*(p-l) + delta
    # g(p) = 3*alp*(p-l)^2 + 2*bet*(p-l) + gam
    # 
    #[[f(h)-g(l)*(h-l)-f(l)] =[[ (h-l)^3 (h-l)^2 ] [[alp]
    # [f(m)-g(l)*(m-l)-f(l)]]= [ (m-l)^3 (m-l)^2 ]] [bet]]
    #
    gam =   g_l
    delta = f_l
    A = np.matrix(
            [
        [(a_h-a_l)**3,(a_h-a_l)**2],
        [(a_m-a_l)**3,(a_m-a_l)**2],
        ])
    b = np.matrix([
        [f_h - gam*(a_h-a_l) - delta],
        [f_m - gam*(a_m-a_l) - delta]])
    x = np.linalg.solve(A,b)
    alp, bet = x[0,0],x[1,0]
    qa = 3*alp
    qb = 2*bet-6*a_l*alp
    qc = 3*(a_l**2)*(alp)-2*bet*a_l+gam
    alp_p = (-qb/(2*qa))+(np.sqrt(qb**2-4*qa*qc)/(2*qa))
    alp_m = (-qb/(2*qa))-(np.sqrt(qb**2-4*qa*qc)/(2*qa))
    if alp_p>a_l and alp_p<a_h: #i.e., is within [a_l,a_h]:
        print('Cubic alp: {}'.format(np.real(alp_p)))
        return np.real(alp_p)
    else:
        print('Cubic alp: {}'.format(np.real(alp_m)))
        return np.real(alp_m)

def truncation(
        vector,rel_thresh,min_thresh,
        method=['abs'],
        gradient=None,
        hessian=None,
        previous=False,
        E0=0):
    '''
    contrains truncation schemes for the different vectors

    truncation is constrained so that we maintain a good search direction
    '''
    dim = vector.shape[0]
    k_added = []
    #Print('Previous: ')
    #Try:
    #    for i in previous:
    #        print(i.T)
    #Except Exception:
    #    print(previous)
    if 'accept' in method:
        for i in range(dim):
            for k in range(len(previous)):
                if abs(previous[k][i,0])>=min_thresh:#
                    # then, yes we used this term before!
                    k_added.append(i)
    max_element = abs(max(vector.max(), vector.min(), key=abs))
    thresh = max(max_element * rel_thresh, min_thresh)
    #
    current = copy(vector)
    temp = copy(current)
    deltas = np.zeros(dim)
    #
    for i in range(dim):
        deltas[i]= current[i,0]*gradient[i,0]
    temp_deltas = copy(deltas)
    descent = deltas.sum()<0
    if 'abs' in method:
        crit = np.argsort(np.abs(vector.flatten())).tolist()
        if len(crit)==1:
            crit = crit[0]

        max_element = abs(max(vector.max(), vector.min(), key=abs))
        thresh = max(max_element * rel_thresh*0.99999, min_thresh)
    elif 'delta' in method:
        crit = np.argsort(deltas)[::-1]
        max_element = max(deltas.max()*0.99999, deltas.min(), key=abs)
        thresh = max_element * rel_thresh
    else:
        raise OptimizationError('No criteria selected')

    ind = 0
    while ind<dim:
        if abs(temp[crit[ind],0])<=min_thresh:
            temp[crit[ind],0] = 0
        if abs(deltas[crit[ind]])<=1e-10:
            temp[crit[ind],0] = 0
            ind+=1
            continue
        # set new current
        if descent:
            current = copy(temp)
            deltas = copy(temp_deltas)
        else:
            pass
            # else we just use old current and dont change terms
        temp = copy(current)
        temp_deltas = copy(deltas)
        #
        # if any of the first two criteria are True, we do NOT include the term
        check_lt_abs = (abs(temp[crit[ind],0])<thresh) and 'abs' in method
        check_delta  = deltas[crit[ind]]>thresh and 'delta' in method
        check_include = crit[ind] in k_added and 'accept' in method
        if (check_lt_abs or check_delta) and not check_include:
            temp[crit[ind],0] = 0
            temp_deltas[crit[ind]]=0
        #
        descent = temp_deltas.sum()<0
        # sort according to smallest elements
        ind+=1
    #print(current.T)
    nz = np.nonzero(current)
    #for i,j in zip(nz[0],nz[1]):
    #    if abs(current[i,j])>min_thresh:
    #        print(i,j,current[i,j])
    #print(deltas)
    vector = copy(current)
    return vector

def vector_to_operator(vector,fermi=True,user=False,keys=[],N=10,S_min=1e-10):
    '''
    translates a compact vector format to an operator using a list of keys
    that associates values of the vector to 2-RDM operators
    '''
    if user:
        op = Operator()
        for k,v in zip(keys,vector):
            if abs(v[0])>=1e-10:
                op+= PauliString(k,-1j*v[0])
    else:
        op = Operator()
        for i in range(len(keys)):
            # convert indices
            if len(vector.shape)==2:
                if abs(vector[i,0])<=S_min:
                    continue
            if fermi:
                # note, for is for the double counting in the RDM to operator
                # because elements are all are counted in the operator -> RDM, but
                # since we are using a compact representation, we can muiltiply this by 4
                # [i,k,j,l] [k,i,j,l] [k,i,l,j], [i,k,l,j]
                inds = copy(keys[i])
                op += 4*FermiString(
                    vector[i,0],
                    indices=inds,
                    ops='++--',
                    N=N,
                )
            else:
                op += QubitString(
                    +4*vector[i,0],
                    indices=keys[i],
                    ops='++--',
                    N=N,
                )
    return op

class LineSearch:
    def __init__(self,acse):
        self.acse = acse
        self.fermi = acse.acse_update in ['q','c']
        self.keys = acse.rdme
        self.user = False
        self.f_evals = 0
        self.g_evals = 0
        self.N = acse.QuantStore.dim
        self.S_min = acse.S_min
        self.err = False

    def run(self,p,g0=None,c1=1e-4,c2=0.9,line_search='nw',**kwargs):
        self.c1 = c1
        self.c2 = c2
        self.p = p
        self.g0 = g0
        if line_search=='nw':
            self._line_search_nocedal(**kwargs)
    
    def phi(self,alp):
        op = vector_to_operator(
                alp*self.p,
                fermi=self.fermi,
                user=self.user,
                keys=self.keys,
                N=self.N,
                S_min = self.S_min,
                )
        if self.acse.acse_update in ['c', 'q']:
            P = op.transform(self.acse.QuantStore.transform)
        elif self.acse.acse_update in ['p']:
            P = op.transform(self.acse.QuantStore.qubit_transform)
        elif self.acse.acse_update in ['u']:
            P = op
        psi =  copy(self.acse.S)+P
        tCirc = self.acse._generate_real_circuit(psi)
        en = np.real(self.acse.Store.evaluate(tCirc.rdm))
        self.acse.Store.update(tCirc.rdm)
        self.acse.Store.alp = alp
        print('alpha: {}, energy: {}'.format(alp,en))
        self.f_evals+=1
        return np.real(en)

    def dphi(self,alp):
        op = vector_to_operator(
                alp*self.p,
                fermi=self.fermi,
                user=self.user,
                keys=self.keys,
                N=self.N,
                S_min = self.S_min,
                )
        #print(op)
        #print(self.p.T)
        #print(self.keys)
        if self.acse.acse_update in ['c']:
            P = op.transform(self.acse.QuantStore.transform)
            psi =  copy(self.acse.S)+P
            tCirc = self.acse._generate_real_circuit(psi)
            rdm = tCirc.rdm
            G = solvecACSE(
                self.acse.Store,
                self.acse.QuantStore,
                S_min=self.S_min,
                verbose=self.acse.verbose,
                matrix=True,
                rdm=rdm,
                keys=self.keys,
            )
        elif self.acse.acse_update in ['q']:
            P = op.transform(self.acse.QuantStore.transform)
            psi = (copy(self.acse.S) + P)
            if type(self.acse.sep_hamiltonian)==type(None):
                H = self.acse.Store.H.qubit_operator
            else:
                H = self.acse.sep_hamiltonian
            G = solveqACSE(
                H=H,
                operator             =psi,
                process              =self.acse.process,
                instruct             =self.acse.Instruct,
                store                =self.acse.Store,
                quantstore           =self.acse.QuantStore,
                S_min                =self.acse.S_min,
                hamiltonian_step_size=self.acse.hamiltonian_step_size,
                expiH_approx         =self.acse.S_expiH_approx,
                verbose              =self.acse.verbose,
                tomo                 =self.acse.tomo_S,
                matrix               =self.acse._A_as_matrix,
            )
        elif self.acse.acse_update in ['p']:
            if type(self.acse.sep_hamiltonian)==type(None):
                H = self.acse.Store.H.qubit_operator
            else:
                H = self.acse.sep_hamiltonian
            P = op.transform(self.acse.QuantStore.qubit_transform)
            psi =  (copy(self.acse.S)+P)
            G = solvepACSE(
                H=H,
                operator             =psi,
                process              =self.acse.process,
                instruct             =self.acse.Instruct,
                store                =self.acse.Store,
                quantstore           =self.acse.QuantStore,
                S_min                =self.acse.S_min,
                hamiltonian_step_size=self.acse.hamiltonian_step_size,
                expiH_approx         =self.acse.S_expiH_approx,
                verbose              =self.acse.verbose,
                tomo                 =self.acse.tomo_S,
                matrix               =self.acse._A_as_matrix,
            )
        else:
            raise QuantumRunError
        g = np.asmatrix(G)
        self.g_evals+=1 
        #print('grad large elements')
        #for i in range(len(G)):
        #    if abs(g[0,i])>1:
        #        print(self.keys[i],g[0,i])
        #
        #print(g)
        #print(g.dot(self.p)[0,0])
        return g,4*g.dot(self.p)[0,0]

    def _zoom(self,alp_l,alp_h,f_l,f_h,gl,max_iters=10):
        f_zed = self.acse.e_k
        g_zed = np.dot(self.g0.T,self.p)[0,0] #
        done = False
        iters = 0
        while not (done or self.err):
            # choose interpolated alpha_j
            # first, try 
            #
            if alp_l > alp_h:
                sign = -1
            else:
                sign = +1

            if iters==0:
                alp_j = _quadratic_interpolate(alp_l,alp_h,f_l,f_h,gl)
                print('x = [{:.4f},{:.4f}], f = [{:.8f},{:.8f}], g({:.4f})={:.8f}'.format(alp_l,alp_h,f_l,f_h,alp_l,gl))
            else:
                alp_j = _cubic_interpolate(alp_l,alp_h,alp_m,f_l,f_h,f_m,gl)
                print('x = [{:.4f},{:.4f},{:.4f}], f = [{:.8f},{:.8f},{:.8f}], g({:.4f})={:.8f}'.format(alp_l,alp_m,alp_h,f_l,f_m,f_h,alp_l,gl))
            print('alp_inter = {}'.format(alp_j))
            ch1 = sign*alp_j>sign*alp_h or sign*alp_j<alp_l*sign
            ch2 = abs(alp_l-alp_j)<(0.1*(alp_h-alp_l))
            ch3 = abs(alp_h-alp_j)<(0.1*(alp_h-alp_l))
            if ch3:
                alp_j = (1/4)*(1*alp_l+3*alp_h)
                print('alp_inter too close to alp_h, bisecting [alp_l,alp_h]')
                print('alp_hbi = {}'.format(alp_j))
            elif ch2:
                alp_j = (1/4)*(3*alp_l+1*alp_h)
                print('alp_inter too close to alp_l, bisecting [alp_l,alp_h]')
                print('alp_lbi = {}'.format(alp_j))
            elif ch1:
                alp_j = (1/2)*(1*alp_l+1*alp_h)
                print('alp_inter outside of bracketed region, bisecting [alp_l,alp_h]')
                print('alp_bi = {}'.format(alp_j))

            f_j = self.phi(alp_j)
            print('alp_inter -> {:.6f}, f(alp_inter) = {:.6f}'.format(alp_j,f_j))
            # now, check sufficent decrease 
            if f_j > f_zed + self.c1*alp_j*g_zed or f_j >= f_l:
                # i.e., we have not met sufficient decrease
                # replace high with j
                # shrink [l,h] -> [l,j]
                alp_m = copy(alp_h)
                f_m = copy(f_h)

                alp_h = copy(alp_j)
                f_h = copy(f_j)
                print('shrinking alpha_h -> {}'.format(alp_j))
            else:
                # we have met sufficient decrease
                g_j,gj = self.dphi(alp_j)
                if abs(gj)<= -self.c2* g_zed: 
                    done = True
                    alp_star = alp_j
                if gj*(alp_h-alp_l)>=0:
                    #
                    alp_m = copy(alp_h)
                    f_m = copy(f_h)

                    alp_h = copy(alp_l)
                    f_h = copy(f_l)
                else:
                    alp_m = copy(alp_l)
                    f_m = copy(f_l)
                # 
                alp_l = copy(alp_j)
                f_l = copy(f_j)
                g_l = copy(g_j)
                gl = copy(gj)
                print('increasing alpha_l -> {}'.format(alp_j))
            iters+=1
            if iters>max_iters:
                done = True
                self.err =True
                warn('max iterations reached, returning best result')
        print('Returning current best estimate...')
        if self.err:
            try:
                g_j
            except UnboundLocalError:
                g_j,gj = self.dphi(alp_j)
            if (f_j<= f_l) and (f_j <= f_h):
                return alp_j,f_j,g_j
            elif (f_l<=f_j) and (f_l<=f_h) and not alp_l==0:
                self.phi(alp_l)
                g_l, gl = self.dphi(alp_l)
                return alp_l,f_l,g_l
            elif (f_h<=f_j) and (f_h<=f_l):
                self.phi(alp_h)
                g_h, gh = self.dphi(alp_h)
                return alp_h,f_h,g_h
            else:
                return alp_j,f_j,g_j
        else:
            return alp_star,f_j,g_j

    def _line_search_nocedal(self,**kwargs):
        f_zed = np.real(self.acse.e_k)
        g_zed = 4*np.dot(self.g0.T,self.p)[0,0] #
        alp_0,alp_max = 0,25
        alp_1 = 1
        done = False
        iters = 1
        f0 = copy(f_zed) #actual alp=0, not alp=alp_0
        g0 = copy(g_zed) #same
        #fepsilon = self.phi(0.01) #evaluate phi(alp_i) 
        #print('Test')
        #print(self.c1)
        #print(fepsilon)
        #print(f0+0.01*g0)
        while not done:
            print('wolfe 1')
            f1 = self.phi(alp_1) #evaluate phi(alp_i) 
            print(f1)
            print(f_zed+self.c1*alp_1*g_zed)
            if f1>f_zed+self.c1*alp_1*g_zed or (f1>= f0 and iters>1):  
                print('bracketing region')
                # i.e., we have found a bracketing region for alp_i 
                # the decrase condition is not met
                alp_star,f_star,g_star = self._zoom(alp_0,alp_1,f0,f1,g0)
                gs = 4*np.dot(g_star,self.p)[0,0]
                done = True
                continue
            g_1,g1 = self.dphi(alp_1) #evaluate g(alph_i) 
            if abs(g1)<= -self.c2*g_zed: #
                print(g1,g_zed)
                print('test ')
                alp_star = alp_1
                f_star = f1
                g_star = g_1
                gs = 4*np.dot(g_star,self.p)[0,0]
                done = True
                continue
            if g1>= 0:
                print('')
                alp_star,f_star,g_star = self._zoom(alp_0,alp_1,f0,f1,g0)
                gs = 4*np.dot(g_star,self.p)[0,0]
                done = True
                continue
            #
            alp_0 = copy(alp_1)
            f0 = copy(f1)

            alp_1 = min(2*alp_1,alp_max)
            g0 = copy(g1)
            iters+=1
            if iters>10:
                done = True
                warn('max iterations')
                self.err= True
                f_star= f1
                g_star = copy(g_1)
                alp_star = alp_1
                gs = 4*np.dot(g_star,self.p)[0,0]
        w1l,w1r = np.real(f_star),np.real(f_zed+self.c1*alp_star*g_zed)
        w2l,w2r = np.real(gs),np.real(g_zed*self.c2)
        w1 = w1l<=w1r
        w2 = w2l>=w2r
        w3 = abs(w2l)<=abs(w2r)
        if self.acse._output==3:
            print('-- Wolfe conditions --  ')
            print('Decrease : {:+.6f} <= {:+.6f}, {}'.format(w1l,w1r,w1l<=w1r))
            print('Curvature: {:+.6f} >= {:+.6f}, {}'.format(w2l,w2r,w2l>=w2r))
            print('Strong Curvature: {:+.6f} >= {:+.6f}, {}'.format(
                abs(w2l),abs(w2r),w3))
        if (w1 and w2 and self.err):
            print('Despite error, Wolfe criteria still met.')
            self.err = False
        self.alp_star = np.real(alp_star)
        self.f_star = np.real(f_star)
        self.g_star = np.real(g_star)
        print('f_calls = ({}),g_calls = ({}),alpha = {}'.format(self.f_evals,self.g_evals,self.alp_star))
        print('alp!: ',self.acse.Store.alp)

