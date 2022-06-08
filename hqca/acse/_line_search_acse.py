'''
Linesearch that attempts to satisy the Wolfe conditions.

See Nocedal & Wright, chapter 3
'''

from copy import deepcopy as copy
import numpy as np
from hqca.operators import *
from hqca.acse._quant_A_acse import solveqACSE
from hqca.acse._class_A_acse import solvecACSE
from hqca.acse._qubit_A import solvepACSE
from hqca.acse._tools_acse import vector_to_operator
from hqca.core import *
from warnings import warn




class LineSearchACSE:
    def __init__(self,acse,p=None):
        ''' Performs a line search given the ACSE and a search direction

        Wrapper for the line search algorithm, which is suitable for 
        ACSE-based CQEs.
        '''
        self.acse = acse
        self.fermi = acse.acse_update in ['q','c']
        self.keys = acse.rdme#+acse.rdme+acse.rdme+acse.rdme
        self.user = False
        self.f_evals = 0
        self.g_evals = 0
        self.N = acse.qs.dim
        self.S_min = acse.S_min
        self.err = False
        self.symm_factor = 4
        #
        self.best_en = 0
        self.best_rdm = None
        self.best_alp = 1
        self.p = p

    def run(self,p,g_0=None,c1=1e-4,c2=0.9,
            line_search='nw',**kwargs):
        g0  = self.symm_factor*g_0.T.dot(p)[0,0]
        self.p = p

        if line_search=='nw':
            alp,fs,gs,err,wolfe = line_search_nocedal(
                    self.phi,
                    self.dphi,
                    c1=c1,c2=c2,
                    f0=self.acse.e_k,
                    g_0=g_0,
                    g0=g0,
                    verbose=self.acse.verbose,
                    **kwargs
                    )
        else:
            raise Exception
        if not self.best_alp==alp:
            if err:
                pass
            else:
                print('Best alp: ')
                print(self.best_alp)
                print('Given alp: ')
                print(alp)
                err = True
                #raise Exception('alp not established properly. ')
        self.wolfe_1 = wolfe[0]
        self.wolfe_2 = wolfe[1]
        self.wolfe_3 = wolfe[2]
        self.alp_star = alp
        self.f_star = fs
        self.g_star = gs
        self.err = err
        self.acse.store.update(self.best_rdm)
        self.acse.circ = self.best_circ  #

    def phi(self,alp):
        op = vector_to_operator(
                alp*self.p,
                fermi=self.fermi,
                keys=self.keys,
                N=self.N,
                S_min = self.S_min,
                )
        #P =op.transform(self.acse.transform_psi)
        P = op
        psi =  copy(self.acse.psi)+P
        circ = self.acse._generate_circuit(psi)
        en = np.real(self.acse.store.evaluate(circ.rdm))
        #
        if en < self.best_en:
            self.rho = circ.rho
            self.best_en = copy(en)
            self.best_circ = circ
            self.best_rdm = circ.rdm
            self.best_alp = copy(alp)
        self.last_alp = copy(alp)
        print('alpha: {}, energy: {}'.format(alp,en))
        self.f_evals+=1
        return np.real(en)

    def dphi(self,alp):
        op = vector_to_operator(
                alp*self.p,
                fermi=self.fermi,
                keys=self.keys,
                N=self.N,
                S_min = self.S_min,
                )
        #P =op.transform(self.acse.transform_psi)
        P = op
        psi =  copy(self.acse.psi)+P
        if self.acse.acse_update in ['c']:
            G = solvecACSE(
                self.acse,
                operator=psi,
                S_min=self.S_min,
                verbose=self.acse.verbose,
                matrix=True,
                tomo = self.acse.tomo_A,
                keys=self.keys,
            )
        elif self.acse.acse_update in ['q']:
            if type(self.acse.sep_hamiltonian)==type(None):
                H = self.acse.store.H.qubit_operator
            else:
                H = self.acse.sep_hamiltonian
            G = solveqACSE(
                self.acse,
                H=H,
                operator             =psi,
                process              =self.acse.process,
                instruct             =self.acse.ins,
                store                =self.acse.store,
                quantstore           =self.acse.qs,
                S_min                =self.acse.S_min,
                hamiltonian_step_size=self.acse.hamiltonian_step_size,
                expiH_approx         =self.acse.S_expiH_approx,
                verbose              =self.acse.verbose,
                tomo                 =self.acse.tomo_A,
                matrix               =self.acse._A_as_matrix,
            )
        elif self.acse.acse_update in ['p']:
            if type(self.acse.sep_hamiltonian)==type(None):
                H = self.acse.store.H.qubit_operator
            else:
                H = self.acse.sep_hamiltonian
            G = solvepACSE(
                self.acse,
                H=H,
                operator             =psi,
                process              =self.acse.process,
                instruct             =self.acse.ins,
                store                =self.acse.store,
                quantstore           =self.acse.qs,
                S_min                =self.acse.S_min,
                hamiltonian_step_size=self.acse.hamiltonian_step_size,
                expiH_approx         =self.acse.S_expiH_approx,
                verbose              =self.acse.verbose,
                tomo                 =self.acse.tomo_A,
                matrix               =self.acse._A_as_matrix,
            )
        else:
            raise QuantumRunError
        g = 1*np.asmatrix(G)
        self.g_evals+=1
        return g,self.symm_factor*g.dot(self.p)[0,0]

def line_search_nocedal(phi,dphi,
        c1=1e-4,c2=0.9,
        max_iters=10,zoom_iters=10,
        alp_1=1,alp_max = 25,
        f0=None,g0=None,g_0=None,
        verbose=False):
    ''' line search for Wolfe conditions, modified from Nocedal 

    Arguments:
        phi
        dphi

    '''

    def _zoom(alp_l,alp_h,f_l,f_h,gl):

        done,err,iters = False,False,0
        while not (done or err):
            # choose interpolated alpha_j
            #
            if alp_l > alp_h: # just in case l,h get switched
                sign = -1
            else:
                sign = +1
            if iters==0: # using 2 points, perform quadratic interpolation
                alp_j = _quadratic_interpolate(alp_l,alp_h,f_l,f_h,gl)
                print('x = [{:.4f},{:.4f}], f = [{:.8f},{:.8f}], g({:.4f})={:.8f}'.format(alp_l,alp_h,f_l,f_h,alp_l,gl))
            else: #using the mid point, perform cubic interpolation
                alp_j = _cubic_interpolate(alp_l,alp_h,alp_m,f_l,f_h,f_m,gl)
                print('x = [{:.4f},{:.4f},{:.4f}], f = [{:.8f},{:.8f},{:.8f}], g({:.4f})={:.8f}'.format(alp_l,alp_m,alp_h,f_l,f_m,f_h,alp_l,gl))
            print('alp_inter = {}'.format(alp_j))
            ch1 = sign*alp_j>sign*alp_h or sign*alp_j<alp_l*sign
            ch2 = abs(alp_l-alp_j)<(0.05*(alp_h-alp_l)) # interpolated result too close  to h, l boudnary
            ch3 = abs(alp_h-alp_j)<(0.05*(alp_h-alp_l))
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

            f_j = phi(alp_j)
            print('alp_inter -> {:.6f}, f(alp_inter) = {:.6f}'.format(alp_j,f_j))
            # now, check sufficent decrease 
            if f_j > f_zed + c1*alp_j*gzed or f_j >= f_l:
                # i.e., we have not met sufficient decrease
                # replace hi with j
                # shrink [l,h] -> [l,j]
                alp_m = copy(alp_h)
                f_m = copy(f_h)

                alp_h = copy(alp_j)
                f_h = copy(f_j)
                print('shrinking alpha_h -> {}'.format(alp_j))
            else:
                # we have met sufficient decrease
                # j -> l
                g_j,gj = dphi(alp_j)

                if abs(gj)<= -c2*gzed:
                    done = True
                    alp_star = copy(alp_j)
                if gj*(alp_h-alp_l)>=0:
                    # gradient in the wrong direction
                    # h -> m
                    # l -> h
                    alp_m = copy(alp_h)
                    f_m = copy(f_h)

                    alp_h = copy(alp_l)
                    f_h = copy(f_l)
                else:
                    # l -> m
                    alp_m = copy(alp_l)
                    f_m = copy(f_l)
                # 
                # j -> l
                alp_l = copy(alp_j)
                f_l = copy(f_j)
                g_l = copy(g_j)
                gl = copy(gj)
                print('increasing alpha_l -> {}'.format(alp_j))
            iters+=1
            if iters>max_iters:
                done = True
                err =True
                warn('max iterations reached, returning best result')
        print('Returning current best estimate...')
        if err:
            try:
                g_j
            except UnboundLocalError:
                g_j,gj = dphi(alp_j)
            if (f_j<= f_l) and (f_j <= f_h):
                return alp_j,f_j,g_j,gj,err
            elif (f_l<=f_j) and (f_l<=f_h) and not alp_l==0:
                g_l, gl = dphi(alp_l)
                return alp_l,f_l,g_l,gl,err
            elif (f_h<=f_j) and (f_h<=f_l):
                g_h, gh = dphi(alp_h)
                return alp_h,f_h,g_h,gh,err
            else:
                return alp_j,f_j,g_j,gj,err
        else:
            return alp_star,f_j,g_j,gj,err

    def _quadratic_interpolate(a_l,a_h,f_l,f_h,g_l):
        ''' quadratic interpolation
    
         from f(p) = alp*(p-l)^2 + bet*(p-l)+gam
              g(p) = 2*alp*(p-l) + bet
         uses f(l), g(l), f(h) to form interpolation
    
        follows scipy implementation
        '''
        Del = a_h-a_l
        alpha = (Del**-2)*(f_h - g_l*Del-f_l)
        beta = g_l
        gamma = f_l
        x = - g_l / (2*alpha) + a_l
        delta = a_l
        return np.real(x)

    def _cubic_interpolate(a_l,a_h,a_m,f_l,f_h,f_m,g_l):
        ''' cubic interpolation

         similar to quadratic interpolation except we
         now need to solve a linear system equations
         f(p) = alp*(p-l)^3 + bet*(p-l)^2 + gam*(p-l) + delta
         g(p) = 3*alp*(p-l)^2 + 2*bet*(p-l) + gam

        [[f(h)-g(l)*(h-l)-f(l)] =[[ (h-l)^3 (h-l)^2 ] [[alp]
         [f(m)-g(l)*(m-l)-f(l)]]= [ (m-l)^3 (m-l)^2 ]] [bet]]

        follows scipy implementation
        '''
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
            #print('Cubic alp: {}'.format(np.real(alp_p)))
            return np.real(alp_p)
        else:
            print('Cubic alp: {}'.format(np.real(alp_m)))
            return np.real(alp_m)

    if type(f0)==type(None):
        f_zed = phi(0)
    else:
        f_zed = copy(f0)
    if type(g0)==type(None) or type(g_0)==type(None):
        g_zed,gzed = dphi(0)
    else:
        gzed,g_zed = g0,g_0
    #
    #
    alp_0 = 0
    done = False
    iters = 1
    err = False
    f0 = copy(f_zed) #actual alp=0, not alp=alp_0
    g0 = copy(gzed) #same
    #
    while not done:
        f1 = phi(alp_1) #evaluate phi(alp_i) 

        if f1>f_zed+c1*alp_1*gzed or (f1>= f0 and iters>1):  # phi is accetable, or we are past sufficient decrease
            print('bracketing region')
            # i.e., we have found a bracketing region for alp_i 
            # the decrase condition is not met
            alp_star,f_star,g_star,gs,err = _zoom(alp_0,alp_1,f0,f1,g0)
            done = True
            continue
        g_1,g1 = dphi(alp_1)
        if abs(g1)<= -c2*gzed: #
            alp_star = alp_1
            f_star = f1
            g_star,gs = g_1,g1
            done = True
            continue
        if g1>= 0: #gradient is acceptable
            alp_star,f_star,g_star,gs,err = _zoom(alp_0,alp_1,f0,f1,g0)
            done = True
            continue
        #
        alp_0 = copy(alp_1)
        f0 = copy(f1)

        alp_1 = min(2*alp_1,alp_max)
        g0 = copy(g1)
        iters+=1

        if iters>max_iters:
            done = True
            warn('max iterations')
            err= True
            f_star= f1
            g_star,gs = copy(g_1),copy(g1)
            alp_star = alp_1
    w1l,w1r = np.real(f_star),np.real(f_zed+c1*alp_star*gzed)
    w2l,w2r = np.real(gs),np.real(gzed*c2)
    w1 = w1l<=w1r
    w2 = w2l>=w2r
    w3 = abs(w2l)<=abs(w2r)
    print('-- Wolfe conditions --  ')
    print('Decrease : {:+.6f} <= {:+.6f}, {}'.format(w1l,w1r,w1l<=w1r))
    print('Curvature: {:+.6f} >= {:+.6f}, {}'.format(w2l,w2r,w2l>=w2r))
    print('Strong Curvature: |{:+.6f}| >= |{:+.6f}|, {}'.format(
        abs(w2l),abs(w2r),w3))
    if (w1 and w2 and err):
        print('Despite error, Wolfe criteria still met.')
        err = False
    return np.real(alp_star), np.real(f_star),np.real(g_star),err,(w1,w2,w3)
    #
