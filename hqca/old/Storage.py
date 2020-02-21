import numpy as np
np.set_printoptions(precision=4,suppress=True)
import sys
from functools import reduce,partial
from hqca.tools import RDMFunctions as rdmf
from hqca.tools import Chem as chem
from hqca.tools import EnergyFunctions as enf
from hqca.tools import EnergyDeterminant as end
from hqca.tools import EnergyOrbital as eno
from pyscf import gto,mcscf
from pyscf import scf as pscf

'''
./tools/EnergyFunctions.py

Module for optimizer or energy functions, i.e. for rotations, etc.

'''

def find_function(
        run_type,
        spec,
        Store,
        QuantStore):
    if run_type=='noft':
        if spec=='qc':
            f = partial(
                    end.energy_eval_nordm,
                    **{
                        'Store':Store,
                        'QuantStore':QuantStore
                        }
                    )
        if spec=='orb':
            f = partial(
                    eno.energy_eval_orbitals,
                    **{
                        'Store':Store,
                        'QuantStore':QuantStore
                        }
                    )
        elif spec=='orb_grad':
            f = partial(
                    #eno.orbital_energy_gradient_givens,
                    eno.orbital_en_grad_numerical,
                    **{
                        'Store':Store,
                        'QuantStore':QuantStore
                        }
                    )
        if spec=='noft_grad':
            f = partial(
                    end.energy_eval_grad_noft_numerical,
                    **{
                        'Store':Store,
                        'QuantStore':QuantStore
                        }
                    )
    elif run_type=='rdm':
        f = partial(
            end.energy_eval_rdm,
            **{'Store':Store,'QuantStore':QuantStore}
            )
    return f

class Storage:
    '''
    Class for storing energetic properties -
    Basically, will store certain properties of the best 2RDM and wavefunction
    without having to return it for every function call.

    Also, will initiate the chemical procedure

    Also stores some basic properties of the molecule. However, does not hold
    properties related to the quantum optimization. Need to generate a
    QuantumStorage object to do that.
    '''
    def __init__(self,
            mol,
            Ne_as='default',
            No_as='default',
            casci=False,
            **kwargs):
        self.S = mol.intor('int1e_ovlp')
        self.T_1e = mol.intor('int1e_kin')
        self.V_1e = mol.intor('int1e_nuc')
        self.ints_1e_ao = self.V_1e+self.T_1e
        self.ints_2e_ao = mol.intor('int2e')
        try:
            self.hf = pscf.RHF(mol)
            self.hf.kernel()
            self.mol = mol
            self.hf.analyze()
            self.C= self.hf.mo_coeff
            self.f = self.hf.get_fock()
        except Exception:
            self.hf = pscf.ROHF(mol)
            self.hf.kernel()
            self.mol = mol
            self.hf.analyze()
            self.C= self.hf.mo_coeff
            self.f = self.hf.get_fock()
        if Ne_as=='default':
            self.Ne_as = mol.nelec[0]+mol.nelec[1]
        else:
            self.Ne_as = int(Ne_as)
        self.Ne_tot = mol.nelec[0]+mol.nelec[1]
        self.Ne_alp = mol.nelec[0]
        self.Ne_bet = mol.nelec[1]
        if No_as=='default':
            self.No_as = self.C.shape[0]
        else:
            self.No_as = int(No_as)
        self.No_tot = self.C.shape[0]
        if casci:
            self.mc = mcscf.CASCI(
                    self.hf,
                    self.No_as,
                    self.Ne_as)
            self.mc.kernel()
            self.e_casci  = self.mc.e_tot
            self.mc_coeff = self.mc.mo_coeff
        else:
            self.mc = None
        print('Hartree-Fock Energy: {:.8f}'.format(float(self.hf.e_tot)))
        print('CASCI Energy: {:.8f}'.format(float(self.e_casci)))
        self.E_ne = self.mol.energy_nuc()
        self.energy_best = 0
        self.energy_wf = 0
        self.energy_int = 0
        self.wf = {}
        self.rdm2=None
        self.Ci = np.linalg.inv(self.C)
        self.ints_1e =None
        self.ints_2e = None
        self.T_alpha = self.C.copy()
        self.T_beta  = self.C.copy()
        self.opt_T_alpha = None
        self.opt_T_beta = None
        self.opt_done = False
        self.error = False
        self.occ_energy_calls = 0
        self.orb_energy_calls = 0
        self.active_space_calc='FCI'
        self.F_alpha = 0
        self.F_beta = 0
        self.spin = self.mol.spin
        self.kw = kwargs

    def gip(self):
        '''
        'Get Initial Parameters (GIP) function.
        '''
        try:
            if self.sp=='noft':
                self.parameters=[0,0]
            elif self.sp=='rdm':
                if self.kw['spin_mapping'] in ['default','alternating']:
                    Na = 1
                    Nb = 1
                elif self.kw['spin_mapping']=='spin-free':
                    Na = 2
                    Nb = 0
                elif self.kw['spin_mapping']=='spatial':
                    Na = 1
                    Nb = 0
                if self.kw['entangled_pairs']=='full':
                    N = 0.5*((self.No_as*Na)**2-self.No_as*Na)
                    N += 0.5*((self.No_as*Nb)**2-self.No_as*Nb)
                elif self.kw['entangled_pairs']=='sequential':
                    N = self.No_as-1
                elif self.kw['entangled_pairs']=='specified':
                    pass
                self.parameters = [0 for i in range(int(N))]
        except AttributeError:
            print('Not assigned.')

    def gsm(self):
        self._generate_spin2spac_mapping()

    def gas(self):
        self._generate_active_space(**self.kw)

    def _generate_active_space(self,
            spin_mapping='default',
            **kw
            ):
        '''
        Note, all orb references are in spatial orbitals. 
        '''
        self.alpha_mo={
                'inactive':[],
                'active':[],
                'virtual':[],
                'qc':[]
                }
        self.beta_mo={
                'inactive':[],
                'active':[],
                'virtual':[],
                'qc':[]
                }
        self.Ne_ia = self.Ne_tot-self.Ne_as
        self.No_ia = self.Ne_ia//2
        self.spin = spin_mapping
        self.No_v  = self.No_tot-self.No_ia-self.No_as
        if self.Ne_ia%2==1:
            raise(SpinError)
        if self.Ne_ia>0:
            self.active_space_calc='CASSCF'
        ind=0
        for i in range(0,self.No_ia):
            self.alpha_mo['inactive'].append(ind)
            ind+=1
        for i in range(0,self.No_as):
            self.alpha_mo['active'].append(ind)
            ind+=1
        for i in range(0,self.No_v):
            self.alpha_mo['virtual'].append(ind)
            ind+=1
        for i in range(0,self.No_ia):
            self.beta_mo['inactive'].append(ind)
            ind+=1
        for i in range(0,self.No_as):
            self.beta_mo['active'].append(ind)
            ind+=1
        for i in range(0,self.No_v):
            self.beta_mo['virtual'].append(ind)
            ind+=1

    def _generate_spin2spac_mapping(self):
        self.s2s = {}
        for i in range(0,self.No_tot):
            self.s2s[i]=i
        for i in range(self.No_tot,2*self.No_tot):
            self.s2s[i]=i-self.No_tot

    def opt_update_wf(self,energy,wf,para):
        if energy<self.energy_wf:
            if energy<self.energy_best:
                self.energy_best = energy
            self.energy_wf  = energy
            self.parameters  = para
            self.wf = wf

    def opt_update_rdm2(self,energy,rdm2,para):
        if energy<self.energy_wf:
            if energy<self.energy_best:
                self.energy_best = energy
            self.energy_wf  = energy
            self.parameters  = para
            self.rdm2 = fx.expand(rdm2)

    def update_rdm2(self):
        self.rdm2 = rdmf.build_2rdm(
                self.wf,
                self.alpha_mo,
                self.beta_mo,
                region='full')
        self.rdm1 = rdmf.check_2rdm(
                self.rdm2,self.Ne_tot)
        self.rdm2 = fx.contract(self.rdm2)

    def update_full_ints(self):
        try:
            self.T_alpha_old = self.T_alpha.copy()
            self.T_beta_old = self.T_beta.copy()
            self.T_alpha = self.opt_T_alpha.copy()
            self.T_beta  = self.opt_T_beta.copy()
        except Exception as e:
            pass
        self.ints_1e = chem.gen_spin_1ei(
                self.ints_1e_ao.copy(),
                self.T_alpha.T,
                self.T_beta.T,
                self.alpha_mo,
                self.beta_mo,
                region='full',
                spin2spac=self.s2s
                )
        self.ints_2e = chem.gen_spin_2ei(
                self.ints_2e_ao.copy(),
                self.T_alpha.T,
                self.T_beta.T,
                self.alpha_mo,
                self.beta_mo,
                region='full',
                spin2spac=self.s2s
                )
        self.ints_2e = fx.contract(self.ints_2e)
        # now, determining trace fidelity
        self.F_alpha = abs((reduce(np.dot, (
                self.T_alpha_old.T,
                self.Ci.T,
                self.Ci,
                self.T_alpha
                )
            ).trace()))*(1/len(self.T_alpha))
        self.F_beta = abs((reduce(np.dot, (
                self.T_beta_old.T,
                self.Ci.T,
                self.Ci,
                self.T_beta
                )
            ).trace()))*(1/len(self.T_beta))
        if self.F_alpha-1>1e-8:
            print('Error in fidelity:')
            print(self.T_alpha_old.T)
            print(self.S)
            print(self.T_alpha)
            print(self.F_alpha)
        if self.F_beta>1:
            print('Error in fidelity:')
            print(self.T_beta_old)
            print(self.S)
            print(self.T_beta.T)
            print(reduce(np.dot,(
                self.T_beta_old.T,
                self.S,
                self.T_beta)))
            print(reduce(np.dot,(
                self.T_beta_old,
                self.S,
                self.T_beta.T)))
            print(self.F_beta)



    def opt_update_int(self,para,energy,U_a,U_b):
        '''
        Basically, always update the energy after finding the next best step in
        an orbital optimization.
        '''
        if energy<self.energy_int:
            if energy<self.energy_best:
                self.energy_best = energy
            self.energy_int  = energy
            self.opt_T_alpha = U_a
            self.opt_T_beta  = U_b
            self.opt_para = para

    def opt_analysis(self):
        print('  --  --  --  --  --  --  -- ')
        print('--  --  --  --  --  --  --  --')
        print('E, scf: {:.9f} H'.format(self.hf.e_tot))
        print('E, run: {:.9f} H'.format(self.energy_best))
        try:
            diff = 1000*( self.energy_best-self.e_casci)
            print('E, fci: {:.9f} H'.format(self.e_casci))
            print('Energy difference from FCI: {:.8f} mH'.format(diff))
        except KeyError:
            pass
        rdm1 = rdmf.check_2rdm(fx.expand(self.rdm2),self.Ne_tot)
        print('Occupations of the 1-RDM:')
        print(np.real(np.diag(rdm1)))
        print('Natural orbital wavefunction:')
        for k,v in self.wf.items():
            print(' |{}>: {}'.format(k,v))
        self.Ci = np.linalg.inv(self.C)
        self.Ti_a = np.linalg.inv(self.T_alpha)
        self.Ti_b = np.linalg.inv(self.T_beta)
        Ni_a = reduce(
                np.dot, (
                    self.Ti_a,self.C)
                )
        Ni_b = reduce(
                np.dot, (
                    self.Ti_b,self.C)
                )
        rdm2_mo = rdmf.rotate_2rdm(
                fx.expand(self.rdm2),
                Ni_a.T,
                Ni_b.T,
                self.alpha_mo,
                self.beta_mo,
                spin2spac=self.s2s,
                region='full'
                )
        rdm1_mo = rdmf.check_2rdm(
                rdm2_mo,
                self.Ne_tot)
        print('1-RDM in the molecular orbital basis.')
        print(np.real(rdm1_mo))
        print('NO coefficients (in terms of MO):')
        print('Alpha: ')
        print(np.real(Ni_a.T))
        print('Beta: ')
        print(np.real(Ni_b.T))
        print('NO coefficients (in terms of AO):')
        print('Alpha: ')
        print(np.real(self.T_alpha))
        print('Beta: ')
        print(np.real(self.T_beta))
        sz = rdmf.Sz(
                rdm1_mo,
                self.alpha_mo,
                self.beta_mo,
                s2s=self.s2s)
        s2 = rdmf.S2(
                rdm2_mo,
                rdm1_mo,
                self.alpha_mo,
                self.beta_mo,
                s2s=self.s2s)
        print('1-RDM in the Lowdin atomic orbital basis.')
        #print('MO coefficients: ')
        rdm1_mo_a = rdm1_mo[0:self.No_tot,0:self.No_tot]
        rdm1_mo_b = rdm1_mo[self.No_tot:,self.No_tot:]
        print('Alpha:')
        print(np.real(reduce(np.dot, (self.C,rdm1_mo_a,self.Ci))))
        print('Beta:')
        print(np.real(reduce(np.dot, (self.C,rdm1_mo_b,self.Ci))))
        print('Sz value: {:.6f}'.format(np.real(sz)))
        print('S2 value: {:.6f}'.format(np.real(s2)))
        print('--  --  --  --  --  --  --  --')
        print('  --  --  --  --  --  --  --')

    def check(self,crit,Occ,Orb,print_run=False):
        print('Checking orbital and occupation energies for convergence...')
        if print_run:
            print('Best Energy: {}'.format(self.energy_best))
            print('Best energy from wf: {}'.format(self.energy_wf))
            print('Best energy from orbitals: {}'.format(self.energy_int))
            print('Parameters: {}'.format(self.parameters))
            print('Wavefunction: {}'.format(self.wf))
        self.energy_best = min(self.energy_wf,self.energy_int)
        if Occ.error:
            self.opt_done=True
            self.error=True
        elif Orb.error:
            self.opt_done=True
            self.error=True
        elif abs(self.energy_int-self.energy_wf)<crit:
            self.opt_done=True
        else:
            diff = abs(self.energy_int-self.energy_wf)*1000
            print('Difference in orbital and occupation energies: {:8f} mH'.format(diff))
        self.energy_wf  = 0
        self.energy_int = 0 

    def update_calls(self,orb,occ):
        self.occ_energy_calls +=  occ
        self.orb_energy_calls +=  orb

    def find_npara_orb(self):
        self.Np_orb = 0
        if self.active_space_calc=='FCI':
            if self.spin in ['default']: #unrestricted
                temp = len(self.alpha_mo['active'])
                self.Np_orb += int(temp*(temp-1)/2)
                temp = len(self.beta_mo['active'])
                self.Np_orb += int(temp*(temp-1)/2)
            elif self.spin=='spatial': #restricted
                temp = len(self.alpha_mo['active'])
                self.Np_orb += int(temp*(temp-1)/2)
        elif self.active_space_calc=='CASSCF':
            sys.exit('Orbital rotations not implemented fully for CASSCF.')

def rotation_parameter_generation(
        spin_mo, # class with inactive, active, and virtual orbitals
        region='active',
        output='matrix',
        para=None
        ):
    '''
    spin_mo - should be a alpha or beta spin dictionary, with active, inactive
    and virtual orbital arrays

    keywords:

    region - 'active', 'active_space', 'as', 'full','occupied'
        -active space rotations
        -every rotation, with the exception of the active space
        -no virtual mixing

    output - 'matrix', 'Npara'
        -outputs the rotation matrix (require para input)
        -otherwise, outputs array with zeros that is the proper length
    '''
    No_tot = len(spin_mo['virtual']+spin_mo['active']+spin_mo['inactive'])
    if output=='matrix':
        rot_mat = np.identity(No_tot)
    count =  0
    hold=[]
    N = No_tot
    if region in ['active','as','active_space']:
        for i in spin_mo['active']:
            for j in spin_mo['active']:
                if i<j:
                    hold.append([i%N,j%N])
    elif region in ['full']:
        for i in spin_mo['inactive']:
            for j in spin_mo['active']:
                hold.append([i%N,j%N])
        for i in spin_mo['inactive']:
            for j in spin_mo['virtual']:
                hold.append([i%N,j%N])
        for i in spin_mo['active']:
            for j in spin_mo['virtual']:
                hold.append([i%N,j%N])
    elif region in ['occupied']:
        for i in spin_mo['inactive']:
            for j in spin_mo['active']:
                hold.append([i%N,j%N])
    else:
        sys.exit('Invalid active space selection.')
    for z in hold:
        if output=='matrix':
            if para[count]==0:
                continue
            temp=np.identity(No_tot)
            c = np.cos(((para[count])))
            s = np.sin(((para[count])))
            temp[z[0],z[0]] = c
            temp[z[1],z[1]] = c
            temp[z[0],z[1]] = -s
            temp[z[1],z[0]] = s
            rot_mat = reduce( np.dot, (rot_mat,temp))
        count+=1 
    if output=='matrix':
        #print(rot_mat)
        return rot_mat
    elif output=='Npara':
        return count



class NotAvailableError(Exception):
    '''
    Means what it says.
    '''
class SpinError(Exception):
    '''
    Wrong active space selection. OR at least, not supported.
    '''

