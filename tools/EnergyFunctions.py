import numpy as np
import sys
from functools import reduce,partial
from hqca.tools import RDMFunctions as rdmf
from hqca.tools import Chem as chem
from hqca.tools import EnergyFunctions as enf
from hqca.tools import EnergyDeterminant as end
from hqca.tools import Functions as fx
from hqca.tools import EnergyOrbital as eno
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
        if spec=='main':
            return end.energy_eval_nordm
        elif spec=='sub':
            return eno.energy_eval_orbitals
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
    without having to return it for every function call

    Also stores some basic properties of the molecule. However, does not hold
    properties related to the quantum optimization. Need to generate a
    QuantumStorage object to do that. 
    '''
    def __init__(self,
            moc_alpha,
            moc_beta,
            ints_1e_ao,
            ints_2e_ao,
            E_ne,
            pr_g,
            **kwargs
            ):
        self.energy_best = 0
        self.energy_wf = 0
        self.energy_int = 0
        self.wf = {}
        self.rdm2=None
        self.ints_1e =None
        self.C_a = moc_alpha
        self.C_b = moc_beta
        self.ints_2e = None
        self.ints_1e_ao = ints_1e_ao
        self.ints_2e_ao = ints_2e_ao
        self.E_ne = E_ne
        self.T_alpha = moc_alpha
        self.pr_g=pr_g
        self.T_beta  = moc_beta
        self.opt_T_alpha = moc_alpha
        self.opt_T_beta  = moc_alpha
        self.opt_done = False
        self.error = False
        self.occ_energy_calls = 0
        self.orb_energy_calls = 0
        self.active_space_calc='FCI'
        self.kw = kwargs

    def gip(self):
        '''
        'Get Initial Parameters (GIP) function.
        '''
        try:
            if self.sp=='noft':
                self.parameters=[0,0]
            elif self.sp=='rdm':
                if self.kw['spin_mapping']=='default':
                    Na = 1
                    Nb = 1
                elif self.kw['spin_mapping']=='spin-free':
                    Na = 2
                    Nb = 0
                elif self.kw['spin_mapping']=='spatial':
                    Na = 1
                    Nb = 0
                if self.kw['entangled_pairs']=='full':
                    N = 0.5*((self.Norb_as*Na)**2-self.Norb_as*Na)
                    N += 0.5*((self.Norb_as*Nb)**2-self.Norb_as*Nb)
                elif self.kw['entangled_pairs']=='sequential':
                    N = self.Norb_as-1
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
            Nels_tot,
            Norb_tot,
            Nels_as,
            Norb_as,
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
        if self.pr_g>0:
            print('Total number of electrons: {}'.format(Nels_tot))
            print('Total number of orbitals: {}'.format(Norb_tot))
            print('Active space electrons: {}'.format(Nels_as))
            print('Active space orbitals: {}'.format(Norb_as))
        self.Nels_tot= Nels_tot
        self.Nels_as = Nels_as
        self.Norb_tot= Norb_tot
        self.Norb_as = Norb_as
        self.Nels_ia = self.Nels_tot-self.Nels_as
        self.Norb_ia = self.Nels_ia//2
        if self.pr_g>0:
            print('Number of inactive orbitals: {}'.format(self.Norb_ia))
        self.Norb_v  = self.Norb_tot-self.Norb_ia-self.Norb_as
        if self.Nels_ia%2==1:
            raise(SpinError)
        if self.Nels_ia>0:
            self.active_space_calc='CASSCF'
        ind=0 
        for i in range(0,self.Norb_ia):
            self.alpha_mo['inactive'].append(ind)
            ind+=1
        for i in range(0,self.Norb_as):
            self.alpha_mo['active'].append(ind)
            ind+=1
        for i in range(0,self.Norb_v):
            self.alpha_mo['virtual'].append(ind)
            ind+=1 
        for i in range(0,self.Norb_ia):
            self.beta_mo['inactive'].append(ind)
            ind+=1
        for i in range(0,self.Norb_as):
            self.beta_mo['active'].append(ind)
            ind+=1
        for i in range(0,self.Norb_v):
            self.beta_mo['virtual'].append(ind)
            ind+=1
        if spin_mapping=='default':
            self.alpha_mo['qc']=self.alpha_mo['active'].copy()
            self.beta_mo['qc']=self.beta_mo['active'].copy()
        elif spin_mapping=='spin-free':
            self.alpha_mo['qc']=self.alpha_mo['active']+self.beta_mo['active']
        elif spin_mapping=='spatial':
            self.alpha_mo['qc']=self.alpha_mo['active'].copy()
            self.beta_mo['qc']=self.beta_mo['active'].copy()

    def _generate_spin2spac_mapping(self):
        self.s2s = {}
        for i in range(0,self.Norb_tot):
            self.s2s[i]=i
        for i in range(self.Norb_tot,2*self.Norb_tot):
            self.s2s[i]=i-self.Norb_tot

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

    def update_full_ints(self):
        self.T_alpha = self.opt_T_alpha.copy()
        self.T_beta  = self.opt_T_beta.copy()
        self.opt_T_alpha = None
        self.opt_T_beta = None
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
        self.ints_2e = np.reshape(
                self.ints_2e,
                (
                    (2*self.Norb_tot)**2,
                    (2*self.Norb_tot)**2
                    )
                )

    def opt_update_int(self,energy,U_a,U_b):
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

    def opt_analysis(self):
        print('  --  --  --  --  --  --  -- ')
        print('--  --  --  --  --  --  --  --')
        print('Analyzing the optimization.')
        try:
            diff = 1000*( self.energy_best-self.kw['e_fci'])
            print('Energy difference from FCI: {:.8f} mH'.format(diff))
        except KeyError:
            pass
        rdm1 = rdmf.check_2rdm(self.rdm2,self.Nels_tot)
        print('Occupations of the 1-RDM:')
        print(np.real(np.diag(rdm1)))
        print('Natural orbital wavefunction:')
        for k,v in self.wf.items():
            print(' |{}>: {}'.format(k,v))
        self.Ci_a = np.linalg.inv(self.C_a)
        self.Ci_b = np.linalg.inv(self.C_b)
        self.Ti_a = np.linalg.inv(self.T_alpha)
        self.Ti_b = np.linalg.inv(self.T_beta)
        Ni_a = reduce(
                np.dot, (
                    self.Ti_a,self.C_a)
                )
        Ni_b = reduce(
                np.dot, (
                    self.Ti_b,self.C_b)
                )
        rdm2_mo = rdmf.rotate_2rdm(
                self.rdm2,
                Ni_a.T,
                Ni_b.T,
                self.alpha_mo,
                self.beta_mo,
                spin2spac=self.s2s,
                region='full'
                )
        rdm1_mo = rdmf.check_2rdm(
                rdm2_mo,
                self.Nels_tot)
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
        #print(self.C_a)
        rdm1_mo_a = rdm1_mo[0:self.Norb_tot,0:self.Norb_tot]
        rdm1_mo_b = rdm1_mo[self.Norb_tot:,self.Norb_tot:]
        print('Alpha:')
        print(np.real(reduce(np.dot, (self.C_a,rdm1_mo_a,self.Ci_a))))
        print('Beta:')
        print(np.real(reduce(np.dot, (self.C_b,rdm1_mo_b,self.Ci_b))))
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

    def update_fci(self,energy,ints_1e_no,ints_2e_no):
        self.fci = energy
        self.energy_int = energy
        self.ints_1e = ints_1e_no
        self.ints_2e = ints_2e_no

    def update_calls(self,orb,occ):
        self.occ_energy_calls +=  occ
        self.orb_energy_calls +=  orb


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
    Norb_tot = len(spin_mo['virtual']+spin_mo['active']+spin_mo['inactive'])
    if output=='matrix':
        rot_mat = np.identity(Norb_tot)
    count =  0
    hold=[]
    N = Norb_tot
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
            temp=np.identity(Norb_tot)
            c = np.cos((np.pi*(para[count])/180))
            s = np.sin((np.pi*(para[count])/180))
            temp[z[0],z[0]] = c
            temp[z[1],z[1]] = c
            temp[z[0],z[1]] = -s
            temp[z[1],z[0]] = s
            rot_mat = reduce( np.dot, (temp,rot_mat))
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


