import numpy as np
import sys
from functools import reduce
from tools import RDMFunctions as rdmf
from tools import Chem as chem

'''
./tools/EnergyFunctions.py

Module for optimizer or energy functions, i.e. for rotations, etc.

'''
class NotAvailableError(Exception):
    '''
    Means what it says.
    '''
class SpinError(Exception):
    '''
    Wrong active space selection. OR at least, not supported.
    '''

class Storage:
    '''
    Class for storing energetic properties -
    Basically, will store certain properties of the best 2RDM and wavefunction
    without having to return it for every function call
    '''
    def __init__(self,
            moc_alpha,
            moc_beta,
            ints_1e_ao,
            ints_2e_ao,
            E_ne,
            **kwargs
            ):
        self.energy_best = 0
        self.energy_wf = 0
        self.energy_int = 0
        self.wf = {}
        self.rdm2=None
        self.ints_1e =None
        self.ints_2e = None
        self.ints_1e_ao = ints_1e_ao
        self.ints_2e_ao = ints_2e_ao
        self.E_ne = E_ne
        self.T_alpha = moc_alpha
        self.T_beta  = moc_beta
        self.opt_T_alpha = moc_alpha
        self.opt_T_beta  = moc_alpha
        self.opt_done = False
        self.error = False
        self.occ_energy_calls = 0
        self.orb_energy_calls = 0
        self.active_space_calc='FCI'
        self._generate_active_space(**kwargs)
        self._generate_spin2spac_mapping()

    def _generate_active_space(self,
            Nels_tot,
            Norb_tot,
            Nels_as,
            Norb_as
            ):
        '''
        Note, all orb references are in spatial orbitals. 
        '''
        self.alpha_mo={
                'inactive':[],
                'active':[],
                'virtual':[]
                }
        self.beta_mo={
                'inactive':[],
                'active':[],
                'virtual':[]
                }
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











