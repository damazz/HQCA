'''
main.py 

Main program for executing the hybrid quantum classical optimizer. Consists of
several parts. 

First, a parameter file and molecular file are specified as arguments to
main.py. These must be provided. A mol.py file from pyscf which fills out the
atomic information is all that is required. For the program input file, should
just copy something already in use, or see the documentation in
/doc/options.txt. 

Integrals at the correct method level are computed, and then an optimization
procedure is carried out. 

Energies for the optimization can be carried in a number of ways. There are
classical options for certain problems, but the more general approach is to use
the IBM quantum computer and QISKIT modules, which will compute the energy of a
certain wavefunction with a quantum computer. The optimization then uses that,
and will proceed as needed. Current implementation favors the Nelder-Mead
process. 

Most of the functionality for the program is in /tools/. The interface for the
quantum computer is in /ibmqx/, where documentation is a little outdated for
certtain aspects. Critical for functionality are:
/tools/Chem.py              - Manages chemical attributes, electron integrals
/tools/EnergyDeterminant.py - Energy functions to be called 
/tools/EnergyOrbital.py    - Energy functions to be called 
/tools/EnergyFunctions.py   - Energy functions to be called 
/tools/Functions.py         - Common functions for various applications
/tools/Optimizers.py        - Houses optimizers functionality
/tools/RDMF.py              - Functions related to RDM manipulation and creation
/tools/IBM_check.py         - Calls to interface with IBM API
/tools/QuantumFramework.py  - Evaluates and carrys out different qc operations
/tools/QuantumTomography.py - Has framework for quantum tomography
/tools/QuantumAlgorithms.py - Contains quantum algorithism
/tools/Triangulations.py    - Contains methods and procedure for traingulation


'''
import pickle
import os, sys
from importlib import reload
import numpy as np
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from functools import reduce
from hqca.tools.QuantumFramework import add_to_config_log
import datetime
import sys
from pyscf import scf
np.set_printoptions(precision=3)

from hqca import sub
from hqca.tools import EnergyFunctions as enf

class sp:
    '''
    Class to perform a single point energy calculation
    '''
    def __init__(self,
            mol,
            theory,
            run_type='sp',
            restart=False):
        '''start function,
        assigns the chemical things and gets stuff going
        '''
        self.run_type = run_type
        self.theory=theory
        self._load_mol(mol)
        if restart:
            self._load_restart()
        else:
            self._choose_theory()

    def _load_mol(self,
            mol):
        self.S = mol.intor('int1e_ovlp')
        self.T_1e = mol.intor('int1e_kin')
        self.V_1e = mol.intor('int1e_nuc')
        self.ints_1e = self.V_1e+self.T_1e
        self.Norb = self.S.shape[0]
        self.ints_2e = mol.intor('int2e')
        self.hf = scf.RHF(mol)
        self.hf.kernel()
        self.C= self.hf.mo_coeff
        try:
            mol.as_Ne
        except Exception:
            mol.as_Ne=mol.nelec[0]+mol.nelec[1]
            mol.as_No= self.C.shape[0]
        store_kw = {
            'Nels_tot':mol.nelec[0]+mol.nelec[1],
            'Norb_tot':self.C.shape[0],
            'Nels_as':mol.as_Ne,
            'Norb_as':mol.as_No,
            'moc_alpha':self.C,
            'moc_beta':self.C,
            'ints_1e_ao':self.ints_1e,
            'ints_2e_ao':self.ints_2e,
            'E_ne':mol.energy_nuc()}
        self.Store = enf.Storage(
            **store_kw)
        self.Store.update_full_ints()


    def _load_restart(self):
        pass

    def _choose_theory(self):
        if self.theory in ['NOFT','noft']:
            self.run = sub.RunNOFT(self.Store)
        elif self.theory in ['rdm','RDM']:
            self.run = sub.RunRDM(self.Store)

    def update_var(self,
            args={}):
        self.run.update_var(args)

    def execute(self):
        self.run.cycle()


#print('Run on: {}'.format(datetime.datetime.now().isoformat()))
#
#if pre.qc_connect:
#    print('Run is connected to the IBMQuantumExperience.')
#    print('Checking for config file.')
#else:
#    print('Running locally. Checking for config file.')    
#add_to_config_log(pre.qc_use_backend,pre.qc_connect)
#
#if pre.restart_run:
#    restart_file = sys.argv[3]
#
#if pre.occ_load_triangle:
#    if (not pre.restart_run):
#        triangle_file = sys.argv[3]
#    else:
#        triangle_file = sys.argv[4]
#    try:
#        with open(triangle_file,'rb') as fb_in:
#            load_triangle = pickle.load(fb_in)
#        print('Successfully loaded triangle file.')
#        print('----------')
#    except:
#        traceback.print_exc()
#        sys.exit('Something is wrong with reading .tri file. Goodbye!')
## NOW, importing from the mol.py file, and get the electron integrals 
## Currently, only support for FCI orbitals, but going to add orbital 
## optimization procedure. 
#
#if pre.chem_orbitals=='FCI':
#    print('Calculating electron integrals in the full CI basis.')
#else:
#    print('Getting the electron integrals in the Hartree-Fock basis.')
#if 'mol' in sys.modules:
#    reload(mol)
#else:
#    import mol
#try:
#    els = mol.Nels
#    orbs= mol.Norb
#except AttributeError:
#    els  = 3
#    orbs = 3
#if pre.orb_seed:
#    try:
#        smol = mol.smol
#    except Exception:
#        smol=None
#        sys.exit('Some sort of error loading the seeded mol object.')
#else:
#    smol=None
#
#E_ne = mol.mol.energy_nuc()
#ints_1e, ints_2e, E_fci, hf_obj = chem.get_spin_ei(
#        mol=mol.mol,
#        elect=els,
#        orbit=orbs,
#        orbitals=pre.chem_orbitals,
#        seed=pre.orb_seed,
#        seed_mol=smol
#        )
#
#mol_els = mol.mol.nelec[0]+mol.mol.nelec[1]
#mol_orb = hf_obj.mo_coeff.shape[0]
#
#
#if pre.chem_orbitals=='FCI':
#    opt_orb = False
#elif pre.chem_orbitals=='HF':
#    opt_orb = True
#
## spin orbital basis (i.e. natural orbitals, or SCF orbitals)
#
#print('Electron integrals obtained. Moving forward.')
#print('Hartree-Fock energy: {}'.format(hf_obj.e_tot))
#print('CASCI/FCI energy: {}'.format(E_fci))
#print('----------')
#print('Wavefunction mapping is: {}'.format(pre.mapping))
#print('Nuclear energy: {:.8f} Hartrees'.format(E_ne))
#print('Quantum algorithm: {}'.format(pre.qc_algorithm))
#print('Quantum backend: {}'.format(pre.qc_use_backend))
#print('----------')
#
## Setting mapping for system. Should be size specific. 
#
#mapping = fx.get_mapping(pre.mapping)
#
#
##
##
## Now, beginning optimization procedure. 
##
##
#
#    try:
#        with open(restart_file,'rb') as fb_in:
#            dat = pickle.load(fb_in)
#            Run = dat[0]
#            Store = dat[1]
#            keys = dat[2]
#            orb_keys = dat[3]
#    except:
#        traceback.print_exc()
#        sys.exit('Something is wrong with reading .tmp file. Goodbye!')
#    Run.error = False
#    Run.check()
#else:
#    # Store holds total optimization information, and general wf info
#    store_keys = {
#            'Nels_tot':mol_els,
#            'Norb_tot':mol_orb,
#            'Nels_as':els,
#            'Norb_as':orbs,
#            'moc_alpha':hf_obj.mo_coeff,
#            'moc_beta':hf_obj.mo_coeff,
#            'ints_1e_ao':ints_1e,
#            'ints_2e_ao':ints_2e,
#            'E_ne':E_ne
#            }
#    Store  = enf.Storage(
#        **store_keys
#            )
#    if opt_orb:
#        Np = enf.rotation_parameter_generation(
#                Store.alpha_mo,
#                region=pre.orb_opt_region,
#                output='Npara'
#                )
#        print('Number of orbital parameters: {}'.format(Np))
#        orb_keys = {
#            'print_run':pre.orb_print,
#            'energy':'orbitals',
#            'region':pre.orb_opt_region,
#            'store':Store
#            }
#    else:
#        orb_keys={}
#    if pre.chem_orbitals=='HF':
#        Store.update_full_ints()
#    if pre.occ_energy=='classical':
#        # Energy optimization procedure is classical, very few key word arguments 
#        # necessary for energy 
#        keys = {
#            'wf_mapping':mapping,
#            'energy':pre.occ_energy,
#            'print_run':pre.print_extra,
#            'store':Store
#            }
#        pre.occ_increase_runs=False
#    elif pre.occ_energy=='qc':
#        # Energy function is computed through the quantum computer 
#        keys = {
#            'wf_mapping':mapping,
#            'algorithm':pre.qc_algorithm,
#            'backend':pre.qc_use_backend,
#            'order':pre.qc_qubit_order,
#            'num_shots':pre.qc_num_shots,
#            'split_runs':pre.qc_combine_run,
#            'connect':pre.qc_connect,
#            'method':pre.occ_method,
#            'print_run':pre.print_extra,
#            'energy':pre.occ_energy,
#            'verbose':pre.qc_verbose,
#            'wait_for_runs':pre.wait_for_runs,
#            'store':Store
#            }
#if pre.restart_run:
#    if pre.occ_load_triangle:
#        keys['triangle']=load_triangle
## Determine single run or not
#
#
## Now, begin the optimziation.
## There are two steps within a single loop. 
##
## First, if necessary, i.e. if using HF or SCF orbitals, we optimize 
## the orbitals. If using the FCI solution, then this part of the 
## optimization is skipped over. 
##
## Second, we perform the occupation number optimization using either the
## classical method or the quantum computer. 
##
## Finally, we check how well converged the two optimizations are with each
## other. If they are within a certain threshhold, then we are done. 
#
#
#iter_total=0
#while Store.opt_done==False:
#    #
#    # Starting the optimization procedure! 
#    #
#    # 
#    # Next few lines are dedicated to setting parameters and updating the
#    # holding variables. Any initialization parameters are kept in this portion. 
#    # First we set parameters for the occupation number optimization, and then
#    # the orbital optimization. 
#    #
#    # RESTART RUN
#    if pre.restart_run and iter_total==0:
#        opt_keys={}
#        pass
#    else:
#        # TRIANGLE
#        if pre.occ_energy=='qc':
#            if keys['num_shots']<2048:
#                keys['num_shots']=2048
#            if pre.occ_method in ['stretch'] and iter_total==0:
#                try:
#                    if (not pre.occ_load_triangle):
#                        print('Measuring triangle for generating an  affine transformation.')
#                        print('Please make sure your circuit rotation is in 2D.')
#                        print('----------')
#                        keys['triangle']=tri.find_triangle(
#                                Ntri=pre.occ_method_Ntri,
#                                **keys)
#                        keys
#                        with open(pre.filename+'.tri', 'wb') as fp:
#                            pickle.dump(
#                                    keys['triangle'],
#                                    fp,
#                                    pickle.HIGHEST_PROTOCOL
#                                    )
#                    else:
#                        keys['triangle']=load_triangle
#                except Exception as e:
#                    print('Error in start. Goodbye!')
#                    traceback.print_exc()
#                print('Succefully have the triangulation.',
#                        'Proceeding with optimization.')
#                print('----------')
#            else:
#                pass
#        else:
#            pass
#        keys['num_shots']=pre.qc_num_shots
#        #
#        # OCCUPTATION OPTIMIZERS
#        #
#        print('Optimizing the wavefunction.')
#            Run = opt.Optimizer(
#                    pre.occ_opt_method,
#                    pre.parameters,
#                    **keys)
#        else:
#            parameters = [Store.parameters]
#            Run = opt.Optimizer(
#                   pre.occ_opt_method,
#                   parameters,
#                   **keys
#                    )
#        if Run.error:
#            print('##########')
#            print('Encountered error in initialization.')
#            print('##########')
#            print('----------')
#            filename = pre.filename
#            with open(pre.filename+'.run.tmp', 'wb') as fp:
#                pickle.dump(
#                        [Run,Store,keys,orb_keys],
#                        fp,
#                        pickle.HIGHEST_PROTOCOL
#                        )
#            Store.opt_done=True
#            continue
#        else:
#            pass
#
#    #
#    # Begin the optimization
#    #
#    # Regardless of run number, we still restart the optimizer once we get
#    # to this step. 
#    #
#    # Now, begin the main occupation number optimizer. 
#    #
#    iter_occ=0
#    while Run.opt_done==False:
#        Run.next_step(**keys)
#        if pre.print_extra:
#            print('----------')
#            print('----------')
#        print('Step: {:02}, Total Energy: {:.8f} Sigma: {:.8f}  '.format(
#            iter_occ,
#            Run.opt.best_f,
#            Run.opt.crit)
#            )
#        if pre.print_extra:
#            print('----------')
#            print('----------')
#            if pre.occ_method=='NM':
#                print(Run.opt.B_x)
#        # Now, we check if our run is converged. 
#        Run.check()
#        #if iter_occ==20:
#        #    Run.opt_done=True
#        #    Run.error=True
#        #    opt_orb=False
#        if iter_occ==pre.occ_max_iter and Run.opt_done==False:
#            # Reached maximum iterations
#            Run.opt_done=True
#            print('----------')
#            print('Max iterations performed. Going to orbital optimizations.')
#            print('----------')
#            Run.error=False
#            continue
#        elif Run.opt_done:
#            # Optimization finished
#            if Run.error:
#                # Finished with error, going to save the file
#                print('Error in run. Saving optimization object to file.')
#                # Objects to save: Run, iterations
#                filename = pre.filename
#                with open(filename+'.run.tmp', 'wb') as fp:
#                    pickle.dump(
#                            [Run,Store,keys,orb_keys],
#                            fp,
#                            pickle.HIGHEST_PROTOCOL
#                            )
#                Store.opt_done=True
#            continue
#        else:
#            pass
#
#        iter_occ+=1
#    '''
#    print('Checking energy.')
#    rdm1 = rdmf.check_2rdm(Store.rdm2,5)
#    e1 = reduce( np.dot, (Store.ints_1e,rdm1.T)).trace()
#    e2 = reduce( 
#            np.dot,
#            (
#                Store.ints_2e,
#                np.reshape(
#                    Store.rdm2.T,
#                    (Store.ints_2e.shape)
#                    )
#                )
#            ).trace()
#    print('Total energy: {} Hartrees'.format(e1+0.5*e2+Store.E_ne))
#    '''
#    Store.update_rdm2()
#    '''
#    rdm1 = rdmf.check_2rdm(Store.rdm2,5)
#    e1 = reduce( np.dot, (Store.ints_1e,rdm1.T)).trace()
#    e2 = reduce( 
#            np.dot,
#            (
#                Store.ints_2e,
#                np.reshape(
#                    Store.rdm2.T,
#                    (Store.ints_2e.shape)
#                    )
#                )
#            ).trace()
#    print('Total energy: {} Hartrees'.format(e1+0.5*e2+Store.E_ne))
#    '''
#    # DONE WITH OCCUPATION NUMBER OPTIMIZATION
#    # ORBITAL RESTART
#    if pre.restart_run and iter_total==0 and opt_orb:
#        pass
#    if opt_orb:
#        print('Optimizing the orbitals.')
#        if iter_total==0:
#            orb_keys['conv_crit_type']='default'
#            orb_keys['conv_threshold']=pre.orb_opt_thresh
#        else:
#            temp={}
#        # ORBITAL OPTIMIZERS
#        if pre.orb_opt_method=='NM' :
#            if iter_total==0:
#                orb_keys['simplex_scale'] = pre.orb_nm_simplex
#            #else:
#                #temp_key = orb_keys['simplex_scale']
#                #orb_keys['simplex_scale']= max(1,temp_key/2)
#        elif pre.orb_opt_method=='GD':
#            if iter_total==0:
#                orb_keys['gamma']='default'
#                orb_keys['gradient']=pre.orb_gd_gradient
#                orb_keys['grad_dist']=pre.orb_gd_grad_dist
#            else:
#                orb_keys['grad_dist']= max(0.001,Orbit.opt.dist/2)
#        # Set parameters
#        # Note, they DO need to be in a double array [[p]]
#        para_orb = []
#        para_orb.append([])
#        for i in range(0,2*Np):
#           para_orb[0].append(0)
#        Orbit = opt.Optimizer(
#                pre.orb_opt_method,
#                para_orb,
#                **orb_keys
#                )
#    else:
#        # (No orbital optimization)
#        # (Use Empty class, which has the right calls)
#        Orbit = opt.Empty()
#        Store.update_fci(E_fci,ints_1e,ints_2e)
#
#
#
#    iter_orb=0
#    while Orbit.opt_done==False:
#        Orbit.next_step(**orb_keys)
#        print('Step: {:02}, Total Energy: {:.8f} Sigma: {:.8f}  '.format(
#            iter_orb,
#            Orbit.opt.best_f,
#            Orbit.opt.crit)
#            )
#        Orbit.check()
#        iter_orb+=1
#
#        if iter_orb==pre.orb_max_iter and Orbit.opt_done==False:
#            Orbit.opt_done=True
#            Orbit.error=True
#
#    # Now, update electron integrals for ON and orb calculations
#    # If the ending/starting energies of difference optimizations do not match,
#    # probably should check this portion and make sure the values used for
#    # energies are correct. 
#    Store.update_full_ints()
#    #if opt_orb:
#    #    #keys['ints_1e_no']=Store.ints_1e
#    #    #keys['ints_2e_no']=Store.ints_2e
#    #    #orb_keys['mo_coeff_a']=Store.T_alpha
#    #    #orb_keys['mo_coeff_b']=Store.T_beta
#
#
#    Store.check(pre.opt_crit,Run,Orbit)
#    if iter_total>=pre.max_iter:
#        Store.opt_done=True
#        print('----------')
#        print('Max number of total iterations reached. Finishing optimization.')
#    iter_total+=1
#    print('----------')
#    print('Total number of iterations: {}'.format(iter_total))
#
#    if (Store.opt_done==True and Store.error==False):
#        print('----------')
#        print('Finished orbital optimization. Done!')
#    elif Store.opt_done==True and Store.error==True:
#        print('##########')
#        print('ERROR: encountered some error along the way. Run is likely')
#        print('not fully converged. Check what went wrong.')
#        print('##########')
#    else:
#        print('----------')
#        print('Finished orbital optimization. Optimizing determinants.')
#        print('----------')
#    try:
#        Store.update_calls(Orbit.opt.energy_calls,Run.opt.energy_calls)
#    except:
#        Store.occ_energy_calls=Run.opt.energy_calls
#
## print Energy, wavefunction 
## TODO: make less specific...still
#try:
#    print('------------------------------')
#    print('------------------------------')
#    print('## Energy Evaluation ## ')
#    print('Final energy                : {:.8f} Hartrees'.format(
#        Store.energy_best))
#    print('Final sigma                 : {:.8f}'.format(
#        abs(Store.energy_int-Store.energy_wf)))
#    print('')
#    print('Hartree Fock Energy         : {:.8f} Hartrees'.format(
#        hf_obj.e_tot))
#    print('CASCI energy              : {:.8f} Hartrees'.format(
#        E_fci))
#    print(
#        'Difference in target and FCI  :', 
#        ' {:.5f} mHartrees'.format(
#            abs(E_fci-Store.energy_best)*1000
#            )
#        )
#    if pre.orb_seed:
#        print('CASCI energy              : {:.8f} Hartrees'.format(
#            hf_obj.mcscf_e_tot))
#
#    print('----------')
#    print('Correlation energy recovered: {:.5f} mHartrees'.format(
#            1000*(Store.energy_best-hf_obj.e_tot)))
#    print('Correlation energy recovered: {:3.3f} %'.format(
#                100*(Store.energy_best-hf_obj.e_tot)/(E_fci-hf_obj.e_tot)
#                )
#            )
#    print('------------------------------')
#    print('------------------------------')
#    print('Wavefunction (Hamiltonian basis):')
#    for k,v in Store.wf.items():
#        print(' |{}>: {}'.format(k,v))
#    print('----------')
#    print('1-Electron Reduced Density Matrix:')
#    rdm2 = rdmf.build_2rdm(Store.wf,
#            alpha=Store.alpha_mo,
#            beta=Store.beta_mo,
#            region='full'
#            )
#    rdm1 = rdmf.check_2rdm(rdm2,mol_els)
#    #print(np.real(rdm1))
#    eigval,eigvec = np.linalg.eig(rdm1)
#
#    #print('Molecular orbitals:')
#    #print(Store.ints_1e)
#except:
#    traceback.print_exc()
#print('Total orbital energy evaluations   : {}'.format(
#    Store.orb_energy_calls))
#print('Total occupation energy evaluations: {}'.format(
#    Store.occ_energy_calls))
#
#
