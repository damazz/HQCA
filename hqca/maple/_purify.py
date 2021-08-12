import os
import numpy as np
import subprocess
import tempfile
from hqca.tools import *
from copy import deepcopy as copy

def purify(rdm,quantstore,verbose=False):
    '''

    Purify function takes a RDM and quantumstorage object (which needs qs.groups,
    qs.Ne,qs.Ne_alp,qs.Ne_bet, qs.No_as, qs.spin_rdm, qs.path_to_maple

    Can purify to differnt spin states, but generally only 

    '''
    cdir = os.getcwd()
    rdm.save(name=cdir+'/_temp',spin=quantstore.spin_rdm)
    rdm.contract()
    path_to_maple = quantstore.path_to_maple
    if quantstore.verbose:
        print('Purifying 2-RDM...')
        print('(with eigenvalues: )')
        print(np.linalg.eigvalsh(rdm.rdm))
        print('-----------------------------')
        print('-----------------------------')
    r = quantstore.No_as
    #
    blank = 'cdir := \"{}/\":\n'.format(cdir)
    blank+='with(LinearAlgebra): with(ArrayTools): with(QuantumChemistry):\n'
    blank+='loaddata := readdata(cat(cdir,\"_temp.rdm\"), 8):\n'
    # 
    if quantstore.spin_rdm:
        order = ['i','j','k','l']
    else:
        order = ['i','k','j','l']
    blank+='Flatten := proc(x) local n, a, i, j, k, l; `local`(a, n, i, j, k, l); `description`("convert chemists to numpy and flatten an array to form a matrix"); n := Size(x); a := Matrix(1 .. n[1]*n[2], 1 .. n[3]*n[4], datatype = float[8]); for i to round(n[1]) do for j to round(n[2]) do for k to round(n[3]) do for l to round(n[4]) do a[(i - 1)*n[1] + j, (k - 1)*n[3] + l] := x[{}, {}, {}, {}]; end do; end do; end do; end do; return a; end proc:\n'.format(order[0],order[1],order[2],order[3])
    blank+='New := Array(1 .. {}, 1 .. {}, 1 .. {}, 1 .. {}, datatype = float[8]):\n'.format(
            str(r),str(r),str(r),str(r))
    blank+='for i in loaddata[3 .. ()] do\n'
    if quantstore.spin_rdm:
        blank+='    New[round(i[1]), round(i[2]), round(i[3]), round(i[4])] := i[5]:\n'
    else:
        blank+='    New[round(i[1]), round(i[3]), round(i[2]), round(i[4])] := i[5]:\n'
    blank+='end do:\n'
    if quantstore.spin_rdm==True:
        kw = 'false'
        state = 'alpha-beta'
    else:
        kw = 'true'
        state = 'spatial'
    blank+='pure := Purify2RDM(New, spin_free = {}, electron_number={}, conv_tol = 0.00000001):\n'.format(kw,quantstore.Ne)
    blank+='ExportMatrix(cat(cdir, \"_temp_purified.csv\"), Flatten(pure[rdm2])):\n'
    #blank+='Transpose(Re(Eigenvalues(Flatten(pure[rdm2]))));'
    # 
    temp = tempfile.NamedTemporaryFile(mode='w+',dir=cdir,delete=False)
    temp.write(blank)
    temp.close()

    if quantstore.verbose:
        subprocess.run([path_to_maple+' '+temp.name],shell=True)
        print('-----------------------------')
        print('-----------------------------')
    else:
        subprocess.run([path_to_maple+' '+temp.name],shell=True,capture_output=True)
    test = np.loadtxt(cdir+'/_temp_purified.csv',delimiter=',')
    test = np.reshape(test, (r,r,r,r))
    # 
    pure = RDM(order=2,
            alpha=quantstore.groups[0],
            beta=quantstore.groups[1],
            Ne=quantstore.Ne,
            rdm=state,
            fragment=test
            )
    if not state=='spatial':
        pure.get_spin_properties()
        pure.contract()
        tr = pure.trace()
        sz = pure.sz
        s2 = pure.s2
        name = copy(temp.name)
        if quantstore.verbose:
            print('Eigenvalues of purified 2-RDM...')
            print(np.linalg.eigvalsh(pure.rdm))
            print('Trace of 2-RDM: {}'.format(pure.trace()))
            print('Projected spin: {}'.format(sz))
            print('Total spin: {}'.format(s2))
    pure.expand()
    cn = abs(abs(tr-quantstore.Ne*(quantstore.Ne-1)))>0.01
    csz = abs(abs(sz)-abs(quantstore.Ne_alp-quantstore.Ne_bet))>0.01
    cs2 = s2>0.01
    if os.path.exists(temp.name):
        os.remove(temp.name)
    if os.path.exists(cdir+'/_temp_purified.csv'):
        os.remove(cdir+'/_temp_purified.csv')
    if os.path.exists(cdir+'/_temp.rdm'):
        os.remove(cdir+'/_temp.rdm')
    if cn or csz or cs2:
        print('RDM violated some property after purification, saving to: {}'.format(
            name+'.rdm')
            )
        rdm.save(name=name,spin=quantstore.spin_rdm)
    return pure



def purify_rdm(name,quantstore):
    cdir = os.getcwd()
    path_to_maple = quantstore.path_to_maple
    print('Purifying 2-RDM: {}'.format(name))
    print('-----------------------------')
    print('-----------------------------')
    r = quantstore.No_as
    #
    blank = 'cdir := \"{}/\":\n'.format(cdir)
    blank+='with(LinearAlgebra): with(ArrayTools): with(QuantumChemistry):\n'
    blank+='loaddata := readdata(cat(cdir,\"{}.rdm\"), 8):\n'.format(name)
    # 
    if quantstore.spin_rdm:
        order = ['i','j','k','l']
    else:
        order = ['i','k','j','l']
    blank+='Flatten := proc(x) local n, a, i, j, k, l; `local`(a, n, i, j, k, l); `description`("convert chemists to numpy and flatten an array to form a matrix"); n := Size(x); a := Matrix(1 .. n[1]*n[2], 1 .. n[3]*n[4], datatype = float[8]); for i to round(n[1]) do for j to round(n[2]) do for k to round(n[3]) do for l to round(n[4]) do a[(i - 1)*n[1] + j, (k - 1)*n[3] + l] := x[{}, {}, {}, {}]; end do; end do; end do; end do; return a; end proc:\n'.format(order[0],order[1],order[2],order[3])
    blank+='New := Array(1 .. {}, 1 .. {}, 1 .. {}, 1 .. {}, datatype = float[8]):\n'.format(
            str(r),str(r),str(r),str(r))
    blank+='for i in loaddata[3 .. ()] do\n'
    if quantstore.spin_rdm:
        blank+='    New[round(i[1]), round(i[2]), round(i[3]), round(i[4])] := i[5]:\n'
    else:
        blank+='    New[round(i[1]), round(i[3]), round(i[2]), round(i[4])] := i[5]:\n'
    blank+='end do:\n'
    if quantstore.spin_rdm==True:
        kw = 'false'
        state = 'alpha-beta'
    else:
        kw = 'true'
        state = 'spatial'
    blank+='Transpose(Re(Eigenvalues(Flatten(New))));\n'
    blank+='pure := Purify2RDM(New, spin_free = {}, electron_number={}, conv_tol = 0.00000000001):\n'.format(kw,quantstore.Ne)
    blank+='ExportMatrix(cat(cdir, \"_temp_purified.csv\"), Flatten(pure[rdm2])):\n'
    #blank+='Transpose(Re(Eigenvalues(Flatten(pure[rdm2]))));'
    # 
    temp = tempfile.NamedTemporaryFile(mode='w+',dir=cdir,delete=False)
    temp.write(blank)
    temp.close()

    subprocess.run([path_to_maple+' '+temp.name],shell=True)
    test = np.loadtxt(cdir+'/_temp_purified.csv',delimiter=',')
    test = np.reshape(test, (r,r,r,r))
    print('-----------------------------')
    print('-----------------------------')
    # 
    pure = RDM(order=2,
            alpha=quantstore.groups[0],
            beta=quantstore.groups[1],
            Ne=quantstore.Ne,
            #S2=0,
            fragment=test,
            rdm=state,
            )
    pure.get_spin_properties()
    pure.contract()
    tr = pure.trace()
    sz = pure.sz
    s2 = pure.s2
    name = copy(temp.name)
    print('Eigenvalues of purified 2-RDM...')
    print(np.linalg.eigvalsh(pure.rdm))
    print('Trace of 2-RDM: {}'.format(pure.trace()))
    print('Projected spin: {}'.format(sz))
    print('Total spin: {}'.format(s2))
    pure.expand()
    cn = abs(abs(tr-quantstore.Ne*(quantstore.Ne-1)))>0.01
    csz = abs(abs(sz)-abs(quantstore.Ne_alp-quantstore.Ne_bet))>0.01
    cs2 = s2>0.01
    if os.path.exists(temp.name):
        os.remove(temp.name)
    if os.path.exists(cdir+'/_temp_purified.csv'):
        os.remove(cdir+'/_temp_purified.csv')
    if os.path.exists(cdir+'/_temp.rdm'):
        os.remove(cdir+'/_temp.rdm')
    #if cn or csz or cs2:
    #    print('RDM violated some property after purification, saving to: {}'.format(
    #        name+'.rdm')
    #        )
    #    rdm.save(name=name,spin=quantstore.spin_rdm)
    return pure

