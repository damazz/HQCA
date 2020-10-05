import os
import subprocess
import tempfile
from hqca.tools import *

def purify(rdm,quantstore):
    cdir = os.getcwd()
    rdm.save(name=cdir+'/_temp',spin=quantstore.spin_rdm)
    path_to_maple = quantstore.path_to_maple
    r = quantstore.No_as
    #
    blank = ''
    blank+=cdir
    blank+='with(LinearAlgebra): with(ArrayTools): with(QuantumChemistry):\n'
    blank+='loaddata := readdata(cat(cdir,\"_temp.rdm\"), 8):\n'
    # 
    if quantstore.spin_rdm:
        order = ['i','j','k','l']
        block = []
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
    if spin_rdm==True:
        kw = 'false'
    else:
        kw = 'true'
    blank+='pure := Purify2RDM(New, spin_free = {}, electron_number={}, conv_tol = 0.10000000):\n'.format(kw,quantstore.Ne)
    blank+='ExportMatrix(cat(cdir, \"_temp_purified.csv\"), Flatten(pure[rdm2])):'
    # 
    temp = tempfile.NamedTemporaryFile(mode='w+',dir=cudir,delete=False)
    temp.write(blank)
    temp.close()
    subprocess.run([path_to_maple+' '+temp.name],shell=True)
    test = np.loadtxt(cdir+'/_temp_purified.csv',delimiter=',')
    if os.path.exists(temp.name):
        os.remove(temp.name)
    if os.path.exists(cdir+'/_temp_purified.csv'):
        os.remove(cdir+'/_temp_purified.csv')
    if os.path.exists(cdir+'/_temp.rdm'):
        os.remove(cdir+'/_temp.rdm')
    test = np.reshape(test, (r,r,r,r))
    # 
    pure = RDM(order=2,
            alpha=quanstore.groups[0],
            beta=quantstore.groups[1],
            state='blank',
            Ne=quantstore.Ne,
            S=quantstore.Ne_alp-quantstore.Ne_bet,
            S2=0,
            state='spatial',
            rdm=test,
            )
    return pure

