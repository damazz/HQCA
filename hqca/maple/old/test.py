'''
hqca/hqca/maple/test.py

Trying to get Maple to work! 
 
 :) 

 we did it! 

'''

#
# need to have a path_to_maple
#
import os
import subprocess
import tempfile
from hqca.tools import *

a = RDM(order=2,
        alpha=[0,1,2],
        beta=[3,4,5],
        state='hf',
        Ne=4,
        S=0,
        )
#path_to_maple = '/home/scott/maple2020/bin/maple -q'
path_to_maple = '/home/scott/maple2020/bin/maple'
cdir = 'cdir := \"/home/scott/Documents/research/software/hqca/hqca/maple/\":\n'
cudir = '/home/scott/Documents/research/software/hqca/hqca/maple/'


def generate_mpl(Ne=4,quantstore):
    blank = ''
    blank+=cdir
    blank+='with(LinearAlgebra): with(ArrayTools): with(QuantumChemistry):\n'
    blank+='loaddata := readdata(cat(cdir,\"_temp.rdm\"), 8):\n'
    blank+='Flatten := proc(x) local n, a, i, j, k, l; `local`(a, n, i, j, k, l); `description`("convert chemists to numpy and flatten an array to form a matrix"); n := Size(x); a := Matrix(1 .. n[1]*n[2], 1 .. n[3]*n[4], datatype = float[8]); for i to round(n[1]) do for j to round(n[2]) do for k to round(n[3]) do for l to round(n[4]) do a[(i - 1)*n[1] + j, (k - 1)*n[3] + l] := x[i, j, k, l]; end do; end do; end do; end do; return a; end proc:\n'
    blank+='New := Array(1 .. 4, 1 .. 4, 1 .. 4, 1 .. 4, datatype = float[8]):\n'
    blank+='for i in loaddata[3 .. ()] do\n'
    blank+='    New[round(i[1]), round(i[2]), round(i[3]), round(i[4])] := i[5]:\n'
    blank+='end do:\n'
    blank+='pure := Purify2RDM(New, spin_free = false, electron_number={}, conv_tol = 0.10000000):\n'.format(Ne)
    blank+='ExportMatrix(cat(cdir, \"_temp_purified.csv\"), Flatten(pure[rdm2])):'
    #
    temp = tempfile.NamedTemporaryFile(mode='w+',dir=cudir,delete=False)
    temp.write(blank)
    temp.close()
    subprocess.run([path_to_maple+' '+temp.name],shell=True)
    if os.path.exists(temp.name):
        os.remove(temp.name)

generate_mpl(Ne=4)

