# prepare.py
#
# File reads in an input file, which in the default is the current test file
# "test.qc" in the "../input" folder, and writes to the configuration file, 
# "./config.txt'. Executed by the main.py script in the parent folder.  
# Intermediate between input/run file and the settings file.  
#

import sys
import numpy as np
import traceback
import inspect

#print(inspect.getfile(inspect.currentframe()))
try:
    filename = sys.argv[2]+sys.argv[1]
    #print(filename)
    with open(filename) as fp:
        try:
            with open('./ibmqx/config.txt','w') as con:
                for line in fp:
                    if line[0]=='#':
                        continue
                    elif line[0]=='\n':
                        continue
                    else:
                        con.write(line)
        except FileNotFoundError:
            with open('./config.txt','w') as con:
                for line in fp:
                    if line[0]=='#':
                        continue
                    elif line[0]=='\n':
                        continue
                    else:
                        con.write(line)
except:
    traceback.print_exc()




