# Input variables should be written as: 'input'= 'variable', with the space 
# included after the equal sign, but not before. 
#
# use_backend = local_qasm_simulator, local_unitary_simulator, check
#____ also, ibmqx2, ibmqx4 have supported functionality
# 


#
# run types....specific to the algorithm
# so actually, there should just be an algorithm/run type
# or, you cna have method with sub methods
# so maybe, you have several types of methods, and then a requisite run type
# so for instance, we can have a 3 rot method, or 3_rot, and then a 6_rot method
#
# within that, you can specify the algorithm; ranges 
# any algorithm or program can use the local_qasm simulator, although unitary has to be compatible with it
# 

# so structure should be:
# run_type
# algorithm (specific to run_type)
# parameters (specific tmtting the log file...not really important, as the ID can't actually be obtained     from IBM without explicitly calling the API or something    
 algorithm)
# backend (also algorithm soecific actually)
# optional parameters, including
#   num_shots= number of runs
#   tomography= (algorithm specific)
#   order_shift= 
#   order=
#   pass=
#   save_file
#   off_diag= (algorithm specific)
#   combine= 
#   date_time=
# 

# run_type
#   single 
#   multi 
#    
#  
# algorithms
#   ry3p:
#   ry6p:

# how to handle parameters:
# really, should just load a parameteres file...

backend= local_qasm_simulator
run_type= user
num_shots= 1024
order= default
save_file= yes
pass= yes
algorithm= ry2p_raven_diag
combine= yes
connect= no
verbose= yes
$ 45.0 45.0
