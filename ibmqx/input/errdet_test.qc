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
# parameters (specific to algorithm)
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

backend= check
run_type= user
num_shots= 2048
order= default
#order_shift= no
save_file= yes
pass= no
algorithm= ry4p_err_det_sparrow
combine= no
tomography= no
$ 45 45 

