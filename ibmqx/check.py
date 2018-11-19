from IBMQuantumExperience import IBMQuantumExperience
from qiskit import QuantumProgram, register
test_program = QuantumProgram() 


if __name__=='__main__':
    import Qconfig 
    test = register(Qconfig.APItoken, Qconfig.config['url'])
    #t(Qconfig.APItoken,Qconfig.config['url'])
    print('Here are the available programs: ')
    backends = test.available_backends()
    print(backends)
    for i in backends:
        if (not 'simulator' in i):
            print(test_program.get_backend_status(i))

def check(backend):
    from ibmqx import Qconfig
    test_program.set_api(Qconfig.APItoken,Qconfig.config['url'])
    print(here)
    test =test_program.get_backend_status(backend)
    try: 
        n = int(test['pending_jobs'])
    except:
        n = 0 
    if test['available']:
        avail = 1
    else:
        avail=0
    return avail,n

