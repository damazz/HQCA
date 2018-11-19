from IBMQuantumExperience import IBMQuantumExperience
from qiskit import QuantumProgram, register
test_program = QuantumProgram()


if __name__=='__main__':
    import Qconfig
    from qiskit import available_backends, get_backend
    test = register(Qconfig.APItoken)
    api = IBMQuantumExperience(Qconfig.APItoken)
    print(api.get_my_credits())
    print('Here are the available programs: ')
    backends = available_backends()
    for i in backends:
        print(get_backend(i).status)
else:
    from tools import Qconfig
    register(Qconfig.APItoken)


def get_backend_object(backend):
    from qiskit import get_backend,available_backends
    test = get_backend(backend)
    return test

def check(backend_object):
    test = backend_object.status
    try:
        n = int(test['pending_jobs'])
    except:
        n = 0
    if test['operational']:
        avail = 1
    else:
        avail=0
    return avail,n

