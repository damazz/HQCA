from qiskit import Aer,IBMQ
import pickle
import traceback
from qiskit.tools.monitor import job_monitor
from qiskit import compile, execute
import qiskit
from qiskit.providers.aer import noise
from hqca.tools import Qconfig
import hqca

def get_noise_model(device,times=None,saved=False):
    if (not saved) or (saved is None):
        IBMQ.load_accounts()
        backend = IBMQ.get_backend(device)
        properties = backend.properties()
    else:
        try:
            loc = hqca.__file__
            if loc[-11:]=='__init__.py':
                loc = loc[:-12]
            loc =  loc+'/results/logs/'+saved
            with open(loc,'rb') as fp:
                data = pickle.load(fp)
        except FileNotFoundError:
            print('Wrong one :(')
        properties = data['properties']
    if times is not None:
        noise_model = noise.device.basic_device_noise_model(
            properties,times)
    else:
        noise_model = noise.device.basic_device_noise_model(
            properties)
    return noise_model

