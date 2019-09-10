from qiskit import Aer,IBMQ
from qiskit.tools.monitor import backend_overview,backend_monitor
print('Loading accounts.')

IBMQ.load_accounts()

print('Loading backends.')
print('')

backend_overview()

#backend_monitor(IBMQ.get_backend('ibmq_16_melbourne'))
backend_monitor(IBMQ.get_backend('ibmqx4'))
