from qiskit import Aer,IBMQ
from qiskit.tools.monitor import backend_overview
print('Loading accounts.')

IBMQ.load_accounts()

print('Loading backends.')
print('')

backend_overview()
