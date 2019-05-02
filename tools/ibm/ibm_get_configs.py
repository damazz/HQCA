from qiskit import Aer,IBMQ,execute
import traceback
from qiskit.tools.monitor import backend_overview
import hqca

def add_to_config_log(backend,connect,location='default'):
    '''
    Function to add the current ibm credentials, configuration to the
    ./results/logs/ directory, so it is stored for potential publications.
    '''
    # check if config file is already there
    from datetime import date
    import pickle
    today = date.timetuple(date.today())
    today = '{:04}{:02}{:02}'.format(today[0],today[1],today[2])
    if location=='default':
        loc  = str(hqca.__file__)
        if loc[-11:]=='__init__.py':
            loc = loc[:-12]
    else:
        loc = location
    loc = loc + '/results/logs/'
    filename = loc+today+'_'+backend.name()
    try:
        with open(filename,'rb') as fp:
            pass
        print('----------')
        print('Backend configuration file already written.')
        print('Stored in following location:')
        print('{}'.format(filename))
        print('----------')
    except FileNotFoundError:
        with open(filename,'wb') as fp:
            data  = {'name':be.name(),
                    'config':be.configuration(),
                    'properties':be.properties()}
            name = be.name()
            pickle.dump(
                    data,
                    fp,0
                    )
        print('----------')
        print('Backend configuration file written.')
        print('Stored in following location:')
        print('{}'.format(filename))
        print('----------')

if __name__=='__main__':
    IBMQ.load_accounts()
    print('One moment please.')
    b = IBMQ.backends()
    print('Ok. Which backend would you like to save?')
    for i in b:
        print(i)
    selected=False
    while not selected:
        ans = input('ibm device: ')
        try:
            be = IBMQ.get_backend(ans)
            add_to_config_log(backend=be,connect=True)
            selected=True
            res = input('Try another? y/n: ')
            if res in ['yes','y','Y','Yes']:
                selected=False
        except Exception as e:
            print('Not compatible device. Please try again.')
            traceback.print_exc()


