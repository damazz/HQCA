# Before you can use the jobs API, you need to set up an access token.
# Log in to the Quantum Experience. Under "Account", generate a personal 
# access token. Replace 'PUT_YOUR_API_TOKEN_HERE' below with the quoted
# token string. Uncomment the APItoken variable, and you will be ready to go.

#APItoken = '8274b4430a7f72d29324ccb958ad286c082ef9b0161781272006ae48c6114804784e98eb8c4f5d7342d033b25ab2522c20a34d00e9e4b1e3d8e27d07a778c61c'
#APItoken = 'dcafc170faba32d807eef288631ba290b77b4183f6d25a6b0f829418f5c4782c1b70deaa0f2f5625b3d613dccc4ff52dd3aefd51ad04dbff'

#APItoken =  '539fd4fe5010f40adcafc170faba32d807eef288631ba290b77b4183f6d25a6b0f829418f5c4782c1b70deaa0f2f5625b3d613dccc4ff52dd3aefd51ad04dbff'
APItoken = '539fd4fe5010f40adcafc170faba32d807eef288631ba290b77b4183f6d25a6b0f829418f5c4782c1b70deaa0f2f5625b3d613dccc4ff52dd3aefd51ad04dbff'
SIM_EXECUTABLE = '/usr/local/lib/python3.5/dist-packages/qiskit/backends/qasm_simulator_cpp'

    # If you have access to IBM Q features, you also need to fill the "hub",
    # "group", and "project" details. Replace "None" on the lines below
    # with your details from Quantum Experience, quoting the strings, for
    # example: 'hub': 'my_hub'
    # You will also need to update the 'url' above, pointing it to your custom
    # URL for IBM Q.
config = {
    'url': 'https://quantumexperience.ng.bluemix.net/api',
    'hub': None,
    'group': None,
    'project': None
}

if 'APItoken' not in locals():
    raise Exception('Please set up your access token. See Qconfig.py.')
