from setuptools import setup


setup(
        name='hqca',
        version='22.4',
        packages=['hqca'],
        install_requires=[
            'delayed_assert==0.3.2',
            'matplotlib==3.3.1',
            'networkx==2.5',
            'numpy==1.21.2',
            'pyscf==1.7.6.post1',
            'pytest==6.1.2',
            'qiskit==0.29.0',
            'qiskit-aer==0.8.2',
            'qiskit-ibmq-provider==0.16.0',
            'qiskit-ignis==0.6.0',
            'scipy==1.7.1',
            'sympy==1.8'
            ],
        )

