from opts import *
from opts.functions import *

new =  Optimizer('nm',verbose=True,
        function=beale2p,
        )
new.initialize([0,0])
new.run()

new =  Optimizer('bfgs',verbose=True,
        function=beale2p,
        gradient=grad_beale2p,
        )
new.initialize([0,0])
new.run()
