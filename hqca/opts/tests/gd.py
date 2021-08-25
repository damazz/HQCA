from opts import *
from opts.functions import *



new =  Optimizer('gd',verbose=True,
        function=beale2p,
        gradient=grad_beale2p)
new.initialize([0,0])
new.run()

