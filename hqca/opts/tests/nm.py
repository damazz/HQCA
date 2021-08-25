from opts import *
from opts.functions import *



new =  Optimizer('nm',verbose=True,
        function=beale2p,
        )
new.initialize([0,0])
new.run()

