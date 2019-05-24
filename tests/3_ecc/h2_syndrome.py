import hqca
from hqca import hqca
kw_qc = {
        'Nels_as':2,
        'Norb_as':2,
        'info':'draw',
        'Nq':4,
        'Nq_backend':5,
        'Nq_ancilla':1,
        'ec':True,
        'ec_method':'parity',
        'ec_ent_list':[1],
        }
qc_obj = hqca.circuit('noft')
qc_obj.update_var('qc',**kw_qc)
qc_obj.build()
qc_obj.run()
