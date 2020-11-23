import numpy as np
'''

circuits for h2:

['XXXX', 'YXXX', 'XXYX', 'YXYX', 'XXZZ', 'YXZZ', 'ZZXX', 'ZZYX', 'ZZZZ']
 'IIII': 'XXXX', 'XXII': 'XXXX', 'IIXX': 'XXXX', 'XXXX': 'XXXX', 'IZII': 'ZZXX', 'ZIII': 'ZZXX', 'ZZII': 'ZZXX', 'ZIXX': 'ZZXX', 'IZXX': 'ZZXX', 'IIIZ': 'XXZZ', 'IIZI': 'XXZZ', 'IIZZ': 'XXZZ', 'XXZI': 'XXZZ', 'XXIZ': 'XXZZ', 'ZIZI': 'ZZZZ', 'ZIIZ': 'ZZZZ', 'IZZI': 'ZZZZ', 'IZIZ': 'ZZZZ', 'YXII': 'YXXX', 'YXXX': 'YXXX', 'YXZI': 'YXZZ', 'YXIZ': 'YXZZ', 'IIYX': 'XXYX', 'XXYX': 'XXYX', 'ZIYX': 'ZZYX', 'IZYX': 'ZZYX', 'YXYX': 'YXYX'}
'''

circ = ['XXXX', 'YXXX', 'XXYX', 'YXYX', 'XXZZ', 'YXZZ', 'ZZXX', 'ZZYX', 'ZZZZ']
mapping = {
    'IIII': 'XXXX', 'XXII': 'XXXX',
    'IIXX': 'XXXX', 'XXXX': 'XXXX',
    'IZII': 'ZZXX', 'ZIII': 'ZZXX',
    'ZZII': 'ZZXX', 'ZIXX': 'ZZXX',
    'IZXX': 'ZZXX', 'IIIZ': 'XXZZ',
    'IIZI': 'XXZZ', 'IIZZ': 'XXZZ',
    'XXZI': 'XXZZ', 'XXIZ': 'XXZZ',
    'ZIZI': 'ZZZZ', 'ZIIZ': 'ZZZZ',
    'IZZI': 'ZZZZ', 'IZIZ': 'ZZZZ',
    'YXII': 'YXXX', 'YXXX': 'YXXX',
    'YXZI': 'YXZZ', 'YXIZ': 'YXZZ',
    'IIYX': 'XXYX', 'XXYX': 'XXYX',
    'ZIYX': 'ZZYX', 'IZYX': 'ZZYX',
    'YXYX': 'YXYX'}

circuits = {k:[] for k in circ}
for k,v in mapping.items():
    circuits[v].append(k)
print(circuits)
