import datetime
import numpy as np
import struct

def load_matrix(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        m = struct.unpack('<Q', f.read(8))[0]
        return np.fromfile(f, dtype=np.single).reshape(n, m)

x = load_matrix('./baseline/x.dat')
y = load_matrix('./baseline/y.dat')

s = datetime.datetime.now()
z = x@y.T
dt = datetime.datetime.now() - s

print("x: ", x[:2, :2])
print("y: ", y[:2, :2])
print("z: ", z[:2, :2])

print(dt.microseconds/1000, 'ms')
