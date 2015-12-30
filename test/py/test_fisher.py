import numpy as np
from yael import ynumpy

dat = np.load('test/py/test_fisher_dat.npy')
gmm = np.load('test/py/test_gmm.pickle')

sw = np.array([(i + 5) % 10 for i in xrange(len(dat))], dtype=np.float32)

dat_weighted = np.vstack([np.vstack([dat[i]] * sw[i])
                          for i in range(len(dat)) if sw[i] != 0])

# np.ones(len(dat), dtype=np.float32)

a = ynumpy.fisher(gmm, dat_weighted.astype(np.float32), include='mu+sigma')
b = ynumpy.fisher_sw(gmm, dat.astype(np.float32), sw, include='mu+sigma')

print ''
print '==========='
print ''

print a
print b
print a - b
