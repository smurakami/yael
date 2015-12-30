import numpy as np
from yael import ynumpy

dat = np.load('test/py/test_fisher_dat.npy')
gmm = np.load('test/py/test_gmm.pickle')

sw = np.array([(i + 5) % 10 for i in xrange(len(dat))], dtype=np.float32)

dat_weighted = np.vstack([np.vstack([dat[i]] * sw[i])
                          for i in range(len(dat)) if sw[i] != 0])
dat_weighted_idx = np.hstack([np.hstack([i] * sw[i])
                              for i in range(len(dat)) if sw[i] != 0])

# np.ones(len(dat), dtype=np.float32)

a = ynumpy.gmm_compute_p(gmm, dat_weighted.astype(np.float32), include='mu')
b = ynumpy.gmm_compute_p_sw(gmm, dat.astype(np.float32), sw, include='mu')

print ''
print '==========='
print ''

print dat_weighted.shape
print dat.shape
print np.sum(np.abs(a - b[dat_weighted_idx]))
# print b[dat_weighted_idx].shape
