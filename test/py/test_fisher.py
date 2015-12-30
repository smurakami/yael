import numpy as np
from yael import ynumpy

dat = np.load('test/py/test_fisher_dat.npy')
gmm = np.load('test/py/test_gmm.pickle')

dat_a = dat[:len(dat)/2]
dat_b = dat[len(dat)/2:]
a = ynumpy.fisher(gmm,
                  np.vstack([dat, dat]).astype(np.float32),
                  include='mu+sigma')
b = ynumpy.fisher(gmm,
                  np.vstack([dat]).astype(np.float32),
                  include='mu+sigma')
sw_a = np.ones(len(dat)/2) * 4
sw_b = np.ones(len(dat)/2) * 2
c = ynumpy.fisher_sw(gmm,
                     dat.astype(np.float32),
                     np.vstack([sw_a, sw_b]).astype(np.float32),
                     include='mu+sigma')

# print a - c
print a - b
# print b - c

# sw = np.array([(i + 5) % 10 for i in xrange(len(dat))])

# dat_weighted = np.vstack([np.vstack([dat[i]] * sw[i])
#                           for i in range(len(dat)) if sw[i] != 0])

# # np.ones(len(dat), dtype=np.float32)

# a = ynumpy.fisher(gmm, dat_weighted.astype(np.float32), include='mu+sigma')
# b = ynumpy.fisher_sw(gmm, dat.astype(np.float32), (sw).astype(np.float32), include='mu+sigma')

# print sw
# print sw * 0.1

# print ''
# print '==========='
# print ''

# print a
# print b
# print a - b
