from yael import ynumpy
import numpy as np

center = [
    np.array([0, 0]),
    np.array([10, 10])
]


def randn(num, d, sigma):
    return np.random.randn(num * d, sigma).reshape((num, d))


print (center[0] + randn(1000, 2, 1)).shape
print (center[1] + np.random.randn(3000, 2, 1)).shape

obs = np.vstack((center[0] + randn(1000, 2, 1), center[1] + randn(3000, 2, 1)))

num = 200000
obs_w = np.vstack((center[0] + randn(num, 2, 1), center[1] + randn(num, 2, 1)))
weight = np.hstack([np.ones(num), np.ones(num) * 3])

ret_original = ynumpy.gmm_learn(obs.astype(np.float32), 2)
ret_modified = ynumpy.gmm_learn_sw(obs_w.astype(np.float32), weight.astype(np.float32), 2)

print ''
print '======= result ======'
print ''

print ret_original
print ret_modified

orig_w, orig_mu, orig_sigma = ret_original
modi_w, modi_mu, modi_sigma = ret_modified

orig_i = np.argsort(orig_w)
modi_i = np.argsort(modi_w)

print "w    :", orig_w[orig_i] - modi_w[modi_i]
print "mu   :"
print orig_mu[orig_i] - modi_mu[modi_i]
print "sigma:"
print orig_sigma[orig_i] - modi_sigma[modi_i]
