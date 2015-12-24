from yael import ynumpy
import numpy as np

obs = np.concatenate((np.random.randn(100, 1), 10 + np.random.randn(300, 1)))

ret_original = ynumpy.gmm_learn(obs.astype(np.float32), 2)
ret_modified = ynumpy.gmm_learn_w(obs.astype(np.float32), np.ones(obs.shape).astype(np.float32), 2)

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
