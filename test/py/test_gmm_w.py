from yael import ynumpy
import numpy as np

label = np.load('test/py/test_label.npy')
dat = np.load('test/py/test_dat.npy')

cluster_num = len(set(label))
print cluster_num

cluster_w = np.array([(i + 2) % 5 for i in xrange(cluster_num)])

obs = np.vstack([np.vstack([dat[label == c]] * cluster_w[c]) for c in xrange(cluster_num) if cluster_w[c] != 0])

obs_w = dat.copy()
weight = cluster_w[label]

print obs_w.shape
print weight.shape

ret_original = ynumpy.gmm_learn(obs.astype(np.float32), cluster_num - 1)
ret_modified = ynumpy.gmm_learn_sw(obs_w.astype(np.float32), weight.astype(np.float32), cluster_num - 1)

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

print ''
print '==========='
print ''

print "w    :"
print orig_w[orig_i]
print modi_w[modi_i]
print "mu   :"
print orig_mu[orig_i]
print modi_mu[modi_i]
print "sigma:"
print orig_sigma[orig_i]
print modi_sigma[modi_i]
