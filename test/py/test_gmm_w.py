from yael import ynumpy
import numpy as np
import matplotlib.pyplot as plt

obs = np.concatenate((np.random.randn(100, 1), 10 + np.random.randn(300, 1)))

ynumpy.gmm_learn_w(obs.astype(np.float32), 2)


