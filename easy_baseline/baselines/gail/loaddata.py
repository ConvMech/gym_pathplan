import numpy as np

data = np.load("data/deterministic.trpo.Humanoid.0.00.npz")
print(len(data['obs']))
print(data['ep_rets'])
print(data['ep_rets'].shape)
print('shape of reward',data['rews'].shape)
print('shape of obs',data['obs'].shape)

print(data['obs'][0].shape)
print(data['obs'][1].shape)
