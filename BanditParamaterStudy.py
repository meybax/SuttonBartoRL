import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand

k = 10
steps = 2000

def measurePerformance(alp, eps, k, steps, init):
    true_Q = rand.randn(k)
    Q = np.full(k, init)

    reward = 0
    for s in range(steps):
        inc = rand.normal(0, 0.01, k)
        true_Q = true_Q + inc
        curr = step(true_Q, Q, alp, eps, k)
        if (s > steps / 2):
            reward += curr
    return reward * 2 / steps

def step(q, Q, step_size, eps, k):
    if rand.rand() > eps:
        action = np.argmax(Q)
    else:
        action = rand.randint(k)

    R = rand.normal(q[action], 0.1)
    Q[action] = Q[action] + step_size * (R - Q[action])
    return R

n = 10
rewards_eps = np.zeros(n)
rewards_alp = np.zeros(n)
rewards_Q = np.zeros(n)
param = np.zeros(n)
for i in range(n):
    param[i] = 2**i / 128
    rewards_eps[i] = measurePerformance(0.1, param[i], k, steps, 0.)
    rewards_alp[i] = measurePerformance(param[i], 0.1, k, steps, 0.)
    rewards_Q[i] = measurePerformance(0.1, 0.1, k, steps, param[i])

fig, ax = plt.subplots()
ax.set_xscale('log', basex=2)

plt.plot(param, rewards_eps, label="eps")
plt.plot(param, rewards_alp, label="alp")
plt.plot(param, rewards_Q, label="Q")

plt.xlabel('Parameter')
plt.ylabel('Reward')

plt.legend()
plt.show()
