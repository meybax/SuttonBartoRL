import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import scipy.special as sp
import math as m

class Policy:

    def __init__(self):
        self.actMap = np.zeros((20, 20))
        self.val = np.zeros((20, 20))

    def act(self, s):
        return self.actMap[s[0]-1, s[1]-1]

    def eval(self, env, acc, gamma):
        delta = -1
        while delta > acc or delta < 0:
            delta = 0
            for s in range(env.states.size):
                curr = env.states[s]
                oldVal = self.val[curr[0]-1, curr[1]-1]
                self.val[curr[0]-1, curr[1]-1] = self.value(curr, self.act(curr), gamma)
                delta = max(delta, abs(self.val[curr[0]-1, curr[1]-1] - oldVal))
                print("delta: " + str(delta))
                print("value: " + str(self.val[curr[0]-1, curr[1]-1]))
                print("old: " + str(oldVal))

    def iter(self, env, gamma):
        stable = True
        for s in range(env.states.size):
            curr = env.states[s]
            oldAction = self.actMap[curr[0]-1, curr[1]-1]
            actionValues = np.zeros(11)
            for a in range(11):
                actionValues[a] = self.value(curr, a - 5, gamma)
            opt = np.argmax(actionValues) - 5
            self.actMap[curr[0]-1, curr[1]-1] = opt
            if opt != oldAction:
                stable = False
            print("Optimal: " + str(opt) + "=" + str(actionValues[opt + 5]))
            print("Previous: " + str(oldAction) + "=" + str(self.val[curr[0]-1, curr[1]-1]))
        return stable

    def value(self, curr, action, gamma):
        sum = 0
        for n in range(env.states.size):
            next = env.states[n]
            rewards = env.getRewards(curr, action)
            for s in range(rewards.size):
                reward = rewards[s]
                prob = env.trans_prob(next, reward, curr, action)
                sum += prob * (reward + gamma * self.val[next[0]-1, next[1]-1])
        return sum

class Car_Env:

    def __init__(self):
        value = np.empty((), dtype=object)
        value[()] = (0, 0)
        states = np.full(400, value, dtype=object)
        for i in range(20):
            for j in range(20):
                states[i*20+j] = (i + 1, j + 1)
        self.states = states

    # assumes s_next is a valid state, no more than 20 cars at each location
    # assumes action is valid, no more than 5 movements
    # assumes reward is a valid combination 0f -2 and +10 for moving and renting cars
    def trans_prob(self, s_next, reward, s, action):
        s = (min(s[0] - action, 20), min(s[1] + action, 20))
        diff = (s_next[0] - s[0], s_next[1] - s[1])
        prob_client = self.skellam(3, 3, diff[0]) * self.skellam(2, 4, diff[1])

        exp_reward = abs(action) * -2
        diff_reward = reward - exp_reward
        prob_reward = self.poisson_sum(3, 4, diff_reward / 10)

        return prob_client * prob_reward

    # probability of poisson difference being n
    def skellam(self, a, b, n):
        return np.e**(-a-b) * (a/b) ** (n/2) * sp.iv(abs(n), 2*np.sqrt(a*b))

    # probability of poisson sum being n
    def poisson_sum(self, a, b, n):
        if (n < 0):
            return 0
        return np.e**(-a-b) * 1/m.factorial(int(n)) * (a+b)**n

    def getRewards(self, s, act):
        base = abs(act) * -2
        rewards = np.zeros(sum(s))
        for i in range(rewards.size):
            rewards[i] = i * 10 + base
        return rewards


pi = Policy()
env = Car_Env()
while True:
    pi.eval(env, 10, 0.9)
    stable = pi.iter(env, 0.9)
    if stable:
        break

cars = np.arange(20)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
fig.suptitle("Car Rental Policy Iteration")
ax.plot_surface(cars, cars, pi.val, cmap=plt.cm.coolwarm)
ax.set(ylabel="#Cars at second", xlabel="#Cars at first", zlabel="Value")
ax.title.set_text('Values')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(cars, cars, pi.actMap, cmap=plt.cm.coolwarm)
ax.set(ylabel="#Cars at second", xlabel="#Cars at first", zlabel="Value")
ax.title.set_text('Actions')
ax.invert_xaxis()

plt.show()