import numpy as np
import tensorflow as tf
import math
import scipy
import scipy.stats


# slow implmenetation of running average
class RunningStats(object):
    def __init__(self, N):
        self.N = N
        self.vals = []
        self.num_filled = 0

    def push(self, val):
        if self.num_filled == self.N:
            self.vals.pop(0)
            self.vals.append(val)
        else:
            self.vals.append(val)
            self.num_filled += 1

    def push_list(self, vals):
        num_vals = len(vals)

        self.vals.extend(vals)
        self.num_filled += num_vals
        if self.num_filled >= self.N:
            diff = self.num_filled - self.N
            self.num_filled = self.N
            self.vals = self.vals[diff:]

    def get_mean(self):
        return np.mean(self.vals[:self.num_filled])

    def get_std(self):
        return np.std(self.vals[:self.num_filled])

    def get_mean_n(self, n):
        start = max(0, self.num_filled-n)
        return np.mean(self.vals[start:self.num_filled])




class SchedulingPolicy(object):
    def __init__(self, state_dim, action_dim, constraint_dim, sess=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim
        self.lambd = np.ones((constraint_dim, 1))



        self.stats = RunningStats(64*100)

        self._build_model(state_dim, action_dim)

        if sess == None:
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run() 
        else:
            self.sess = sess


    def _build_model(self, state_dim, action_dim):
        with tf.variable_scope('policy'):
            self.state_input = tf.placeholder(tf.float32, [None, state_dim], name='state_input')


            layer1 = tf.contrib.layers.fully_connected(self.state_input,
                32,
                activation_fn=tf.nn.relu,
                scope='layer1')

            layer2 = tf.contrib.layers.fully_connected(layer1,
                16,
                activation_fn=tf.nn.relu,
                scope='layer2')


            # # gaussian distribution
            # self.output = tf.contrib.layers.fully_connected(layer1,
            #     2 * action_dim,
            #     activation_fn=None,
            #     scope='output')


            # self.mean = tf.gather(self.output, np.array(range(action_dim)), axis=1)
            # self.var = tf.gather(self.output, np.array(range(action_dim, 2*action_dim)), axis=1)
            # self.var = tf.nn.sigmoid(self.var) # replace with sigmoid
            # self.mean = tf.nn.sigmoid(self.mean) * 10


            # # exponential distribution
            # self.output = tf.contrib.layers.fully_connected(layer1,
            #     action_dim,
            #     activation_fn=None,
            #     scope='output')
            # self.inv_mean = tf.nn.sigmoid(self.output) * 5 + 0.05


            # gamma distribution with fixed shape alpha=2
            self.output = tf.contrib.layers.fully_connected(layer1,
                action_dim,
                activation_fn=None,
                scope='output')
            self.alpha = 2
            #action_rs = self.output[:,0:action_dim].reshape(100,-1,24)
            self.beta = tf.nn.softmax(self.output)


            self.selected_action = tf.placeholder(tf.float32, [None, action_dim], name='action')
            self.cost = tf.placeholder(tf.float32, [None], name='cost')


            # # exponential distribution
            # self.log_probs = tf.log(self.inv_mean) - self.inv_mean * self.selected_action
            # self.log_probs = tf.reduce_sum(self.log_probs, axis=1)

            # gamma distribution
            #self.log_probs = self.alpha * tf.log(self.beta) + \
            #    (self.alpha - 1) * tf.log(self.selected_action) - \
            #    self.beta * self.selected_action - \
            #    np.log(scipy.special.gamma(self.alpha))
            # self.log_probs = tf.reduce_sum(self.log_probs, axis=1)

            # multinomial distribution
            print(self.beta.shape)
            self.log_probs = tf.log(self.beta[:,self.selected_action]) 
            self.log_probs = tf.reduce_sum(self.log_probs, axis=1)


            self.loss = self.log_probs * self.cost
            self.loss = tf.reduce_mean(self.loss)

            lr = 5e-4
            self.optimize = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def get_action(self, inputs):
        fd = {self.state_input: inputs}

        # # gaussian distribution
        # mean, var = self.sess.run([self.mean, self.var], feed_dict=fd)
        # action = np.random.normal(mean, var)
        # action[action < 0] = 0.05

        # # exponential distribution
        # inv_mean = self.sess.run(self.inv_mean, feed_dict=fd)
        # action = np.random.exponential(inv_mean)

        # multinomial distribution
        beta = self.sess.run(self.beta, feed_dict=fd)
        action = np.random.multinomial(1,self.beta)
        

        return action

    def learn(self, inputs, actions, f0, f1):
        """
        Args:
            inputs (TYPE): N by m
            actions (TYPE): N by m
            f0 (TYPE): N by 1
            f1 (TYPE): N by p

        Returns:
            TYPE: Description
        """
        cost = f0 + np.reshape(np.sum(self.lambd * f1, axis=1), (-1, 1))

        cost = np.reshape(cost, (-1))

        self.stats.push_list(cost)
        cost_minus_baseline = cost - self.stats.get_mean()

        # improve policy weights
        # policy gradient step
        fd = {self.state_input: inputs,
              self.selected_action: actions,
              self.cost: cost_minus_baseline}
        loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd)

        # gradient ascent step on lambda
        delta_lambd = np.mean(f1, axis=0)
        lambd_lr = 0.001
        self.lambd += delta_lambd * lambd_lr

        # project lambd into positive orthant
        self.lambd = np.maximum(self.lambd, 0)

        # TODO: learning rate decrease on lambd
        return loss


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from control_system import *
    from functools import partial

    tf.set_random_seed(0)
    np.random.seed(0)

    mu = 2
    num_channels = 3
    pmax = 10

    sys = WirelessSchedulingSystem(num_channels, pmax=pmax, mu=mu)

    policy = SchedulingePolicy(sys.state_dim, sys.action_dim, sys.constraint_dim)

    batch_size = 64
    lambd_history = []
    f0_history = []
    f1_history = []
    for k in range(1000):
        h = sys.sample(batch_size)
        actions = policy.get_action(h)
        f0 = sys.f0(h, actions)
        f1 = sys.f1(h, actions)

        policy.learn(h, actions, f0, f1)

        lambd_history.append(np.asscalar(policy.lambd))
        f0_history.append(np.asscalar(np.mean(f0)))
        f1_history.append(np.asscalar(np.mean(f1)))

        print("Iteration " + str(k))
        print("====================")
        print("f0: " + str(np.mean(f0)))
        print("f1: " + str(np.mean(f1)))
        print("lambd: " + str(policy.lambd))


    plt.plot(lambd_history)
    plt.savefig("lambd.png")

    plt.cla()
    plt.plot(f0_history)
    plt.savefig("f0.png")

    plt.cla()
    plt.plot(f1_history)
    plt.savefig("f1.png")