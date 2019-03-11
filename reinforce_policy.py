import numpy as np
import tensorflow as tf
import math
import scipy
import scipy.stats
import pdb

# N is batch size
# S is state_dim
# A is action_dim
# C is constraint_dim
# P is number of parameters per action distribution

# distributions with multiple parameters are output so that the
# first action_dim number of outputs are the first parameter of each of the distributions

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



class ProbabilityAction(object):
    def __init__(self, num_param, action_dim):
        """
        Class that implements various probabilistic actions

        Args:
            num_param (TYPE): number of parameters for a single distribution (P)
            example: single variable gaussian will have two parameters: mean and variance
        """
        self.num_param = num_param
        self.action_dim = action_dim

    def log_prob(self, params, selected_action):
        """
        Given a batch of distribution parameters
        and selected actions.
        Compute log probability of those actions

        Args:
            params (TYPE): N by (A*P) Tensor
            selected_action (TYPE): N by A Tensor

        Returns:
            Length N Tensor of log probabilities, N by (A*P) Tensor of corrected parameters
        """
        raise Exception("Not Implemented")

    def get_action(self, params):
        """

        Args:
            params (TYPE): N by (A*P) Numpy array

        Returns: N by A Numpy array of sampled actions
        """
        raise Exception("Not Implemented")


class GammaDistribution2(ProbabilityAction):
    """
    Gamma Distribution with fixed alpha=2
    """
    def __init__(self, action_dim, lower_bound, upper_bound):
        self.alpha = 2
        super().__init__(1, action_dim)

    def log_prob(self, params, selected_action):
        beta = tf.nn.sigmoid(params) * 4

        log_probs = self.alpha * tf.log(beta) + \
                (self.alpha - 1) * tf.log(selected_action) - \
                beta * selected_action - \
                np.log(scipy.special.gamma(self.alpha))
        log_probs = tf.reduce_sum(log_probs, axis=1)

        return log_probs, beta

    def get_action(self, params):
        # params is beta
        action = np.random.gamma(shape=self.alpha, scale=1.0/params)
        return action


class GaussianDistribution(ProbabilityAction):
    #TODO: FIX THIS
    def __init__(self, action_dim, lower_bound, upper_bound):
        super().__init__(2, action_dim)

    def log_prob(self, params, selected_action):
        mean = tf.gather(params, np.array(range(self.action_dim)), axis=1, name='mean')
        var = tf.gather(params, np.array(range(self.action_dim, 2*self.action_dim)), axis=1, name='var')

        mean = tf.nn.sigmoid(mean) * 10
        var = tf.nn.sigmoid(var) * 1
        output = None

        # TODO: this can create NaNs when self.var is small
        log_probs = -0.5 * tf.log(2*np.pi*var) - tf.square(selected_action - mean) / (2. * var)
        log_probs = tf.reduce_sum(log_probs, axis=1)

        return log_probs, outputex

    def get_action(self, params):
        mean = params[:, :action_dim]
        var = params[:, action_dim:]

        print("mean: " + str(mean))
        print("var: " + str(var))

        action = np.random.normal(mean, var) # TODO: this is not actually var, its std
        # # restrict actions to be at least some small positive value
        # action[action < 0.05] = 0.05
        return action

class ClassificationDistribution(ProbabilityAction):
    #TODO: FIX THIS
    def __init__(self, action_dim, num_classes):
        super().__init__(num_classes, action_dim)

    def log_prob(self, params, selected_action):
        log_probs = 0
        output_list = []
        output = tf.zeros((tf.shape(params)[0],1))
        #print(params)
        for i in np.arange(self.action_dim):
            ru_class = tf.gather(params, np.array(range(i*self.num_param, (i+1)*self.num_param)), axis=1)
            ru_class_sig = 1 / (1 + tf.math.exp(-ru_class))
            log_probs = log_probs + tf.reduce_sum(tf.log(ru_class_sig)*(selected_action[i]==i),axis=1)
            output = tf.concat([output, ru_class_sig],axis=1)
            #output_list.append(ru_class_sig)
        #print(tf.stack(output_list).shape)

        return log_probs, output[:,1:]

    def get_action(self, params):
        output_list = []
        N = np.shape(params)[0]

        action = np.zeros((N,self.action_dim))
        for i in np.arange(self.action_dim):
            ru_class = np.take(params, np.array(range(i*self.num_param, (i+1)*self.num_param)), axis=1)
            ru_class_sig = 1 / (1 + np.exp(-ru_class))
            for j in np.arange(N):
                action[j,i] = np.where(np.random.multinomial(1,ru_class_sig[j,:]/np.sum(ru_class_sig[j,:])))[0][0]
                #action[j,i] = np.argmax(ru_class_sig[j,:])

            
        #action = np.stack(output_list)
        return action

class ClassificationDistribution2(ProbabilityAction):
    #TODO: FIX THIS
    def __init__(self, action_dim, num_classes):
        super().__init__(num_classes, action_dim)

    def log_prob(self, params, selected_action):
        log_probs = 0
        output_list = []
        output = tf.zeros((tf.shape(params)[0],1))
        #print(params)
        for i in np.arange(self.action_dim):
            ru_class = tf.gather(params, np.array(range(i*self.num_param, (i+1)*self.num_param)), axis=1)
            ru_class_sig = 1 / (1 + tf.math.exp(-ru_class))
            log_probs = log_probs + tf.reduce_sum(tf.log(ru_class_sig)*(selected_action[i]==i),axis=1)
            output = tf.concat([output, ru_class_sig],axis=1)
            #output_list.append(ru_class_sig)
        #print(tf.stack(output_list).shape)

        return log_probs, output[:,1:]

    def get_action(self, params):
        output_list = []
        N = np.shape(params)[0]

        action = np.zeros((N,self.action_dim))
        for i in np.arange(self.action_dim):
            ru_class = np.take(params, np.array(range(i*self.num_param, (i+1)*self.num_param)), axis=1)
            ru_class_sig = 1 / (1 + np.exp(-ru_class))
            for j in np.arange(N):
                #action[j,i] = np.where(np.random.multinomial(1,ru_class_sig[j,:]/np.sum(ru_class_sig[j,:])))[0][0]
                action[j,i] = np.argmax(ru_class_sig[j,:])

            
        #action = np.stack(output_list)
        return action

class TruncatedGaussianBernoulliDistribution(ProbabilityAction):
    def __init__(self, action_dim, lower_bound, upper_bound):
        super(TruncatedGaussianBernoulliDistribution,self).__init__(1.5, action_dim)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)

    def log_prob(self, params, selected_action):

        pt1 = int(self.action_dim/2)
        pt2 = int(self.action_dim)
        pt3 = int(1.5*self.action_dim)

        mean = tf.gather(params, np.array(range(pt1)), axis=1, name='mean')
        std = tf.gather(params, np.array(range(pt1, pt2)), axis=1, name='std')
        alpha = tf.gather(params, np.array(range(pt2, pt3)), axis=1, name='alpha')

        mean = tf.nn.sigmoid(mean) * (self.upper_bound - self.lower_bound) + self.lower_bound
        #std = tf.nn.sigmoid(std) * np.sqrt(self.upper_bound - self.lower_bound) + np.sqrt(self.lower_bound)
        std = tf.nn.sigmoid(std)*1+.1
        p = tf.nn.sigmoid(alpha) *(.99-.01) + .01

        rate_action = tf.gather(selected_action, np.array(range(pt1)), axis=1, name='rate')
        transmit_action = tf.gather(selected_action, np.array(range(pt1,pt2)), axis=1, name='transmit')

        output = tf.concat([mean, std, p], axis=1)

        dist = tf.distributions.Bernoulli(probs=p)
        log_probs2 = (transmit_action) * tf.log(p) + (1.0 - transmit_action)*tf.log(1-p)
        log_probs2 = tf.reduce_sum(log_probs2, axis=1)

        

        dist = tf.distributions.Normal(mean, std)
        log_probs = dist.log_prob(rate_action) - tf.log((dist.cdf(self.upper_bound) - dist.cdf(self.lower_bound)))

        log_probs = tf.reduce_sum(log_probs, axis=1) + log_probs2

        return log_probs, output

    def get_action(self, params):
        pt1 = int(self.action_dim/2)
        pt2 = int(self.action_dim)
        pt3 = int(1.5*self.action_dim)

        mean = params[:, :pt1]
        std = params[:, pt1:pt2]
        alpha = params[:,pt2:pt3]

        if np.sum(np.isnan(alpha)) > 0:
            pdb.set_trace()


        N = params.shape[0]

        lower_bound = (np.vstack([self.lower_bound for _ in range(N)]) - mean) / std
        upper_bound = (np.vstack([self.upper_bound for _ in range(N)]) - mean) / std

       # lower_bound = (np.vstack([self.lower_bound for _ in range(N)]))
       # upper_bound = (np.vstack([self.upper_bound for _ in range(N)]))
        rate_action = scipy.stats.truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std)
       # rate_action = mean

       # transmit_action = (alpha > 0.5)
       # transmit_action = transmit_action.astype(float)

        transmit_action = np.random.binomial(1,alpha)

        action = np.concatenate([rate_action, transmit_action], axis=1)

        #dist = scipy.stats.truncnorm(lower_bound, upper_bound, loc=mean, scale=std)
        #action = dist.rvs()

        return action


    def get_action2(self, params):
        pt1 = int(self.action_dim/2)
        pt2 = int(self.action_dim)
        pt3 = int(1.5*self.action_dim)

        mean = params[:, :pt1]
        std = params[:, pt1:pt2]
        alpha = params[:,pt2:pt3]

        rate_action = mean

       # transmit_action = (alpha > 0.5)
       # transmit_action = transmit_action.astype(float)

        transmit_action = np.random.binomial(1,alpha)

        action = np.concatenate([rate_action, transmit_action], axis=1)

        #dist = scipy.stats.truncnorm(lower_bound, upper_bound, loc=mean, scale=std)
        #action = dist.rvs()

        return action


def mlp_model(state_dim, action_dim, num_param):
    with tf.variable_scope('policy'):
        state_input = tf.placeholder(tf.float32, [None, state_dim], name='state_input')

        layer1 = tf.contrib.layers.fully_connected(state_input,
            32,
            activation_fn=tf.nn.relu,
            scope='layer1')

        layer2 = tf.contrib.layers.fully_connected(layer1,
            16,
            activation_fn=tf.nn.relu,
            scope='layer2')

        output = tf.contrib.layers.fully_connected(layer2,
            int(action_dim*num_param),
            activation_fn=None,
            scope='output')

    return state_input, output


def mlp_model2(state_dim, action_dim, num_param, layers=[128, 64, 32]):
    with tf.variable_scope('policy'):

        state_input = tf.placeholder(tf.float32, [None, state_dim], name='state_input')

        net = state_input
        for idx, layer in enumerate(layers):
            net = tf.contrib.layers.fully_connected(net,
                layer,
                activation_fn=tf.nn.relu,
                scope='layer'+str(idx))

        output = tf.contrib.layers.fully_connected(net,
            int(action_dim*num_param),
            activation_fn=None,
            scope='output')

    return state_input, output

def mlp_model_multi2(state_dim, action_dim, num_param, layers=[32, 16]):
    def _build_network(input, num_param, scope, layers):
        with tf.variable_scope(scope):
            net = input
            for idx, layer in enumerate(layers):
                net = tf.contrib.layers.fully_connected(net,
                    layer,
                    activation_fn = tf.nn.relu,
                    scope='layer'+str(idx))

            output = tf.contrib.layers.fully_connected(net,
                num_param,
                activation_fn=None,
                scope='output')
        return output

    state_input = tf.placeholder(tf.float32, [None, state_dim], name='state_input')
    param_multiplier = [1, 0.5]
    output_list = []
    for i in range(2):
       # single_input = tf.slice(state_input, [0, i], [-1, 1])

        output = _build_network(state_input, int(action_dim*param_multiplier[i]), "agent" + str(i), layers)
        output_list.append(output[..., tf.newaxis])

    output_list = tf.concat(output_list, axis=1)
    output_list = tf.reshape(output_list, [tf.shape(output)[0], -1])

    return state_input, output_list



class ReinforcePolicy(object):
    def __init__(self,
        state_dim,
        action_dim,
        constraint_dim,
        model_builder=mlp_model,
        distribution=None,
        theta_lr = 5e-4,
        lambda_lr = 0.005,
        sess=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim
        self.lambd = 10*np.ones((constraint_dim, 1))

        self.model_builder = model_builder
        self.dist = distribution

        self.lambd_lr = lambda_lr
        self.theta_lr = theta_lr

        self.stats = RunningStats(64*100)

        self._build_model(state_dim, action_dim, model_builder, distribution)

        if sess == None:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.InteractiveSession(config=config)
            tf.global_variables_initializer().run()
        else:
            self.sess = sess


    def _build_model(self, state_dim, action_dim, model_builder, distribution):

        # self.state_input, self.output, self.selected_action, self.log_probs =\
        #     model_builder(state_dim, action_dim, self.dist)

        self.state_input, self.output = model_builder(state_dim, action_dim, self.dist.num_param)

        self.selected_action = tf.placeholder(tf.float32, [None, action_dim], name='selected_action')


        self.log_probs, self.params = self.dist.log_prob(self.output, self.selected_action)

        self.cost = tf.placeholder(tf.float32, [None], name='cost')

        self.loss = self.log_probs * self.cost
        self.loss = tf.reduce_mean(self.loss)

        lr = self.theta_lr
        self.optimize = tf.train.AdamOptimizer(lr).minimize(self.loss)



    def get_action(self, inputs):
        fd = {self.state_input: inputs}
        params = self.sess.run(self.params, feed_dict=fd)
        if np.sum(np.isnan(params)) > 0:
            pdb.set_trace()

        action = self.dist.get_action(params)

        return inputs, action


    def learn(self, inputs, actions, f0, f1):
        """
        Args:acti
            inputs (TYPE): N by m
            actions (TYPE): N by m
            f0 (TYPE): N by 1
            f1 (TYPE): N by p

        Returns:
            TYPE: Description
        """
        cost = f0 + np.dot(f1, self.lambd)
        cost = np.reshape(cost, (-1))
        self.stats.push_list(cost)
        cost_minus_baseline = cost - self.stats.get_mean()

        # improve policy weights
        # policy gradient step
        fd = {self.state_input: inputs,
              self.selected_action: actions,
              self.cost: cost_minus_baseline}

        output = self.sess.run(self.output, feed_dict=fd)
        loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd)


        # gradient ascent step on lambda
        delta_lambd = np.mean(f1, axis=0)
        delta_lambd = np.reshape(delta_lambd, (-1, 1))
        self.lambd += delta_lambd * self.lambd_lr

        # project lambd into positive orthant
        self.lambd = np.maximum(self.lambd, 0)

        # self.lambd_lr *= 0.9997
        self.lambd_lr *= 0.99997

        # TODO: learning rate decrease on lambd
        return loss



if __name__ == '__main__':
    action_dim = 3

    upper_bound= np.ones(3, dtype=np.float32) * 2
    lower_bound = np.ones(3, dtype=np.float32) * 1


    params = tf.placeholder(tf.float32, [None, action_dim*2])
    selected_action = tf.placeholder(tf.float32, [None, action_dim])



    mean = tf.gather(params, np.array(range(action_dim)), axis=1, name='mean')
    var = tf.gather(params, np.array(range(action_dim, 2*action_dim)), axis=1, name='var')

    mean = tf.nn.sigmoid(mean) * (upper_bound - lower_bound) + lower_bound
    var = tf.nn.sigmoid(var) * np.sqrt(upper_bound - lower_bound) # TODO: add a little epsilon?

    output = tf.concat([mean, var], axis=1)


    dist = tf.distributions.Normal(mean, var)

    log_probs = dist.log_prob(selected_action) - tf.log(dist.cdf(upper_bound) - dist.cdf(lower_bound))



    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    fd = {params: np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
          selected_action: np.array([[1, 3, 5], [7, 9, 25]])}
    m, v, o, l = sess.run([mean, var, output, log_probs], feed_dict=fd)


    import pdb
    pdb.set_trace()



