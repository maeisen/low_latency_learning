import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import math
import scipy
import scipy.stats
import pdb
from architecture_modified import cnngs
from scipy import sparse

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
        self.param_dim = int(num_param*action_dim)

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


#####################################################
########## Categorical ###########################
######################################################
class CategoricalDistribution(ProbabilityAction):
    #TODO: FIX THIS
    def __init__(self, action_dim, num_classes):
        super().__init__(num_classes, action_dim)

    def log_prob(self, params, selected_action):
        log_probs = 0
        output = tf.zeros((tf.shape(params)[0],self.action_dim*self.num_param))

        for i in np.arange(self.action_dim):
            ru_class = tf.gather(params, np.array(range(i*self.num_param, (i+1)*self.num_param)), axis=1)
            ru_class = tf.nn.softmax(ru_class, axis=1)
            probs = tf.matrix_diag(ru_class[:,selected_action[:,i]])
            log_probs = log_probs + tf.reduce_sum(tf.log(probs),axis=1)
            output[:, np.array(range(i*self.num_param, (i+1)*self.num_param)) ] = ru_class

        return log_probs, output



    def get_action(self, params):
        output_list = []
        N = np.shape(params)[0]

        action = np.zeros((N,self.action_dim))
        for i in np.arange(self.action_dim):
            ru_class = np.take(params, np.array(range(i*self.num_param, (i+1)*self.num_param)), axis=1)
            for j in np.arange(N):
                action[j,i] = np.where(np.random.multinomial(1,))[0][0]
                #action[j,i] = np.argmax(ru_class_sig[j,:])

        return action

#####################################################
########## Beta Distribution #######################
######################################################
class BetaDistribution(ProbabilityAction):
    def __init__(self, action_dim, lower_bound, upper_bound):
        super(BetaDistribution,self).__init__(2, action_dim)

        if lower_bound.shape != (action_dim,) or \
           upper_bound.shape != (action_dim,):
           raise Exception("Lower and upperbounds not the right shape")
        self.lower_bound = np.array(lower_bound, dtype=np.float32)
        self.upper_bound = np.array(upper_bound, dtype=np.float32)

    def log_prob(self, params, selected_action):
        alpha = tf.gather(params, np.array(range(self.action_dim)), axis=1, name='alpha')
        beta = tf.gather(params, np.array(range(self.action_dim, 2*self.action_dim)), axis=1, name='beta')

        alpha = tf.nn.softplus(alpha) + 1
        beta = tf.nn.softplus(beta) + 1 # TODO: add a little epsilon?

        output = tf.concat([alpha, beta], axis=1)

        dist = tf.distributions.Beta(alpha, beta)
        log_probs = dist.log_prob(selected_action/self.upper_bound)

        log_probs = tf.reduce_sum(log_probs, axis=1)
        return log_probs, output

    def get_action(self, params):
        alpha = params[:, :self.action_dim]
        beta = params[:, self.action_dim:]

        N = params.shape[0]

        lower_bound = np.vstack([self.lower_bound for _ in range(N)]) + 1e-6
        upper_bound = np.vstack([self.upper_bound for _ in range(N)]) - 1e-6

        action = np.random.beta(alpha,beta)*(upper_bound - lower_bound) + lower_bound

        return action

    def random_action(self,N):
        lower_bound = np.vstack([self.lower_bound for _ in range(N)]) + 1e-6
        upper_bound = np.vstack([self.upper_bound for _ in range(N)]) - 1e-6
        action = np.random.beta(2,2,size=(N,self.action_dim))*(upper_bound - lower_bound) + lower_bound
        return action

    def get_action2(self, params):
        alpha = params[:, :self.action_dim]
        beta = params[:, self.action_dim:]

        N = params.shape[0]

        lower_bound = np.vstack([self.lower_bound for _ in range(N)]) + 1e-6
        upper_bound = np.vstack([self.upper_bound for _ in range(N)]) - 1e-6

        action = alpha / (alpha+ beta) *(upper_bound - lower_bound) + lower_bound

        return action


#####################################################
########## Truncated Gaussian #######################
######################################################
class TruncatedGaussianDistribution(ProbabilityAction):
    def __init__(self, action_dim, lower_bound, upper_bound):
        super(TruncatedGaussianDistribution,self).__init__(2, action_dim)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)

    def log_prob(self, params, selected_action):
        pt1 = int(self.action_dim)
        pt2 = int(self.action_dim*2)

        mean = tf.gather(params, np.array(range(pt1)), axis=1, name='mean')
        std = tf.gather(params, np.array(range(pt1,pt2)), axis=1, name='std')
        mean = tf.nn.sigmoid(mean) * (self.upper_bound - self.lower_bound) + self.lower_bound
        std = tf.nn.sigmoid(std) * (.5-.01) + 0.01

        output = tf.concat([mean, std], axis=1) 
       # output = mean

        dist = tfp.distributions.Normal(mean, std)
        log_probs = dist.log_prob(selected_action) - tf.log((dist.cdf(self.upper_bound) - dist.cdf(self.lower_bound))) - tf.log(std)

        log_probs = tf.reduce_sum(log_probs, axis=1)

        return log_probs, output

    def get_action(self, params):
        pt1 = int(self.action_dim)
        pt2 = int(self.action_dim*2)

        mean = params[:, :pt1]
        std = params[:,pt1:pt2]

        N = params.shape[0]
        lower_bound = (np.vstack([self.lower_bound for _ in range(N)]) - mean) / std
        upper_bound = (np.vstack([self.upper_bound for _ in range(N)]) - mean) / std

        rate_action = scipy.stats.truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std)

        return rate_action

    def random_action(self, N):
        mean = (self.upper_bound+self.lower_bound)/2
        std = 0.5

        rate_action = scipy.stats.truncnorm.rvs(self.lower_bound, self.upper_bound, loc=mean, scale=std, size=(N,self.action_dim))

        return rate_action

    def get_action2(self, params):
        pt1 = int(self.action_dim)
        pt2 = int(self.action_dim*2)

        mean = params[:, :pt1]

        rate_action = mean
        return rate_action

#####################################################
########## Bernoulli  ###############################
######################################################
class BernoulliDistribution(ProbabilityAction):
    def __init__(self, action_dim, num_classes):
        super(BernoulliDistribution,self).__init__(1, action_dim)

    def log_prob(self, params, selected_action):

        p = tf.nn.sigmoid(params)*(.99-.01) + .01

        transmit_action = selected_action

        output = p

        log_probs2 = (transmit_action) * tf.log(p) + (1.0 - transmit_action)*tf.log(1-p)
        log_probs = tf.reduce_sum(log_probs2, axis=1)

        return log_probs, output

    def get_action(self, params):
        action = np.random.binomial(1,params)
        return action

    def random_action(self,N):
        action = np.random.binomial(1,0.3,size=(N,self.action_dim))
        return action

    def get_action2(self, params):
        action = (params>= 0.5).astype(float)
        return action

#####################################################
########## Beta/Bernoulli  ###############################
######################################################
class BetaBernoulliDistribution(ProbabilityAction):
    def __init__(self, num_users, lower_bound, upper_bound, num_classes):
        super(BetaBernoulliDistribution,self).__init__(num_classes+2, num_users*(num_classes+1))
        self.beta = BetaDistribution(num_users,lower_bound,upper_bound)
        self.bernoulli = BernoulliDistribution(num_users*num_classes, num_classes)
        self.num_users = num_users
        self.num_classes = num_classes
        self.param_dim = num_users*(num_classes+2)

    def log_prob(self, params, selected_action):

        pt1 = int(2*self.num_users)
        pt2 = int((2+self.num_classes)*self.num_users)

        

        ## Used for GNN implementation -- otherwise comment out ###
        param_dim = self.num_users*self.num_param
      #  output_list = []
      #  for i in range(self.num_param):
      #      output_list.append(tf.gather(params, np.array(range(i, param_dim,self.num_param)), axis=1))
      #  params2 = tf.concat(output_list,axis=1)

        beta_p = tf.gather(params, np.array(range(pt1)), axis=1, name='gp')
        bernoulli_p = tf.gather(params, np.array(range(pt1, pt2)), axis=1, name='bp')

        pt1 = int(self.num_users)
        pt2 = int((1+self.num_classes)*self.num_users)

        rate_action = tf.gather(selected_action, np.array(range(pt1)), axis=1, name='rate')
        transmit_action = tf.gather(selected_action, np.array(range(pt1,pt2)), axis=1, name='transmit')

        log_probs1, output1 = self.beta.log_prob(gaussian_p,rate_action)
        log_probs2, output2 = self.bernoulli.log_prob(bernoulli_p,transmit_action)

        output = tf.concat([output1, output2], axis=1)

        log_probs = log_probs1 + log_probs2

        return log_probs, output

    def get_action(self, params):
        pt1 = int(2*self.num_users)
        pt2 = int((2+self.num_classes)*self.num_users)

        beta_p = params[:,:pt1]
        bernoulli_p = params[:,pt1:pt2]

        action1 = self.beta.get_action(gaussian_p)
        action2 = self.bernoulli.get_action(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

        return action

    def random_action(self,N):

        action1 = self.gaussian.random_action(N)
        action2 = self.beta.random_action(N)

        action = np.concatenate([action1, action2], axis=1)

        return action


    def get_action2(self, params):
        pt1 = int(2*self.num_users)
        pt2 = int((2+self.num_classes)*self.num_users)

        gaussian_p = params[:,:pt1]
        beta_p = params[:,pt1:pt2]

        action1 = self.gaussian.get_action2(gaussian_p)
        action2 = self.beta.get_action2(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

        return action

#####################################################
########## Gaussian/Bernoulli  ###############################
######################################################
class GaussianBernoulliDistribution(ProbabilityAction):
    def __init__(self, num_users, lower_bound, upper_bound, num_classes):
        super(GaussianBernoulliDistribution,self).__init__(num_classes+2, num_users*(num_classes+1))
        self.gaussian = TruncatedGaussianDistribution(num_users,lower_bound,upper_bound)
        self.bernoulli = BernoulliDistribution(num_users*num_classes, num_classes)
        self.num_users = num_users
        self.num_classes = num_classes
        self.param_dim = num_users*(num_classes+2)

    def log_prob(self, params, selected_action):

        pt1 = int(2*self.num_users)
        pt2 = int((2+self.num_classes)*self.num_users)

        

        ## Used for GNN implementation -- otherwise comment out ###
        param_dim = self.num_users*self.num_param
      #  output_list = []
      #  for i in range(self.num_param):
      #      output_list.append(tf.gather(params, np.array(range(i, param_dim,self.num_param)), axis=1))
      #  params2 = tf.concat(output_list,axis=1)

        gaussian_p = tf.gather(params, np.array(range(pt1)), axis=1, name='gp')
        bernoulli_p = tf.gather(params, np.array(range(pt1, pt2)), axis=1, name='bp')

        pt1 = int(self.num_users)
        pt2 = int((1+self.num_classes)*self.num_users)

        rate_action = tf.gather(selected_action, np.array(range(pt1)), axis=1, name='rate')
        transmit_action = tf.gather(selected_action, np.array(range(pt1,pt2)), axis=1, name='transmit')

        log_probs1, output1 = self.gaussian.log_prob(gaussian_p,rate_action)
        log_probs2, output2 = self.bernoulli.log_prob(bernoulli_p,transmit_action)

        output = tf.concat([output1, output2], axis=1)

        log_probs = log_probs1 + log_probs2

        return log_probs, output

    def get_action(self, params):
        pt1 = int(2*self.num_users)
        pt2 = int((2+self.num_classes)*self.num_users)

        gaussian_p = params[:,:pt1]
        bernoulli_p = params[:,pt1:pt2]

        action1 = self.gaussian.get_action(gaussian_p)
        action2 = self.bernoulli.get_action(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

        return action

    def random_action(self,N):

        action1 = self.gaussian.random_action(N)
        action2 = self.bernoulli.random_action(N)

        action = np.concatenate([action1, action2], axis=1)

        return action


    def get_action2(self, params):
        pt1 = int(2*self.num_users)
        pt2 = int((2+self.num_classes)*self.num_users)

        gaussian_p = params[:,:pt1]
        bernoulli_p = params[:,pt1:pt2]

        action1 = self.gaussian.get_action2(gaussian_p)
        action2 = self.bernoulli.get_action2(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

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

def gnn_fc_model(state_dim, action_dim, num_param, layers=[5]*12, model=None, batch_size=200):

    is_train = tf.placeholder(tf.bool, name="is_train")
    L = len(layers)

    GSO = 'Adjacency'
    archit = 'no_pooling'
    f = [3]*12
    A = np.eye(state_dim)
    print(A.shape)
    gnn = cnngs(GSO, A,  # Graph parameters
            layers, f, [1]*L, [action_dim*num_param], # Architecture
            'temp', './', archit = archit,decay_steps=1)

    graph_input = tf.placeholder(tf.float32, [batch_size, state_dim+2,state_dim+2], name='graph_input')
    S = [graph_input]*L

    state_input = tf.placeholder(tf.float32, [batch_size, state_dim+2, 1], name='state_input')
    x = state_input
   # x = tf.expand_dims(x, 2)  # T x N x F=1 or N x F=1


    dropout = 1
   # T, N, F = x.get_shape()
    if archit == 'aggregation':
        maxP = min(S.shape[0],20)
        x = gt.collect_at_node(x,gnn.S,[gnn.R],maxP)

    with tf.variable_scope('policy'):
        for l in range(L):
            with tf.variable_scope('gsconv{}'.format(l+1)):
                if gnn.archit == 'hybrid':
                    # Padding:
                    Tx, Nx, Fx = x.get_shape()
                    Tx, Nx, Fx = int(Tx), int(Nx), int(Fx)
                    if Nx < N:
                        x = tf.pad(x, [[0,0],[0,int(N-Nx)],[0,0]])
                    # Diffusion:
                    RR = [int(x) for x in range(gnn.R[l])]
                    x = gt.collect_at_node(x,S,RR,gnn.P[l])

                with tf.name_scope('filter'):
                    Tx, Nx, Fx = x.get_shape()
                    x = gnn.filter(x, l, S[l])
                with tf.name_scope('pooling'):
                    x = gnn.pool(x, l)
                with tf.name_scope('nonlin'):
                    if l<L:
                        x = gnn.nonlin(x)
        T, N, F = x.get_shape()
       # x = tf.reshape(x, [int(T), int(N*F)])  # T x M (Recall M = N*F)
        x = tf.reshape(x, [-1, int(N*F)])  # T x M (Recall M = N*F)
        for l in range(len(gnn.M)-1):
            with tf.variable_scope('fc{}'.format(l+1)):
                x = gnn.fc(x, l)
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = gnn.fc(x, int(state_dim*num_param), relu=False)
            x = gnn.batch_norm(x)
    output = x
    print(output)
    return state_input, graph_input, is_train, output

def cnn_model(state_dim, action_dim, num_param, layers=[3]*32, model=None):
    with tf.variable_scope('policy'):

        is_train = True

        L = len(layers)



        state_input0 = tf.placeholder(tf.float32, [None, state_dim], name='state_input')
        state_input = tf.expand_dims(state_input0,axis=2)
        is_train = tf.placeholder(tf.bool, name="is_train")

        graph_input = tf.placeholder(tf.float32, [None, 1,1], name='graph_input')

       # ff = [2,3,5,10,10,5,3,2,2,1]
        ff = [1]*L

        channel = state_input[:,:int(2*state_dim/3),:]
        control = state_input[:,int(2*state_dim/3):,:]
        

        net1 = tf.expand_dims(control,axis=3)
        net2 = tf.reshape(channel,[-1,int(state_dim/3),2,1])

        print(net1)
        print(net2)

        net2 = tf.concat([net2,net1],axis=2)

        for idx, layer in enumerate(layers):
            # net1 = tf.contrib.layers.convolution1d(net1,
            #     kernel_size = layer,
            #     num_outputs = ff[idx],
            #     activation_fn=tf.nn.relu,
            #     weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            #     scope='layer_a_'+str(idx))

            net2 = tf.contrib.layers.convolution2d(net2,
                kernel_size = layer,
                num_outputs = ff[idx],
                activation_fn=tf.nn.relu,
                weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                scope='layer_b_'+str(idx))

        print(net2)

        net = tf.reshape(net2, [-1, 9])


        output = tf.contrib.layers.fully_connected(net,
            int(num_param),
            activation_fn=None,
            scope='output')
        print(output)
        output = tf.layers.batch_normalization(output, training=is_train)

        print(output)

    return state_input0, graph_input, is_train, output

def cnn_model2(state_dim, action_dim, num_param, layers=[5]*2, model=None):
    with tf.variable_scope('policy'):

        state_input0 = tf.placeholder(tf.float32, [None, state_dim], name='state_input')
        state_input = tf.expand_dims(state_input0,axis=2)
        graph_input = tf.placeholder(tf.float32, [None, 1,1], name='graph_input')
        is_train = tf.placeholder(tf.bool, name="is_train")

        #ff = [2,3,5,10,10,5,3,2,2,1]
        ff = [100, 1]

        channel = state_input[:,:int(3*state_dim/4),:]
        control = state_input[:,int(3*state_dim/4):,:]


        net1 = control
        net2 = tf.reshape(channel,[-1,int(state_dim/4),int(state_dim/16),1])
        print(net1)
        print(net2)
        
        for idx, layer in enumerate(layers):
            net1 = tf.contrib.layers.convolution1d(net1,
                kernel_size = layer,
                num_outputs = ff[idx],
                activation_fn=tf.nn.relu,
                weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                scope='layer_a_'+str(idx))

            net2 = tf.contrib.layers.convolution2d(net2,
                kernel_size = layer,
                num_outputs = ff[idx],
                activation_fn=tf.nn.relu,
                weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                scope='layer_b_'+str(idx))

        net2 = tf.reshape(net2,[-1,int((state_dim/4)*(state_dim/16)),ff[1]])
        print(net1)
        print(net2)
        net = tf.concat([net1[:,:,0], net2[:,:,0]], axis=1)

        print(net)
        print(num_param)

        output = tf.contrib.layers.fully_connected(net,
            int(num_param),
            activation_fn=None,
            scope='output')
        output = tf.layers.batch_normalization(output, training=is_train)

        print(output)

    return state_input0, graph_input, is_train, output

def mlp_model2(state_dim, action_dim, num_param, layers=[128, 1000, 256], model=None):
    res = False
    res_skip = 8
    with tf.variable_scope('policy'):

        state_input = tf.placeholder(tf.float32, [None, state_dim], name='state_input')
        graph_input = tf.placeholder(tf.float32, [None, 1,1], name='graph_input')
        is_train = tf.placeholder(tf.bool, name="is_train")

        channel = state_input[:,:int(3*state_dim/4)]
        control = state_input[:,int(3*state_dim/4):]

        #net = state_input

        [m,v] = tf.nn.moments(channel,axes=[0],keep_dims=False)
        channel = tf.nn.batch_normalization(channel,m,v,None,None,variance_epsilon=0.01)

        [m,v] = tf.nn.moments(control,axes=[0],keep_dims=False)
        control = tf.nn.batch_normalization(control,m,v,None,None,variance_epsilon=0.01)

        net = tf.concat([channel, control], axis=1)

    #    sorted_control_id = tf.argsort(control,axis=1)
    #    sorted_control = tf.sort(control, axis=1)

    #    channel = tf.reshape(channel,[-1,8,3])
    #    for i in np.arange(3):
    #        sorted_channel[:,:,i] = channel[sorted_control_id,:]

       # sorted_channel_id = tf.argsort(channel, axis=1)
       # sorted_channel = tf.sort(channel, axis=1)
    #    sorted_channel = tf.reshape(sorted_channel, [-1,24])
    #    net = tf.concat([sorted_channel, sorted_control], axis=1)

        
        regs = {}
        res_layer = 0
        for idx, layer in enumerate(layers):
            #regs[idx] = tf.contrib.layers.l2_regularizer(scale=0.1)
            if res and (np.mod(idx,res_skip)==0):
                net = res_layer + 0.5*tf.contrib.layers.fully_connected(net,
                    layer,
                    activation_fn=tf.nn.relu,
                    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'),
                    weights_regularizer = regs[idx],
                    scope='layer'+str(idx))
                res_layer = net
            else:
                net = tf.contrib.layers.fully_connected(net,
                    layer,
                    activation_fn=tf.nn.selu,
                    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN'),
                    scope='layer'+str(idx))                

        output = tf.contrib.layers.fully_connected(net,
            int(num_param),
            activation_fn=None,
            scope='output')
        output = tf.layers.batch_normalization(output, training=is_train)

        print(output)

    return state_input, graph_input, is_train, output

def mlp_model2_init(state_dim, action_dim, num_param, layers=[128, 64, 32], model=None):
    with tf.variable_scope('policy'):

        state_input = tf.placeholder(tf.float32, [None, state_dim], name='state_input')

        net = state_input
        for idx, layer in enumerate(layers):
            L = model.get_layer('dense' + idx*('_' + str(idx)))
            P = L.get_weights()
            net = tf.contrib.layers.fully_connected(net,
                layer,
                activation_fn=tf.nn.relu,
                weights_initializer = tf.initializers.constant(P[0]),
                biases_initializer = tf.initializers.constant(P[1]),
                scope='layer'+str(idx))

        L = model.get_layer('dense_2')
        P = L.get_weights()
        output = tf.contrib.layers.fully_connected(net,
            int(action_dim*num_param),
            activation_fn=None,
            weights_initializer = tf.initializers.constant(P[0]),
            biases_initializer = tf.initializers.constant(P[1]),
            scope='output')

    return state_input, output


def mlp_model_multi2(state_dim, action_dim, num_param, layers=[32]*32, model=None, num_users=12, num_classes=3):
    def _build_network(input, is_train, num_param, scope, layers):
        with tf.variable_scope(scope):
            net = input
            for idx, layer in enumerate(layers):
                net = tf.contrib.layers.fully_connected(net,
                    layer,
                    activation_fn = tf.nn.relu,
                    weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                    scope='layer'+str(idx))

            output = tf.contrib.layers.fully_connected(net,
                num_param,
                activation_fn=None,
                scope='output')
            output = tf.layers.batch_normalization(output, training=is_train)

        return output

    def _build_network_init(input, is_train, num_param, scope, layers, model, res=False, res_skip = 8):
        LL = len(layers)
        with tf.variable_scope(scope):
            net = input
            regs = {}
            res_layer = 0
            for idx, layer in enumerate(layers):
                L = model.get_layer('dense_' + str(idx))
                P = L.get_weights()
                regs[idx] = tf.contrib.layers.l2_regularizer(scale=0.01)
                if res and (np.mod(idx,res_skip)==0):
                    net = res_layer + 0.1*tf.contrib.layers.fully_connected(net,
                        layer,
                        activation_fn=tf.nn.relu,
                        weights_initializer = tf.initializers.constant(P[0]),
                 #       weights_regularizer = regs[idx],
                        biases_initializer = tf.initializers.constant(P[1]),
                        scope='layer'+str(idx))
                    res_layer = net
                else:
                    net = tf.contrib.layers.fully_connected(net,
                        layer,
                        activation_fn=tf.nn.relu,
                        weights_initializer = tf.initializers.constant(P[0]),
                 #       weights_regularizer = regs[idx],
                        biases_initializer = tf.initializers.constant(P[1]),
                        scope='layer'+str(idx))

            L = model.get_layer('dense_' + str(LL))
            P = L.get_weights()
            output = tf.contrib.layers.fully_connected(net,
                num_param,
                activation_fn=tf.nn.sigmoid_cross_entropy_with_logits,
                weights_initializer = tf.initializers.constant(P[0]),
                biases_initializer = tf.initializers.constant(P[1]),
                scope='output')
           # output = tf.layers.batch_normalization(output, training=is_train)
        return output

    state_input = tf.placeholder(tf.float32, [None, state_dim], name='state_input')
    graph_input = tf.placeholder(tf.float32, [None, 1,1], name='graph_input')
    is_train = tf.placeholder(tf.bool, name="is_train")
    output_list = []

    layers2 = [256, 128, 64]

    output1 = _build_network_init(state_input, is_train, 2*num_users, "rate_policy", layers2, model[1])
    output_list.append(output1[..., tf.newaxis])

    output2 = _build_network_init(state_input, is_train, num_classes*num_users, "transmit_policy", layers, model[0], res=False)
    output_list.append(output2[..., tf.newaxis])

    output_list = tf.concat(output_list, axis=1)
    output_list = tf.reshape(output_list, [tf.shape(output1)[0], 2*num_users +num_classes*num_users ])

    print(output_list)
    print(num_param)

    output = tf.contrib.layers.fully_connected(output_list,
        int(num_param),
        activation_fn=None,
        scope='output')
    output = tf.layers.batch_normalization(output, training=is_train)

    return state_input, graph_input, is_train, output



class ReinforcePolicy(object):
    def __init__(self,
        sys,
        model_builder=mlp_model,
        distribution=None,
        theta_lr = 5e-4,
        lambda_lr = 0.005,
        sess=None,
        model=None):

        self.is_train = True

        self.state_dim = sys.state_dim
        self.action_dim = sys.action_dim
        self.constraint_dim = sys.constraint_dim
        self.lambd = 10*np.ones((sys.constraint_dim, 1))

        self.model_builder = model_builder
        self.dist = distribution

        self.lambd_lr = lambda_lr
        self.theta_lr = theta_lr

        self.stats = RunningStats(64*100)

        self._build_model(sys.state_dim, sys.action_dim, model_builder, distribution,model)

        if sess == None:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.InteractiveSession(config=config)
            tf.global_variables_initializer().run()
        else:
            self.sess = sess



    def _build_model(self, state_dim, action_dim, model_builder, distribution, model):


        self.state_input, self.graph_input, self.is_train, self.output = model_builder(state_dim, action_dim, self.dist.num_param, model=model)


        tvars = tf.trainable_variables()
        self.g_vars = [var for var in tvars if 'policy/' in var.name or 'rate_policy/' in var.name or 'transmit_policy/' in var.name]

        self.selected_action = tf.placeholder(tf.float32, [None, action_dim], name='selected_action')


        self.log_probs, self.params = self.dist.log_prob(self.output, self.selected_action)

        self.l2_loss = tf.losses.get_regularization_loss()

        self.cost = tf.placeholder(tf.float32, [None], name='cost')

        self.loss = self.log_probs * (self.cost + self.l2_loss) 
        self.loss = tf.reduce_mean(self.loss)

        lr = self.theta_lr
     #   self.gradients = tf.train.AdamOptimizer(lr).compute_gradients(self.loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr)
            self.gradients = self.optimizer.compute_gradients(self.loss,var_list=self.g_vars)
            self.c_gs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self.gradients]
            self.optimize = self.optimizer.apply_gradients(self.c_gs)

    def normalize_graph(self,S):
        #SS = np.power(10,S/10)/5
        SS = np.matmul(S,np.transpose(S,(0,2,1)))
        n = SS.shape[1]
        SS[:,range(n), range(n)] = 0
        #SS = scipy.linalg.sqrtm(SS)
        norms = np.linalg.norm(SS,axis=(1,2))
        SS = SS/norms[:,None,None]

        return SS

    def bipartite_graph(self,S):
        #SS = np.power(10,S/10)/5
        N, m, n = S.shape
        SS = np.zeros((N,m+n,m+n))
        SS[:,0:m,m:] = S
        SS[:,m:,0:m] = np.transpose(S,(0,2,1))
        #SS = scipy.linalg.sqrtm(SS)
        norms = np.linalg.norm(SS,axis=(1,2))
        SS = SS/norms[:,None,None]

        return SS

    def combine_inputs(self,node,graph):
        l = (graph.shape[2]+1)*graph.shape[1]
        N = graph.shape[0]
        m = graph.shape[1]

       # pdb.set_trace()
        graph_p = np.reshape(graph,(N,-1))
        combined = np.zeros((N,l))
        
        mask1 = np.ones((N,l), dtype=bool)
        mask2 = np.zeros((N,l), dtype=bool)

        mask1[:,np.arange(0,l,m)] = False
        mask2[:,np.arange(0,l,m)] = True
        
        np.place(combined,mask2,node)
        np.place(combined,mask1,graph_p)

        combined = np.reshape(combined, (N,m,-1))
        return combined

    def combine_inputs2(self,node, graph):
        N, m, n = graph.shape
        l = (n+1)*m

        combined = np.reshape(node, (N,m,1))
        zz = np.ones((N,n,1))
        combined = np.concatenate([combined, zz], axis=1)


        return combined

    def get_action(self, inputs, S, pause=False, training=True):
       # Sn = self.normalize_graph(S)
       # c_inputs = self.combine_inputs(inputs, S)

        Sn = self.bipartite_graph(S)
        c_inputs = self.combine_inputs2(inputs, S)        


        fd = {self.state_input: c_inputs, self.graph_input: Sn, self.is_train: training}
        if pause:
            ppp = self.sess.run(self.output, feed_dict=fd)
            pdb.set_trace()
        
        params = self.sess.run(self.params, feed_dict=fd)
        if np.sum(np.isnan(params)) > 0:
            pdb.set_trace()

        action = self.dist.get_action(params)
       

        action = np.expand_dims(action, axis=2)
        inputs = np.expand_dims(inputs, axis=2)

        return inputs, action

    def get_action2(self, inputs, S, training=False):
       # Sn = self.normalize_graph(S)
       # c_inputs = self.combine_inputs(inputs, S)

        Sn = self.bipartite_graph(S)
        c_inputs = self.combine_inputs2(inputs, S)  

        fd = {self.state_input: c_inputs, self.graph_input: Sn, self.is_train: training}
        
        params = self.sess.run(self.params, feed_dict=fd)


        action = self.dist.get_action2(params)

        action = np.expand_dims(action, axis=2)
        inputs = np.expand_dims(inputs, axis=2)

        return inputs, action

    def random_action(self, inputs, pause=False):
        N = inputs.shape[0]
        action = self.dist.random_action(N)

        action = np.expand_dims(action, axis=2)
        inputs = np.expand_dims(inputs, axis=2)

        return inputs, action


    def learn(self, inputs, actions, f0, f1,S):
        """
        Args:acti
            inputs (TYPE): N by m
            actions (TYPE): N by m
            f0 (TYPE): N by 1
            f1 (TYPE): N by p

        Returns:
            TYPE: Description
        """

       # Sn = self.normalize_graph(S)
       # c_inputs = self.combine_inputs(inputs, S)

        Sn = self.bipartite_graph(S)
        c_inputs = self.combine_inputs2(inputs, S)  

        cost = f0 + np.dot(f1, self.lambd)
        cost = np.reshape(cost, (-1))
        self.stats.push_list(cost)
        cost_minus_baseline = cost - self.stats.get_mean()

        # improve policy weights
        # policy gradient step
      #  print(c_inputs.shape)
        fd = {self.state_input: c_inputs,
              self.graph_input: Sn,
              self.selected_action: actions,
              self.is_train: True,
              self.cost: cost_minus_baseline}
        grads = 0


     #   gradients = self.sess.run(self.gradients,feed_dict=fd)
     #   grads = np.zeros(len(gradients))
     #   for i in np.arange(len(gradients)):
     #       grads[i] = np.linalg.norm(gradients[i][0])

        output = self.sess.run(self.output, feed_dict=fd)

        loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd)


        # gradient ascent step on lambda
        delta_lambd = np.mean(f1, axis=0)
        delta_lambd = np.reshape(delta_lambd, (-1, 1))
     #   delta_lambd = np.maximum(0,delta_lambd)
        self.lambd += delta_lambd * self.lambd_lr

        # project lambd into positive orthant
        self.lambd = np.maximum(self.lambd, 0)

        # self.lambd_lr *= 0.9997
        self.lambd_lr *= 0.99999

        # TODO: learning rate decrease on lambd
        return cost, grads



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


    dist = tfp.distributions.Normal(mean, var)

    log_probs = dist.log_prob(selected_action) - tf.log(dist.cdf(upper_bound) - dist.cdf(lower_bound))



    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    fd = {params: np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
          selected_action: np.array([[1, 3, 5], [7, 9, 25]])}
    m, v, o, l = sess.run([mean, var, output, log_probs], feed_dict=fd)


    import pdb
    pdb.set_trace()



