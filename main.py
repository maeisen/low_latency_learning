import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.io
from scipy import sparse

from control_system import *
from reinforce_policy_T import ReinforcePolicy
from reinforce_policy_T import *

#######################
## UTILITY FUNCTIONS ##
#######################
def moving_average(data, window=10):
    cumsum = np.cumsum(data)
    moving_sum = cumsum[window:] - cumsum[:-window]
    moving_avg = moving_sum / window


    moving_avg = np.concatenate((np.zeros(window), moving_avg))
    moving_avg[:window] = cumsum[:window] / window
    return moving_avg

def run_sys(sys, policy, num_iter, batch_size=64):

    num_iters_run = 1000


    history_dict = {'lambd': [],
                   'f0': [],
                   'f1': [],
                   'p': [],
                   'Ao': [],
                   'Ac': [],
                   'x': []}
    runtime_dict = {'f0': [],
                   'f0_2': [],
                   'f0_3': [],
                   'f0_4': [],
                   'p': [],
                   'p2': [],
                   'p3': [],
                   'p4': [],
                   'T': [],
                   'x': [],
                   'x2': [],
                   'x3': [],
                   'x4': [],
                   'Ao':[],
                   'Ac':[]}
    runtime_dict2 = {'f0': [],
                   'f0_2': [],
                   'f0_3': [],
                   'f0_4': [],
                   'p': [],
                   'p2': [],
                   'p3': [],
                   'p4': [],
                   'T': [],
                   'x': [],
                   'x2': [],
                   'x3': [],
                   'x4': [],
                   'Ao':[],
                   'Ac':[]}
    history_dict['Ao'].append(sys.Ao)
    history_dict['Ac'].append(sys.Ac)
    runtime_dict['Ao'].append(sys.Ao)
    runtime_dict['Ac'].append(sys.Ac)
    runtime_dict2['Ao'].append(sys.Ao)
    runtime_dict2['Ac'].append(sys.Ac)

    ##### TRAIN ###############################################
    for k in range(num_iter):
        state0 = sys.sample(batch_size)

        if k%1000 == 0:
            print("Iteration " + str(k))

        states, actions = policy.get_action(state0)

        f0 = sys.f0(states, actions)
        f1 = sys.f1(states, actions)
        L = f0 + np.dot(f1, policy.lambd)


        history_dict['lambd'].append(policy.lambd)

        history_dict['f0'].append(np.mean(f0))
        history_dict['f1'].append(np.mean(f1,axis=0))

        if k%1000 == 0:
            #print(actions)
            history_dict['x'].append(states)
            history_dict['p'].append(actions)

        policy.learn(states[:,:,0], actions[:,:,0], f0, f1)

    pdb.set_trace()

    ##### RUN ##########################
    state0 = sys.sample(2)
    state1 = np.copy(state0)
    state2 = np.copy(state1)
    state3 = np.copy(state1)
    state4 = np.copy(state2)
    for k in range(num_iters_run):
        
        print("Run Iteration " + str(k))
        state0 = sys.sample(2)
        #state1 = sys.sample(2)
        #state2 = np.copy(state1)
        #state3 = np.copy(state1)
        #state4 = np.copy(state2)

        actions = policy.get_action(state1)
        T = sys.get_time(actions)
        actions2 = sys.round_robin(state2, T[0])
        actions3 = sys.priority_ranking(state3, T[0])
        actions4 = sys.calls(state4, T[0])

        f0_1 = sys.f0(state1,actions)
        f0_2 = sys.f0_m(state2,actions2)
        f0_3 = sys.f0_m(state3,actions3)
        f0_4 = sys.f0_m(state4,actions4)

        runtime_dict['f0'].append(f0_1)
        runtime_dict['f0_2'].append(f0_2)
        runtime_dict['f0_3'].append(f0_3)
        runtime_dict['f0_4'].append(f0_4)
        runtime_dict['p'].append(actions)
        runtime_dict['p2'].append(actions2)
        runtime_dict['p3'].append(actions3)
        runtime_dict['p4'].append(actions4)
        runtime_dict['T'].append(T[0])
        runtime_dict['x'].append(state1)
        runtime_dict['x2'].append(state2)
        runtime_dict['x3'].append(state3)
        runtime_dict['x4'].append(state4)

        state1 = sys.update_system(state1,actions,2,rate_select=True)
        state2 = sys.update_system(state2,actions2,2,rate_select=False)
        state3 = sys.update_system(state3,actions3,2,rate_select=False)
        state4 = sys.update_system(state4,actions4,2,rate_select=False)

        actions = policy.get_action(state0)
        T = sys.get_time(actions)
        actions2 = sys.round_robin(state0, T[0])
        actions3 = sys.priority_ranking(state0, T[0])
        actions4 = sys.calls(state0, T[0])

        f0_1 = sys.f0(state0,actions)
        f0_2 = sys.f0_m(state0,actions2)
        f0_3 = sys.f0_m(state0,actions3)
        f0_4 = sys.f0_m(state0,actions4)

        runtime_dict2['f0'].append(f0_1)
        runtime_dict2['f0_2'].append(f0_2)
        runtime_dict2['f0_3'].append(f0_3)
        runtime_dict2['f0_4'].append(f0_4)
        runtime_dict2['p'].append(actions)
        runtime_dict2['p2'].append(actions2)
        runtime_dict2['p3'].append(actions3)
        runtime_dict2['p4'].append(actions4)
        runtime_dict2['T'].append(T[0])
        runtime_dict2['x'].append(state0)

    return history_dict, runtime_dict, runtime_dict2

def save_data(data, filename):
    scipy.io.savemat(filename, data)

def save_rt_data(data, filename):
    scipy.io.savemat(filename, data)


def generate_training_set(sys, num_samples, filename):
    print("Generation Training Set")

    history_dict = {'f0': [],
                    'p': [],
                    'x': [],
                    'Ao':[],
                    'Ac':[]}
    history_dict['Ao'].append(sys.Ao)
    history_dict['Ac'].append(sys.Ac)

    for k in range(num_samples):
        states = sys.sample(1)

        if k%10000 == 0:
            print("Iteration " + str(k))

        actions3 = sys.priority_ranking(states, sys.tmax)
        f0_3 = sys.f0_m(states,actions3)

        history_dict['f0'].append(f0_3)
        history_dict['p'].append(actions3)
        history_dict['x'].append(states)
    scipy.io.savemat(filename, history_dict)


####################
## TEST FUNCTIONS ##
####################
def wireless_control_test():
    mu = 2 # parameter for exponential distribution of wireless channel distribution
    num_users = 9 # number of wireless channels (action_dim and state_dim)
    num_rus = 9 

    theta_lr = 1e-3
    lambda_lr = .001

    sys = WirelessSchedulingSystem_CC(num_users,num_rus=num_rus)
    distribution = ClassificationDistribution2(sys.action_dim,num_rus)
    scheduling_policy = ReinforcePolicy(sys.state_dim, 
        sys.action_dim, 
        sys.constraint_dim,model_builder=mlp_model, distribution=distribution, theta_lr = theta_lr, lambda_lr = lambda_lr)

    history_dict, runtime_dict = run_sys(sys, scheduling_policy, 40000,num_reruns=1, batch_size=64)
    save_data(history_dict, "wireless_control_data.mat")
    save_rt_data(runtime_dict, "wireless_control_datab.mat")


def wireless_control_test2():
    mu = 1 # parameter for exponential distribution of wireless channel distribution
    num_users = 10 # number of wireless channels (action_dim and state_dim)
    num_rus = 10 
    tmax = .0005

    lower_bound = 1.6
    upper_bound = 8.0

    theta_lr = 5e-4
    lambda_lr = .00001

    sys = WirelessSchedulingSystem_TD(num_users, tmax=tmax)


    distribution = TruncatedGaussianBernoulliDistribution(sys.action_dim,
        lower_bound=lower_bound, 
        upper_bound=upper_bound)
    scheduling_policy = ReinforcePolicy(sys.state_dim, 
        sys.action_dim, 
        sys.constraint_dim,model_builder=mlp_model2, distribution=distribution, theta_lr = theta_lr, lambda_lr = lambda_lr)

    history_dict, runtime_dict, runtime_dict2 = run_sys(sys, scheduling_policy, 40000,batch_size=100)

    save_data(history_dict, "wireless_control_data6.mat")
    save_rt_data(runtime_dict, "wireless_control_data6b.mat")
    save_rt_data(runtime_dict2, "wireless_control_data6c.mat")

def wireless_control_test_T():
    mu = 1 # parameter for exponential distribution of wireless channel distribution
    num_users = 10 # number of wireless channels (action_dim and state_dim)
    num_rus = 10 
    tmax = .0005

    lower_bound = 1.6
    upper_bound = 8.0

    T = 10

    theta_lr = 5e-4
    lambda_lr = .00001

    sys = WirelessSchedulingSystem_TD(num_users, tmax=tmax, T=T)


    distribution = TruncatedGaussianBernoulliDistribution(sys.action_dim,
        lower_bound=lower_bound, 
        upper_bound=upper_bound)
    scheduling_policy = ReinforcePolicy(sys, model_builder=mlp_model2, distribution=distribution, theta_lr = theta_lr, lambda_lr = lambda_lr, T=T)

    history_dict, runtime_dict, runtime_dict2 = run_sys(sys, scheduling_policy, 15000,batch_size=100)

    save_data(history_dict, "wireless_control_data7.mat")
    save_rt_data(runtime_dict, "wireless_control_data7b.mat")
    save_rt_data(runtime_dict2, "wireless_control_data7c.mat")


def training_set_test():
    mu = 1 # parameter for exponential distribution of wireless channel distribution
    num_users = 10 # number of wireless channels (action_dim and state_dim)
    num_rus = 10 
    tmax = .0005

    lower_bound = 1.6
    upper_bound = 8.0

    theta_lr = 7e-5
    lambda_lr = .001

    sys = WirelessSchedulingSystem_TD(num_users, tmax=tmax)

    generate_training_set(sys,1000000,"training_set.mat")




if __name__ == '__main__':

    import argparse
    import sys

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    tf.set_random_seed(rn)
    np.random.seed(rn1)

   # wireless_control_test2()
    wireless_control_test_T()
    #training_set_test()

