import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
from scipy import sparse

from control_system import *
from reinforce_policy import ReinforcePolicy
from reinforce_policy import *
import networkx as nwx
import graphtools as gt


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

def run_sys(sys, policies, num_iter, num_reruns=1, batch_size=64):
    history_list = []
    runtime_list = []
    num_iters_run = 500
    for i in range(num_reruns):
        print("Simulation " + str(i))

        history_dict = {}
        runtime_dict = {}

        for policy_name, _ in policies.items():
            policy_dict = {'lambd': [],
                           'f0': [],
                           'f1': [],
                           'L': [],
                           'p': [],
                           'Ao': [],
                           'Ac': [],
                           'x': []}
            policy_dict2 = {'f0': [],
                           'f1': [],
                           'f0_m': [],
                           'f1_m': [],
                           'x': [],
                           'Ao':[],
                           'Ac':[]}
            history_dict[policy_name] = policy_dict
            history_dict[policy_name]['Ao'].append(sys.Ao)
            history_dict[policy_name]['Ac'].append(sys.Ac)
            runtime_dict[policy_name] = policy_dict2
            runtime_dict[policy_name]['Ao'].append(sys.Ao)
            runtime_dict[policy_name]['Ac'].append(sys.Ac)

        for k in range(num_iter):
            states = sys.sample(batch_size)

            if k%1000 == 0:
                print("Iteration " + str(k))

            for policy_name, policy in policies.items():
                actions = policy.get_action(states)
                f0 = sys.f0(states, actions)
                f1 = sys.f1(states, actions)
                L = f0 + np.dot(f1, policy.lambd)


                history_dict[policy_name]['lambd'].append(policy.lambd)

                history_dict[policy_name]['f0'].append(np.mean(f0))
                history_dict[policy_name]['f1'].append(np.mean(f1,axis=0))
                history_dict[policy_name]['L'].append(np.mean(L))

                if k%1000 == 0:
                    #print(actions)
                    history_dict[policy_name]['x'].append(states)
                    history_dict[policy_name]['p'].append(actions)

                policy.learn(states, actions, f0, f1)
        
        for k in range(num_iters_run):
            states = sys.sample(2)
            print("Run Iteration " + str(k))
            
            for policy_name, policy in policies.items():
                actions = policy.get_action2(states)
                actions2 = sys.round_robin(states)

                runtime_dict[policy_name]['f1'].append(actions)
                runtime_dict[policy_name]['f1_m'].append(actions2)
                runtime_dict[policy_name]['x'].append(states)


                f0 = sys.f0(states, actions)
                f0_m = sys.f0_m(states,actions2)
                f1 = sys.f1(states, actions)


                runtime_dict[policy_name]['f0'].append(f0[0,:])
                runtime_dict[policy_name]['f0_m'].append(f0_m[0,:])


        history_list.append(history_dict)
        runtime_list.append(runtime_dict)

    return history_list, runtime_list

def save_data(data, filename):
    data_dict = {}
    # plotting variables over time
    for data_name in ['lambd', 'f0', 'f1', 'L', 'p', 'Ac', 'Ao', 'x']:
        data_list = []
        for policy_name, _ in data.items():
            data_list.append(data[policy_name][data_name])
        data_list = np.array(data_list)
        data_dict[data_name] = data_list
    scipy.io.savemat(filename, data_dict)

def save_rt_data(data, filename):
    data_dict = {}
    # plotting variables over time
    for data_name in ['f0', 'f1', 'f0_m', 'f1_m', 'x','Ao','Ac']:
        data_list = []
        for policy_name, _ in data.items():
            data_list.append(data[policy_name][data_name])
        data_list = np.array(data_list)
        data_dict[data_name] = data_list
    scipy.io.savemat(filename, data_dict)



####################
## TEST FUNCTIONS ##
####################
def wireless_control_test():
    mu = 2 # parameter for exponential distribution of wireless channel distribution
    num_users = 9 # number of wireless channels (action_dim and state_dim)
    num_rus = 9 

    theta_lr = 5e-4
    lambda_lr = .05

    sys = WirelessSchedulingSystem_CC(num_users,num_rus=num_rus)
    distribution = ClassificationDistribution2(sys.action_dim,num_rus)
    scheduling_policy = ReinforcePolicy(sys.state_dim, 
        sys.action_dim, 
        sys.constraint_dim,model_builder=mlp_model, distribution=distribution, theta_lr = theta_lr, lambda_lr = lambda_lr)
    policies = {'reinforce': scheduling_policy}


    history_dict, runtime_dict = run_sys(sys, policies, 30000,num_reruns=1, batch_size=64)
    history_dict = history_dict[0]
    runtime_dict = runtime_dict[0]
    save_data(history_dict, "wireless_control_data.mat")
    save_rt_data(runtime_dict, "wireless_control_datab.mat")
    #plot_data(history_dict, "wireless_control_")

####################
## TEST FUNCTIONS ##
####################
def wireless_control_test2():
    mu = 2 # parameter for exponential distribution of wireless channel distribution
    num_users = 20 # number of wireless channels (action_dim and state_dim)
    num_rus = 20 

    lower_bound = 1.6
    upper_bound = 5.0

    theta_lr = 5e-4
    lambda_lr = .005

    sys = WirelessSchedulingSystem_TD(num_users,num_rus=num_rus)
    distribution = TruncatedGaussianBernoulliDistribution(sys.action_dim,
        lower_bound=lower_bound, 
        upper_bound=upper_bound)
    scheduling_policy = ReinforcePolicy(sys.state_dim, 
        sys.action_dim, 
        sys.constraint_dim,model_builder=mlp_model, distribution=distribution, theta_lr = theta_lr, lambda_lr = lambda_lr)
    policies = {'reinforce': scheduling_policy}


    history_dict, runtime_dict = run_sys(sys, policies, 20000,num_reruns=1, batch_size=64)
    history_dict = history_dict[0]
    runtime_dict = runtime_dict[0]
    save_data(history_dict, "wireless_control_data1.mat")
    save_rt_data(runtime_dict, "wireless_control_data1b.mat")
    #plot_data(history_dict, "wireless_control_")




if __name__ == '__main__':
    # data = np.array([0.16645503916151938, 0.601523896825384, 0.8141793467540994, 1.4449876414838947, 1.0427023417915917, 1.9309799810970443, 2.415899078581715, 2.439764538130838, 4.875788384249898])
    # plt.plot([5, 10, 15, 20, 25, 30, 35, 40, 45], data)
    # plt.xlabel('number of channels'64
    # plt.ylabel('objective gap')
    # plt.show()



    import argparse
    import sys

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    tf.set_random_seed(rn)
    np.random.seed(rn1)

    wireless_control_test2()

    # parser = argparse.ArgumentParser(description="Constrained Policy Optimization")
    # parser.add_argument('--type',
    #     dest='type',
    #     action='store',
    #     type=str,
    #     required=True,
    #     choices=['wireless_cap', 'wireless_cap2', 'wireless_cap_interference', 'wireless_control'])
    # args = parser.parse_args(sys.argv[1:])


    # if args.type == 'wireless_cap':
    #     wireless_capacity_test()
    # elif args.type == 'wireless_cap2':
    #     wireless_capacity_large_test()
    # elif args.type == 'wireless_cap3':
    #     wireless_capacity_test3()
    # elif args.type == 'wireless_cap_interference':
    #     wireless_capacity_interference_test()
    # elif args.type == 'wireless_control':
    #     wireless_control_test()



