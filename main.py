import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
from scipy import sparse

from control_system import *
from scheduling_policy import SchedulingPolicy
from scheduling_policy import *
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
    for i in range(num_reruns):
        print("Simulation " + str(i))

        history_dict = {}
        for policy_name, _ in policies.items():
            policy_dict = {'lambd': [],
                           'f0': [],
                           'f1': [],
                           'L': []}
            history_dict[policy_name] = policy_dict

        for k in range(num_iter):
            states = sys.sample(batch_size)
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

                policy.learn(states, actions, f0, f1)

        history_list.append(history_dict)
    return history_list

def plot_data(data, save_prefix='images/temp_'):
    # plotting variables over time

    for data_name in ['lambd', 'f0', 'f1', 'L']:
        plt.cla()

        data_list = []
        for policy_name, _ in data.items():
            plt.plot(data[policy_name][data_name], label=policy_name)

        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel(data_name)
        plt.title(data_name)
        save_file_name = save_prefix + data_name + '.png'
        plt.savefig(save_file_name, bbox_inches='tight')


def save_data(data, filename):
    data_dict = {}
    # plotting variables over time
    for data_name in ['lambd', 'f0', 'f1', 'L']:
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

    cap_min = .2
    sys = WirelessSchedulingSystem(num_users)
    scheduling_policy = SchedulingPolicy(sys.state_dim, 
        sys.action_dim, 
        sys.constraint_dim)
    policies = {'reinforce': scheduling_policy}

    history_dict = run_sys(sys, policies, 5000)[0]

    save_data(history_dict, "wireless_control_data.mat")
    #plot_data(history_dict, "wireless_control_")

    


if __name__ == '__main__':
    # data = np.array([0.16645503916151938, 0.601523896825384, 0.8141793467540994, 1.4449876414838947, 1.0427023417915917, 1.9309799810970443, 2.415899078581715, 2.439764538130838, 4.875788384249898])
    # plt.plot([5, 10, 15, 20, 25, 30, 35, 40, 45], data)
    # plt.xlabel('number of channels')
    # plt.ylabel('objective gap')
    # plt.show()



    import argparse
    import sys

    tf.set_random_seed(0)
    np.random.seed(1)

    wireless_control_test()

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



