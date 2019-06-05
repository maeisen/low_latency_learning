import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.io
from scipy import sparse

from control_system import *
from reinforce_policy import ReinforcePolicy
from reinforce_policy import *



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


###########################################################
##### T R A I N I N G #####################################
########################################################3##
def train_sys(sys, policy, num_iter_train, batch_size=64, explore_decay=0.999, save_filename="test.mat"):

    # initalize dictionaries for storing training and execution data
    history_dict = {'lambd': [],
                   'f0': [],
                   'f1': [],
                   'p': [],
                   'Ao': [],
                   'Ac': [],
                   'x': [],
                   'tt': [],
                   'g': []}
    history_dict['Ao'].append(sys.Ao)
    history_dict['Ac'].append(sys.Ac)


    print("::::::::TRAINING PHASE:::::::")
    eps = 1.0
    for k in range(num_iter_train):
        state0 = sys.sample(batch_size)
        channel_state = sys.sample_graph(batch_size)

        if k%1000 == 0:
            print("Iteration " + str(k))

        explore = np.random.binomial(1,eps)

        if explore:
            states, actions = policy.random_action(state0) # generate policy rollout over time horizon
        else:
            states, actions = policy.get_action(state0, channel_state) # generate policy rollout over time horizon

        f0 = sys.f0(states, actions, channel_state) # evaluate control cost over rollout
        f1 = sys.f1(states, actions, channel_state) # evaluate transmission time over rollout
        tt = sys.get_time(actions[:,:,0])


        if k%100 == 0:
            history_dict['lambd'].append(policy.lambd)  # store dual parameter

            history_dict['f0'].append(np.mean(f0)) # store f0
            history_dict['f1'].append(f1)  # store f1
            history_dict['tt'].append(tt)

        if k%1000 == 0:
            history_dict['x'].append(states) # store training state
            history_dict['p'].append(actions) # store training action



        loss, grads = policy.learn(states[:,:,0], actions[:,:,0], f0, f1, channel_state) # perform learning step

        history_dict['g'].append(grads)
        eps = np.maximum(0.001,explore_decay*eps)

    save_policy(policy,"nn_test.mat") # save NN in MAT file
  #  pdb.set_trace()
    save_data(history_dict,save_filename)

    return history_dict


###########################################################
##### E X E C U T I O N ###################################
########################################################3##
def run_sys(sys, policy, num_iter_exec, batch_size = 100, include_L = True, include_RR = True, include_PR = True, save_filename1="test2.mat", save_filename2="test2b.mat"):

    # initalize dictionaries for storing training and execution data
    runtime_dict = {'f0': [],
                   'f0_2': [],
                   'f0_3': [],
                   'p': [],
                   'p2': [],
                   'p3': [],
                   'T': [],
                   'T2': [],
                   'x': [],
                   'x2': [],
                   'x3': [],
                   'Ao':[],
                   'Ac':[]}
    runtime_dict2 = {'f0': [],
                   'f0_2': [],
                   'f0_3': [],
                   'p': [],
                   'p2': [],
                   'p3': [],
                   'T': [],
                   'T2': [],
                   'x': [],
                   'x2': [],
                   'x3': [],
                   'Ao':[],
                   'Ac':[]}
    runtime_dict['Ao'].append(sys.Ao)
    runtime_dict['Ac'].append(sys.Ac)
    runtime_dict2['Ao'].append(sys.Ao)
    runtime_dict2['Ac'].append(sys.Ac)

    state0 = sys.sample(batch_size)
    state1 = np.copy(state0)
    state2 = np.copy(state1)
    state3 = np.copy(state1)
    graph_state = sys.sample_graph(batch_size)
    graph_state1 = np.copy(graph_state)
    graph_state2 = np.copy(graph_state)
    graph_state3 = np.copy(graph_state)
    print("::::::::EXECUTION PHASE:::::::")
    for k in range(num_iter_exec):
        
        if k%100 == 0:
            print("Iteration " + str(k))
        state0 = sys.sample(batch_size)
        graph_state = sys.sample_graph(batch_size)


        # actions = policy.get_action(state1)
        # T = sys.get_time(actions)
        # actions2 = sys.round_robin(state2, T[0])
        # actions3 = sys.priority_ranking(state3, T[0])

        # f0_1 = sys.f0(state1,actions)
        # f0_2 = sys.f0_m(state2,actions2)
        # f0_3 = sys.f0_m(state3,actions3)
        # f0_4 = sys.f0_m(state4,actions4)

        # runtime_dict['f0'].append(f0_1)
        # runtime_dict['f0_2'].append(f0_2)
        # runtime_dict['f0_3'].append(f0_3)
        # runtime_dict['f0_4'].append(f0_4)
        # runtime_dict['p'].append(actions)
        # runtime_dict['p2'].append(actions2)
        # runtime_dict['p3'].append(actions3)
        # runtime_dict['p4'].append(actions4)
        # runtime_dict['T'].append(T[0])
        # runtime_dict['x'].append(state1)
        # runtime_dict['x2'].append(state2)
        # runtime_dict['x3'].append(state3)
        # runtime_dict['x4'].append(state4)

        # state1 = sys.update_system(state1,actions,2,rate_select=True)
        # state2 = sys.update_system(state2,actions2,2,rate_select=False)
        # state3 = sys.update_system(state3,actions3,2,rate_select=False)
        # state4 = sys.update_system(state4,actions4,2,rate_select=False)

        if include_L:
            states0, actions = policy.get_action2(state0, graph_state, training=False)
            T = sys.get_time(actions[:,:,0])
            actions = sys.fill_in(actions[:,:,0], T)
            f0_1 = sys.f0(states0,actions, graph_state)
            runtime_dict2['f0'].append(f0_1)
            runtime_dict2['T'].append(T)
            runtime_dict2['p'].append(actions) 

            states, actions = policy.get_action2(state1, graph_state, training=False)
            T = sys.get_time(actions[:,:,0])
            actions = sys.fill_in(actions[:,:,0], T)
            runtime_dict['T'].append(T)
            runtime_dict['x'].append(state1)  
            state1, graph_state1 = sys.update_system(states, actions, graph_state1) 
        else:
            states0 = np.expand_dims(state0, axis=2)

        if include_RR:

            actions2 = sys.round_robin(states0,graph_state)
            f0_2 = sys.f0(states0,actions2, graph_state)
            runtime_dict2['f0_2'].append(f0_2)
            runtime_dict2['p2'].append(actions2)
           
            ss = np.expand_dims(state2,axis=2)
            actions2 = sys.round_robin(ss,graph_state)
            runtime_dict['x2'].append(state2)
            state2, graph_state2 = sys.update_system(ss, actions2, graph_state2) 


        if include_PR:
            actions3 = sys.priority_ranking(states0,graph_state)
            f0_3 = sys.f0(states0,actions3, graph_state)
            runtime_dict2['f0_3'].append(f0_3)
            runtime_dict2['p3'].append(actions3)

            ss = np.expand_dims(state3,axis=2)
            actions3 = sys.priority_ranking(ss,graph_state)    
            runtime_dict['x3'].append(state3)
            state3, graph_state3 = sys.update_system(ss, actions3, graph_state3) 

        runtime_dict2['x'].append(state0)

    save_data(runtime_dict,save_filename1)
    save_data(runtime_dict2,save_filename2)

    return runtime_dict, runtime_dict2


#############################################
############ Save NN architecture ###############
##############################################
def save_policy(policy, filename):
    data_save = {}
    index = 0
        
    tvars = tf.trainable_variables()
    num_layers = int(len(tvars)/2)
    variable_names = []
    for i in np.arange(num_layers):
        variable_names.append("weight"+str(i))
        variable_names.append("bias"+str(i))
    variable_names.append("weight"+str(num_layers))
    tvars_vals = policy.sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        data_save[variable_names[index]] = val
        index +=1

    scipy.io.savemat(filename,data_save)

#############################################
############ Save training data ###############
##############################################
def save_data(data, filename):
    scipy.io.savemat(filename, data)


#############################################
############ Create training set using heuristic ###############
##############################################
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
        states = np.expand_dims(states,axis=2)
        graph_state = sys.sample_graph(1)

        if k%10000 == 0:
            print("Iteration " + str(k))

        actions3 = sys.priority_ranking(states, sys.tmax)
        f0_3 = sys.f0(states,actions3, graph_state)

        history_dict['f0'].append(f0_3)
        history_dict['p'].append(actions3)
        history_dict['x'].append(states)
    scipy.io.savemat(filename, history_dict)


####################
## TEST FUNCTIONS ##
####################


#######################################################################
#### OFDM Test w/o time horizon learning (i.e. instanteous control cost)####
#######################################################################
def wireless_control_test_f():
    mu = 1 # parameter for exponential distribution of wireless channel distribution

    # load channel info
    DD = scipy.io.loadmat("channel_info_short_python.mat")
    CSI = DD['H2'] - 10

    num_users = 12 # number of devices 
    tmax = .0005  # latenct bound

    num_channels = 2

    batch_size = 200

    # bounds for data rate policy ###
    lower_bound = 1.6
    upper_bound = 12.0

    # learning rates
    theta_lr = 1e-3
    lambda_lr = 5e-3

     # Initialize scheduling system
    sys = WirelessSchedulingSystem_OFDM_Graph(num_users, tmax=tmax, mu=mu, CSI=CSI, num_channels=num_channels, online_samples=False)

    # Set polict distribution
    distribution = GaussianBernoulliDistribution(num_users,lower_bound=lower_bound, upper_bound=upper_bound, num_classes=num_channels)
    # Init policy trainer
    scheduling_policy = ReinforcePolicy(sys,model_builder=gnn_fc_model, distribution=distribution, theta_lr = theta_lr, lambda_lr = lambda_lr)

    # Run simulation
    history_dict = train_sys(sys, scheduling_policy, num_iter_train = 150000, batch_size=batch_size, save_filename="pl11.mat")
    runtime_dict, runtime_dict2 = run_sys(sys, scheduling_policy, num_iter_exec = 1000,  batch_size=batch_size,save_filename1 = "pl22.mat",save_filename2 ="pl21.mat")


#######################################################################
#### Test using time horizon learning (i.e. reinforcement learning)###
#######################################################################
def wireless_control_test_T():
    mu = 1 # parameter for exponential distribution of wireless channel distribution

    num_users = 10 # number of devices 
    tmax = .0005  # latenct bound

    # bounds for data rate policy ###
    lower_bound = 1.6
    upper_bound = 8.0

    # load channel info
    DD = scipy.io.loadmat("channel_info_short_python.mat")
    CSI = DD['H20'] - 10

    # time horizon
    T = 10

    # learning rates
    theta_lr = 5e-3
    lambda_lr = .0001

    # Initialize scheduling system
    sys = WirelessSchedulingSystem_TD(num_users, tmax=tmax, T=T, mu=mu, CSI=CSI)

    # Set policy distribution
    distribution = TruncatedGaussianBernoulliDistribution(sys.action_dim,
        lower_bound=lower_bound, 
        upper_bound=upper_bound)
    # Initialize policy trainer
    scheduling_policy = ReinforcePolicy(sys, model_builder=mlp_model2, distribution=distribution, theta_lr = theta_lr, lambda_lr = lambda_lr, T=T)

    # Run simulation
    history_dict, runtime_dict, runtime_dict2 = run_sys(sys, scheduling_policy, num_iter_train = 10000, num_iter_exec = 0, batch_size=64)

    
    save_data(history_dict, "wireless_control_data7.mat") # save training data
    save_rt_data(runtime_dict, "wireless_control_data7b.mat") # save execution data  (without control system)
    save_rt_data(runtime_dict2, "wireless_control_data7c.mat") # save execution data (with control system)


#######################################################################
#### Test used to generate training set with heurisitc ################
#######################################################################
def training_set_test():
    mu = 1 # parameter for exponential distribution of wireless channel distribution

    # load channel info
    DD = scipy.io.loadmat("channel_info_short_python.mat")
    CSI = DD['H2'] - 10

    num_users = 12 # number of devices 
    tmax = .0005  # latenct bound

    num_channels = 3

    # bounds for data rate policy ###
    lower_bound = 1.6
    upper_bound = 9.0

    # learning rates
    theta_lr = 7e-3
    lambda_lr = .0001

     # Initialize scheduling system
    sys = WirelessSchedulingSystem_OFDM(num_users, tmax=tmax, mu=mu, CSI=CSI, num_channels=num_channels)

    generate_training_set(sys,400000,"training_set2.mat")

#######################################################################
#### Test used to train, starting from supervised model ################
#######################################################################
def supervised_test(retrain=False):

    mu = 1 # parameter for exponential distribution of wireless channel distribution

    num_users = 6 # number of devices 
    tmax = .0005  # latenct bound
    num_channels = 2


    # bounds for data rate policy ###
    lower_bound = 1.6
    upper_bound = 11.0

    data = scipy.io.loadmat("training_set2.mat")


    # create model
    if retrain:
        X = data['x']
        Y = data['p']
        X = X[:,0,:]
        Y1 = Y[:,0,:num_users]
        Y2 = Y[:,0,num_users:]

        X = X[:,:,0]
        Y1 = Y1[:,:,0]
        Y2 = Y2[:,:,0]

        layers=[32]*31
        L = len(layers)

        model1 = keras.models.Sequential()
        model1.add(keras.layers.Dense(32, input_dim=(num_users)*(num_channels+1), activation='relu', name='dense_0'))
        for idx, layer in enumerate(layers):
            model1.add(keras.layers.Dense(layer, activation='relu', name='dense_'+str(idx+1)))
        model1.add(keras.layers.Dense(num_channels*num_users, activation='sigmoid', name='dense_'+str(L+1)))
    
        model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model1.fit(X, Y2, epochs=1, batch_size=100)
        model1.save("trained_model2.h5")

        model2 = keras.models.Sequential()
        model2.add(keras.layers.Dense(256, input_dim=(num_users)*(num_channels+1), activation='relu', name='dense_0'))
        model2.add(keras.layers.Dense(128, activation='relu', name='dense_1'))
        model2.add(keras.layers.Dense(64, activation='relu', name='dense_2'))
        model2.add(keras.layers.Dense(num_users, activation='linear', name='dense_3'))

    
        model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model2.fit(X, Y1, epochs=5, batch_size=100)
        model2.save("trained_model3.h5")

        models = [model1, model2]

    else:
        model1 = keras.models.load_model("trained_model2.h5")
        model2 = keras.models.load_model("trained_model3.h5")
        models = [model1, model2]

    # load channel info
    DD = scipy.io.loadmat("channel_info_short_python.mat")
    CSI = DD['H2'] - 13


    # learning rates
    theta_lr = 5e-4
    lambda_lr = .01

     # Initialize scheduling system
    sys = WirelessSchedulingSystem_OFDM(num_users, tmax=tmax, mu=mu, CSI=CSI, num_channels=num_channels)

    # Set polict distribution
    distribution = GaussianBernoulliDistribution(num_users,lower_bound=lower_bound, upper_bound=upper_bound, num_classes=num_channels)
    # Init policy trainer
    scheduling_policy = ReinforcePolicy(sys,model_builder=mlp_model_multi2, distribution=distribution, theta_lr = theta_lr, lambda_lr = lambda_lr, model=models)

    # Run simulation
    history_dict = train_sys(sys, scheduling_policy, num_iter_train = 100000, batch_size=200, explore_decay = 0.0,save_filename="ss1.mat")
    runtime_dict, runtime_dict2 = run_sys(sys, scheduling_policy, num_iter_exec = 1000, include_L = True, save_filename2="ss2.mat",save_filename1="ss3.mat")



##############################################################################
##### M   A   I   N  #########################################################
##############################################################################
if __name__ == '__main__':

    import argparse
    import sys

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    tf.set_random_seed(rn)
    np.random.seed(rn1)


   # Select which test to run

    wireless_control_test_f()
   # wireless_control_test_T()
   # training_set_test()

  #  supervised_test(retrain=False)
#
