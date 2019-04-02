import numpy as np
import pdb
import scipy.io
from functools import partial

class ProbabilisticSystem(object):
    def __init__(self, state_dim, action_dim, constraint_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim

        self.channel_state_dim = state_dim
        self.control_state_dim = 0

    def sample(self, batch_size):
        raise Exception("Not yet implemented")

    def f0(self, state, action):
        raise Exception("Not yet implemented")

    def f1(self, state, action):
        raise Exception("Not yet implemented")

    def psolve(self, state, lambd):
        raise Exception("Not yet implemented")

    def _exponential_sample(self, batch_size):
        samples = np.random.exponential(self.mu, size=(batch_size, self.state_dim))
        return samples

    def _normal_sample(self, batch_size):
        samples = np.random.normal(0, 1, size=(batch_size, self.state_dim))
        return samples

    def _exponential_normal_sample(self, batch_size):
        samples = np.random.exponential(self.mu, size=(batch_size, self.channel_state_dim))
        samples2 = np.random.normal(0, 1, size=(batch_size, self.control_state_dim))
        return np.hstack((samples,samples2))

    def _exponential_uniform_sample(self, batch_size, bound):
        samples = np.random.exponential(self.mu, size=(batch_size, self.channel_state_dim))
        samples2 = np.random.uniform(-bound, bound, size=(batch_size, self.control_state_dim))
        return np.hstack((samples,samples2))

    def _exponential_constant_sample(self, batch_size, state):
        samples = np.random.exponential(self.mu, size=(batch_size, self.channel_state_dim))
        samples2 = np.tile(state,[batch_size, 1])
        return np.hstack((samples,samples2))


#######################################################################################3
########################################################################################
################## W I F I (CC) ##############################################################
##########################################################################################
###########################################################################################
class WirelessSchedulingSystem_CC(ProbabilisticSystem):
    def __init__(self, num_users, p=1, Ao=None, Ac=None, W=None, rho = .95, mu=2, S=1, num_rus = 25,num_channels=9):
        super().__init__(num_users*num_rus + num_users*p, num_users, S*num_channels)

        self.channel_state_dim = num_users*num_rus
        self.control_state_dim = num_users*p
        self.ru_dim = num_users*num_channels
        self.mcs_dim = num_users

        self.packet_size = 100

        self.p = p
        self.rho = rho

        self.pmax = 1
        self.mu = mu
        self.num_users = num_users
        self.num_rus = num_rus
        self.S = S
        self.num_channels = num_channels


        self.rate_by_mcs = [1.6,2.4,3.3,4.9,6.5,7.3,8.1,9.8,10.8,12.2,13.5]
        self.rate_by_mcs = np.asarray(self.rate_by_mcs)
        self.bw_by_ru = [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,4,4,4,4,4,4,10]
        self.bw_by_ru = np.asarray(self.bw_by_ru)

        self.bw_by_ru = [1,1,1,1,1,1,1,1,1]
        self.bw_by_ru = np.asarray(self.bw_by_ru)

        T = scipy.io.loadmat('snr_info.mat')
        self.per_by_snr = T.get('mcs_snr_per_wcl_20B')

        if Ac == None:
            self.Ac = (0.95-0.35)*np.random.random_sample(num_users)+0.5
        else:
            if (Ac.shape != (1, num_users)):
                raise Exception("Ac is not the correct shape")
            self.Ac = Ac

        if Ao == None:
            self.Ao = (1.2-1.01)*np.random.random_sample(num_users)+1.01
        else:
            if (Ao.shape != (1, num_users)):
                raise Exception("Ao is not the correct shape")
            self.Ao = Ao

        if W == None:
            self.W = np.ones(num_users)
        else:
            if (W.shape != (1, num_users)):
                raise Exception("W is not the correct shape")
            self.W = W

    def sample(self, batch_size):
        return self._exponential_normal_sample(batch_size)

    def db_to_col(self,snr_value):
        snr_value = np.minimum(snr_value,35)
        snr_value = np.maximum(snr_value,-35)

        col = np.floor((snr_value+35)/0.1+1)-1
        return col.astype(int)

    def packet_delivery_rate(self,snr_value,mcs,per_by_snr):
        col = self.snr_to_col(snr_value)
        pdr = 1 - per_by_snr[mcs-1,col]
        return pdr

    def snr_to_db(self,tx_power,noise_snr):
        db = 10*np.log10(tx_power/noise_snr)
        return db

    def mcs_to_time(self,mcs,packet_size):
        time = 8*packet_size*0.000001/(self.rate_by_mcs[mcs]*np.tile(self.bw_by_ru,(mcs.shape[0],1)))
        return time


    def f0(self, state, action):
        """
        objective function (total transmission time)
        Args:
            state (TYPE): channel state, N by state_dim matrix
            action (TYPE): power allocation, N by action_dim matrix

        Returns:
            TYPE: N by 1 matrix of negative total channel capacity
        """
        N = action.shape[0]

        action_a = np.asarray(action)
        active_users = np.where(action_a>=0)
        action = action.astype(int)

        time_per_user_per_ru = self.get_time(state)
        all_times = time_per_user_per_ru[active_users[0],active_users[1],action[active_users]]  

        all_times_array = np.reshape(all_times,(N,self.num_users))
        
        total_time = np.sum(all_times_array,axis=1)
        return np.reshape(total_time,(-1,1))
        #return np.ones((N,1))

    def get_binary(self, action):

        a2 = action.tolist()
        f = np.zeros(shape=(len(a2),self.action_dim,9))
        index = 1

        # binary matrix of RUs (25 total)
        choiselst = np.array([
            #[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
            #[1, 1, 0, 0, 0, 0, 0, 0, 0],
            #[0, 1, 1, 0, 0, 0, 0, 0, 0],
            #[0, 0, 1, 1, 0, 0, 0, 0, 0],
            #[0, 0, 0, 1, 1, 0, 0, 0, 0],
            #[0, 0, 0, 0, 1, 1, 0, 0, 0],
            #[0, 0, 0, 0, 0, 1, 1, 0, 0],
            #[0, 0, 0, 0, 0, 0, 1, 1, 0],
            #[0, 0, 0, 0, 0, 0, 0, 1, 1],
            #[1, 1, 1, 1, 0, 0, 0, 0, 0],
            #[0, 1, 1, 1, 1, 0, 0, 0, 0],
            #[0, 0, 1, 1, 1, 1, 0, 0, 0],
            #[0, 0, 0, 1, 1, 1, 1, 0, 0],
            #[0, 0, 0, 0, 1, 1, 1, 1, 0],
            #[0, 0, 0, 0, 0, 1, 1, 1, 1],
            #[1, 1, 1, 1, 1, 1, 1, 1, 1]
            ])


        for i in np.arange(0,len(a2)):
            x = np.asarray(a2[i]).astype(int)
            #condlist = [np.equal(x,0),np.equal(x,1),np.equal(x,2),np.equal(x,3),np.equal(x,4),np.equal(x,5),np.equal(x,6),np.equal(x,7),np.equal(x,8),np.equal(x,9),np.equal(x,10),np.equal(x,11),np.equal(x,12),np.equal(x,13),np.equal(x,14),np.equal(x,15),np.equal(x,16),np.equal(x,17),np.equal(x,18),np.equal(x,19),np.equal(x,20),np.equal(x,21),np.equal(x,22),np.equal(x,23),np.equal(x,24)]
            f[i,:,:] = choiselst[x]
        return f

    def get_time(self,state):
        """
        Compute transmission time for each user in each RU
        Inputs: State (N x state_dim)
        Outputs: Transmission time per RU (N x num_users x num_RUs)
        """
        channel_state = state[:,0:self.channel_state_dim]
        control_state = state[:,self.channel_state_dim:]

        N = np.size(state,0)

        channel_state_col = self.db_to_col(self.snr_to_db(2,channel_state))

        channel_state_mat = np.reshape(channel_state_col,(N,self.num_users,self.num_rus))
        control_state_mat = np.reshape(control_state,(N,self.num_users,self.p))

        #channel_state_col = self.db_to_col(self.snr_to_db(2,channel_state_mat))

        time_per_user_per_ru = np.zeros((N,self.num_users,self.num_rus))

        pp = 1 - self.per_by_snr[:,channel_state_mat]
        for i in np.arange(self.num_users):
            #p_min = control_state_mat[:,i,:].T * (self.Ao.T * self.Ao - self.rho) * control_state_mat[:,i,:]
            #p_min = p_min / (control_state_mat[:,i,:].T * (self.Ao.T * self.Ao - self.Ac.T * self.Ac) * control_state_mat[:,i,:])
            #p_min = (np.power(self.Ao[i],2)-self.rho) / (np.power(self.Ao[i],2) - np.power(self.Ac[i],2))
            p_min = 0.9
            mcs_by_ru = np.maximum(np.sum(pp[:,:,i,:]>=p_min,axis=0),1)-1
            time_per_user_per_ru[:,i,0:] = self.mcs_to_time(mcs_by_ru[:,0:],self.packet_size)

        return time_per_user_per_ru 

        
        
    def f1(self, state, action):
        """
        constraint functions (no intersecting channels)
        Args:
            state (TYPE): channel state, N by state_dim matrix
            action (TYPE): power allocation, N by action_dim matrix
        
        Returns:
            TYPE: N by constrain_dim matrix of power budget violations
        """

        ru_mat = self.get_binary(action)

        #ru_mat = np.reshape(rus,(N,self.num_users,self.num_channels))
        N = np.size(state,0)
        lhs = np.zeros((N,self.constraint_dim))
        lhs[:,0:self.num_channels] = np.array([np.sum(ru_mat, axis=1) - np.ones((N,self.num_channels))])

        return lhs


#######################################################################################3
########################################################################################
################## W I F I (LC) ##############################################################
##########################################################################################
###########################################################################################
class WirelessSchedulingSystem_LC(ProbabilisticSystem):
    def __init__(self, num_users, p=1, Ao=None, Ac=None, W=None, tmax=0.001, mu=2, S=1, num_rus = 9, T= 1, CSI=np.ones(5)):
        super().__init__(num_users*num_rus + num_users*p, 3*num_users, T*(S+1))

        self.channel_state_dim = num_users*num_rus
        self.control_state_dim = num_users*p

        self.packet_size = 100
        self.tx_power = 5

        self.bound = 10

        self.p = p

        self.pmax = 1
        self.mu = mu
        self.num_users = num_users
        self.num_rus = num_rus
        self.S = S

        if (CSI==1).all():
            self.sample_source = "Random"
            self.CSI = None
        else:
            self.sample_source = "Loaded"
            self.CSI = CSI


        self.T = T

        self.rate_by_mcs = [1.6,2.4,3.3,4.9,6.5,7.3,8.1,9.8,10.8,12.2,13.5]
        self.rate_by_mcs = np.asarray(self.rate_by_mcs)
    #    self.bw_by_ru = [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,4,4,4,4,4,4,10]
    #    self.bw_by_ru = np.asarray(self.bw_by_ru)

        self.bw = 1

        T = scipy.io.loadmat('snr_info.mat')
        self.per_by_snr = T.get('mcs_snr_per_wcl_20B')

        self.users = np.random.choice(48,num_users,replace=False)
        self.states = np.random.normal(loc=0.0, scale = 1.0, size=(num_users))


        if Ac == None:
            #self.Ac = (.9-.6)*np.random.random_sample(num_users)+0.6
            self.Ac = 0.8 * np.ones(num_users)
        else:
            if (Ac.shape != (1, num_users)):
                raise Exception("Ac is not the correct shape")
            self.Ac = Ac

        if Ao == None:
           # self.Ao = (1.03-1.001)*np.random.random_sample(num_users)+1.001
            self.Ao = 1.01 * np.ones(num_users)
        else:
            if (Ao.shape != (1, num_users)):
                raise Exception("Ao is not the correct shape")
            self.Ao = Ao

        if W == None:
            self.W = np.ones(num_users)
        else:
            if (W.shape != (1, num_users)):
                raise Exception("W is not the correct shape")
            self.W = W

    def sample(self, batch_size):
        #return self._exponential_normal_sample(batch_size)
        if self.sample_source == "Random":
            return self._exponential_uniform_sample(batch_size, self.bound)
        else:
            cycle = np.random.randint(1000, size=batch_size)
            temp = self.CSI[self.users,:,:]
            temp = temp[:,cycle,:]
            channel = np.reshape(np.transpose(temp,[1,0,2]),(batch_size,self.channel_state_dim))
            control = np.random.uniform(-self.bound, self.bound, size=(batch_size, self.control_state_dim))
           # control = np.tile(self.states, [batch_size, 1])
            return np.hstack((channel,control))

    def db_to_col(self,snr_value):
        snr_value = np.minimum(snr_value,35)
        snr_value = np.maximum(snr_value,-35)

        col = np.floor((snr_value+35)/0.1+1)-1
        return col.astype(int)

    def packet_delivery_rate(self,snr_value,mcs,per_by_snr):
        
        col = self.db_to_col(snr_value)
        pdr = 1 - per_by_snr[mcs-1,col]
        return pdr

    def snr_to_db(self,tx_power,noise_snr):
        db = 10*np.log10(tx_power*noise_snr)
        return db

    def mcs_to_time(self,mcs,packet_size):
        time = 8*packet_size*0.000001/(self.rate_by_mcs[mcs-1]*self.bw)
        return time


    def rate_to_mcs(self,rate):
        rate2 = np.asarray(rate)
        mcs = np.digitize(rate2,self.rate_by_mcs) 
        mcs = np.minimum(mcs,10)
        return mcs

    def f0(self, state, action, pause=False):
        """
        objective function (total transmission time)
        Args:
            state (TYPE): channel state, N by state_dim matrix
            action (TYPE): power allocation, N by action_dim matrix

        Returns:
            TYPE: N by 1 matrix of negative total channel capacity
        """
        N = action.shape[0]
        total_cost = 0
        T = state.shape[2]
  
        if T < 2:

            rate_action = action[:,0:self.num_users,0]
            ru_action = action[:,self.num_users:2*self.num_users,0]
            transmit_action = action[:,2*self.num_users,3*self.num_users:,0]

            mcs_action = self.rate_to_mcs(rate_action)


            channel_state = state[:,0:self.channel_state_dim,0]
            channel_state_m = np.reshape(channel_state,(N,self.num_users,self.num_channels))

            control_state = state[:,self.channel_state_dim:,0]
            control_state_m = np.reshape(control_state,(N,self.num_users))
        
            cost_c = np.square(control_state_m * np.tile(self.Ac,(N,1)))
            cost_o = np.square(control_state_m * np.tile(self.Ao,(N,1))) 

            if self.sample_source == "Random":
                snr_state = self.snr_to_db(self.tx_power,channel_state)
            else:
                snr_state = channel_state

            snr_state = snr_state[:,:,ru_action]
            snr_state = np.diagonal(snr_state,axis1=1,axis2=3)
            snr_state = np.diagonal(snr_state,axis1=0,axis2=1)
            snr_state = snr_state.T

            if pause:
                pdb.set_trace()

            delivery_rates = transmit_action * self.packet_delivery_rate(snr_state,mcs_action,self.per_by_snr)

            total_cost = delivery_rates * cost_c + (1-delivery_rates) * cost_o
            

        else:
            T = state.shape[2]

            control_state = state[:,self.channel_state_dim:,:]
            control_state_m = np.reshape(control_state,(N,self.num_users,T))
        
            cost_c = np.square(control_state_m) 
            total_cost = np.sum(cost_c,axis=2)

        total_cost = np.sum(total_cost,axis=1)
        return np.reshape(total_cost,(-1,1))

    def get_binary(self, action):

        a2 = action.tolist()
        N = action.shape[0]
        f = np.zeros(shape=(N,self.num_users,self.num_rus))
        index = 1

        # # binary matrix of RUs (25 total)
        # choiselst = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 1, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 1, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 1, 1, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 1, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1, 1, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 1, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 1, 1],
        #     [1, 1, 1, 1, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 0, 0, 0, 0],
        #     [0, 0, 1, 1, 1, 1, 0, 0, 0],
        #     [0, 0, 0, 1, 1, 1, 1, 0, 0],
        #     [0, 0, 0, 0, 1, 1, 1, 1, 0],
        #     [0, 0, 0, 0, 0, 1, 1, 1, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1]])

        choiselst = np.identity(self.num_rus).astype(int)


        for i in np.arange(0,len(a2)):
            x = np.asarray(a2[i]).astype(int)
            f[i,:,:] = choiselst[x]
        return f

    def get_time(self,rate_action,ru_action,transmit_action):
        """
        Compute transmission time for each user in each RU
        Inputs: State (N x state_dim)
        Outputs: Transmission time per RU (N x num_users x num_RUs)
        """

        mcs_action = self.rate_to_mcs(rate_action)
        total_time = transmit_action * self.mcs_to_time(mcs_action,self.packet_size)
        total_time = np.amax(total_time,axis=1)

        return total_time

        
        
    def f1(self, state, action):
        """
        constraint functions (no intersecting channels)
        Args:
            state (TYPE): channel state, N by state_dim matrix
            action (TYPE): power allocation, N by action_dim matrix
        
        Returns:
            TYPE: N by constrain_dim matrix of power budget violations
        """

        rate_action = action[:,0:self.num_users,0]
        ru_action = action[:,self.num_users:2*self.num_users,0]
        transmit_action = action[:,2*self.num_users,3*self.num_users:,0]

        transmit_tile = np.repeat(transmit_action[:, :, np.newaxis], self.num_rus, axis=2)


        total_time = self.get_time(rate_action,ru_action,transmit_action)

        ru_mat = self.get_binary(ru_action)
        ru_mat = np.sum(transmit_tile * ru_mat,axis=1)

        max_ru_max = np.amax(ru_mat,axis=1)

        lhs[:,0] = 1000*(np.asarray(max_ru_mat >= 1).astype(int) - 0.01)
        lhs[:,1] = 1000*(np.asarray(total_time >= self.tmax).astype(int) - 0.05)

        return lhs


#######################################################################################3
########################################################################################
################## O F D M A ##############################################################
##########################################################################################
###########################################################################################
class WirelessSchedulingSystem_OFDMA(ProbabilisticSystem):
    def __init__(self, num_users, num_channels = 9, p=1, Ao=None, Ac=None, W=None, mu=2, CSI=1, tmax = .0005, bound = 10, T=1):
        super().__init__(num_users*num_channels + num_users*p, num_users + num_users*num_channels, 2*num_channels*T)

        self.channel_state_dim = num_users*num_channels
        self.control_state_dim = num_users*p

        self.transmit_dim = num_users*num_channels
        self.rate_dim = num_users

        self.num_channels = num_channels

        self.tmax = tmax

        self.packet_size = 100

        self.to_transmit = 0

        self.bound = bound

        self.tx_power = 5

        self.p = p

        self.pmax = 1
        self.mu = mu
        self.num_users = num_users

        self.channels = np.arange(self.num_channels)

        if (CSI==1).all():
            self.sample_source = "Random"
            self.CSI = None
        else:
            self.sample_source = "Loaded"
            self.CSI = CSI

        self.users = np.random.choice(48,num_users,replace=False)
        self.states = np.random.normal(loc=0.0, scale = 1.0, size=(num_users))


        self.rate_by_mcs = [1.6,2.4,3.3,4.9,6.5,7.3,8.1,9.8,10.8,12.2,13.5]
        self.rate_by_mcs = np.asarray(self.rate_by_mcs)
        self.bw = 2

        T = scipy.io.loadmat('snr_info.mat')
        self.per_by_snr = T.get('mcs_snr_per_wcl_20B')

        if Ac == None:
            #self.Ac = (.9-.6)*np.random.random_sample(num_users)+0.6
            self.Ac = 0.8 * np.ones(num_users)
        else:
            if (Ac.shape != (1, num_users)):
                raise Exception("Ac is not the correct shape")
            self.Ac = Ac

        if Ao == None:
           # self.Ao = (1.03-1.001)*np.random.random_sample(num_users)+1.001
            self.Ao = 1.01 * np.ones(num_users)
        else:
            if (Ao.shape != (1, num_users)):
                raise Exception("Ao is not the correct shape")
            self.Ao = Ao

        if W == None:
            self.W = 0.2*np.ones(num_users)
        else:
            if (W.shape != (1, num_users)):
                raise Exception("W is not the correct shape")
            self.W = W

    def sample(self, batch_size):
        #return self._exponential_normal_sample(batch_size)
        if self.sample_source == "Random":
            return self._exponential_uniform_sample(batch_size, self.bound)
        else:
            cycle = np.random.randint(1000, size=batch_size)
            temp = self.CSI[self.users,:,:]
            temp = temp[:,cycle,:]
            temp = temp[:,:,self.channels]
            channel = np.reshape(np.transpose(temp,[1,0,2]),(batch_size,self.channel_state_dim))
            control = np.random.uniform(-self.bound, self.bound, size=(batch_size, self.control_state_dim))
           # control = np.tile(self.states, [batch_size, 1])
            return np.hstack((channel,control))

    def db_to_col(self,snr_value):
        snr_value = np.minimum(snr_value,35)
        snr_value = np.maximum(snr_value,-35)

        col = np.floor((snr_value+35)/0.1+1)-1
        return col.astype(int)

    def packet_delivery_rate(self,snr_value,mcs,transmit,per_by_snr):
        
        col = self.db_to_col(snr_value)
        M = np.repeat(mcs[:, :, np.newaxis], self.num_channels, axis=2)

        pdr = transmit * (1 - per_by_snr[M-1,col])
        return pdr

    def snr_to_db(self,tx_power,noise_snr):
        db = 10*np.log10(tx_power*noise_snr)
        return db

    def mcs_to_time(self,mcs,packet_size):
        time = 8*packet_size*0.000001/(self.rate_by_mcs[mcs-1]*self.bw)
        return time


    def rate_to_mcs(self,rate):
        rate2 = np.asarray(rate)
        mcs = np.digitize(rate2,self.rate_by_mcs) 
        mcs = np.minimum(mcs,10)
        return mcs


    def f0(self, state, action, pause=False):
        N = action.shape[0]
        total_cost = 0
        T = 1
  
        if T < 2:

            rate_action = action[:,0:self.rate_dim,0]
            transmit_action = np.reshape(action[:,self.rate_dim:,0], (N,self.num_users,self.num_channels))

            mcs_action = self.rate_to_mcs(rate_action)


            channel_state = np.reshape(state[:,0:self.channel_state_dim,0],(N,self.num_users,self.num_channels))
            control_state = np.reshape(state[:,self.channel_state_dim:,0],(N,self.num_users))

            cost_c = np.square(control_state * np.tile(self.Ac,(N,1)))
            cost_o = np.square(control_state * np.tile(self.Ao,(N,1))) 

            if self.sample_source == "Random":
                snr_state = self.snr_to_db(self.tx_power,channel_state)
            else:
                snr_state = channel_state

            if pause:
                pdb.set_trace()

            delivery_rates = 1 - np.prod(1-self.packet_delivery_rate(snr_state,mcs_action,transmit_action,self.per_by_snr),axis=2)

            total_cost = delivery_rates * cost_c + (1-delivery_rates) * cost_o
            

        else:
            T = state.shape[2]

            control_state = state[:,self.channel_state_dim:,:]
            control_state_m = np.reshape(control_state,(N,self.num_users,T))
        
            cost_c = np.square(control_state_m) 
            total_cost = np.sum(cost_c,axis=2)

        total_cost = np.sum(total_cost,axis=1)
        return np.reshape(total_cost,(-1,1))

    def get_time(self,action):
        N = action.shape[0]
        rate_action = action[:,0:self.rate_dim]
        mcs_action = self.rate_to_mcs(rate_action)
        transmit_action = np.reshape(action[:,self.rate_dim:], (N,self.num_users,self.num_channels))

        total_time = np.zeros((N,self.num_channels))
        for j in np.arange(self.num_channels):
            
            temp_time = transmit_action[:,:,j] * self.mcs_to_time(mcs_action,self.packet_size)
            total_time[:,j] = np.sum(temp_time,axis=1)

        
        return total_time

    def f1(self, state, action):
        
        N = state.shape[0]
        lhs = np.zeros((N,self.constraint_dim))
        total_time = self.get_time(action[:,:,0])
        lhs[:,:self.num_channels] = np.maximum(0,10*(np.asarray(total_time > self.tmax).astype(int) - 0.05))
        lhs[:,self.num_channels:] = np.maximum(0,10*(np.asarray(total_time < 0.5*self.tmax).astype(int) - 0.05))

        #lhs[:,0:self.num_channels] = np.array([np.sum(ru_mat, axis=1) - 0*np.ones((N,self.num_channels))])

        return lhs

    def round_robin(self,state, tmax=None):

        if tmax==None:
            tmax = self.tmax

        p_min = 0.95

        channel_state = np.reshape(state[0,0:self.channel_state_dim,0],(1,self.num_users,self.num_channels))

        transmit_action = np.zeros((1,self.num_users,self.num_channels))
        mcs_action = np.ones((1,self.num_users))
        rate_action = 1.6*np.ones((1,self.num_users))

        if self.sample_source == "Random":
            snr_state = self.snr_to_db(self.tx_power,channel_state)
        else:
            snr_state = np.amin(channel_state,axis=2)

        col_state = self.db_to_col(snr_state)
        p2 = 1 - self.per_by_snr[:,col_state[0,:]]

        time_needed = np.zeros(self.num_users)

        for i in np.arange(self.num_users):
            max_mcs = np.maximum(np.sum(p2[:,i]>=p_min),1)
            mcs_action[0,i] = max_mcs
            rate_action[0,i] = self.rate_by_mcs[max_mcs-1]
            time_needed[i] = self.mcs_to_time(max_mcs,self.packet_size)
  
        for j in np.arange(self.num_channels):    
            total_time = 0
            total_time += time_needed[self.to_transmit]

            while (total_time <= tmax) & (np.sum(transmit_action[0,:,j]) < self.num_users):
                transmit_action[0,self.to_transmit,j] = 1.0
                self.to_transmit = (self.to_transmit + 1)%self.num_users
                total_time += time_needed[self.to_transmit]

        transmit_action = np.reshape(transmit_action,(1,self.num_users*self.num_channels))
        action = np.concatenate([rate_action, transmit_action], axis=1)
        return np.expand_dims(action,axis=2)


    def fill_in(self,action, time_used):


        rate_action = action[:,0:self.rate_dim]
        mcs_action = self.rate_to_mcs(rate_action)
        transmit_action = np.reshape(action[:,self.rate_dim:], (-1,self.num_users,self.num_channels))

        for j in np.arange(self.num_channels):
            whos_left = np.where(transmit_action[0,:,j]==0)[0]
            if len(whos_left > 0):
                np.random.shuffle(whos_left)
                for i in whos_left:
                    if time_used[j] <= self.tmax:
                        time_needed = self.mcs_to_time(mcs_action[0,i],self.packet_size)
                        if (time_used[j] + time_needed) <= self.tmax:
                            transmit_action[0,i,j] = 1.0
                            time_used[j] += time_needed

        transmit_action = np.reshape(transmit_action,(-1,self.num_users*self.num_channels))
        action = np.concatenate([rate_action, transmit_action], axis=1)
        return np.expand_dims(action,axis=2)



#######################################################################################3
########################################################################################
################## T D M A ##############################################################
##########################################################################################
###########################################################################################
class WirelessSchedulingSystem_TD(ProbabilisticSystem):
    def __init__(self, num_users, p=1, Ao=None, Ac=None, W=None, tmax=0.001, mu=2, CSI=1, t_max = .0005, bound = 10, T=1):
        super().__init__(num_users + num_users*p, 2*num_users, T)

        self.channel_state_dim = num_users
        self.control_state_dim = num_users*p

        self.transmit_dim = num_users
        self.rate_dim = num_users

        self.tmax = t_max

        self.packet_size = 100

        self.to_transmit = 0

        self.bound = bound

        self.tx_power = 5

        self.p = p

        self.pmax = 1
        self.mu = mu
        self.num_users = num_users

        if (CSI==1).all():
            self.sample_source = "Random"
            self.CSI = None
        else:
            self.sample_source = "Loaded"
            self.CSI = CSI

        self.users = np.random.choice(48,num_users,replace=False)
        self.states = np.random.normal(loc=0.0, scale = 1.0, size=(num_users))


        self.rate_by_mcs = [1.6,2.4,3.3,4.9,6.5,7.3,8.1,9.8,10.8,12.2,13.5]
        self.rate_by_mcs = np.asarray(self.rate_by_mcs)
        self.bw = 2

        T = scipy.io.loadmat('snr_info.mat')
        self.per_by_snr = T.get('mcs_snr_per_wcl_20B')

        if Ac == None:
            self.Ac = (.9-.6)*np.random.random_sample(num_users)+0.6
            #self.Ac = 0.8 * np.ones(num_users)
        else:
            if (Ac.shape != (1, num_users)):
                raise Exception("Ac is not the correct shape")
            self.Ac = Ac

        if Ao == None:
            self.Ao = (1.03-1.001)*np.random.random_sample(num_users)+1.001
           # self.Ao = 1.01 * np.ones(num_users)
        else:
            if (Ao.shape != (1, num_users)):
                raise Exception("Ao is not the correct shape")
            self.Ao = Ao

        if W == None:
            self.W = np.ones(num_users)
        else:
            if (W.shape != (1, num_users)):
                raise Exception("W is not the correct shape")
            self.W = W

    def sample(self, batch_size):
        #return self._exponential_normal_sample(batch_size)
        if self.sample_source == "Random":
            return self._exponential_uniform_sample(batch_size, self.bound)
        else:
            cycle = np.random.randint(1000, size=batch_size)
            temp = self.CSI[self.users,:]
            channel = temp[:,cycle].T
           # control = np.random.uniform(-self.bound, self.bound, size=(batch_size, self.control_state_dim))
            control = np.tile(self.states, [batch_size, 1])
            return np.hstack((channel,control))

    def db_to_col(self,snr_value):
        snr_value = np.minimum(snr_value,35)
        snr_value = np.maximum(snr_value,-35)

        col = np.floor((snr_value+35)/0.1+1)-1
        return col.astype(int)

    def packet_delivery_rate(self,snr_value,mcs,per_by_snr):
        
        col = self.db_to_col(snr_value)
        pdr = 1 - per_by_snr[mcs-1,col]
        return pdr

    def snr_to_db(self,tx_power,noise_snr):
        db = 10*np.log10(tx_power*noise_snr)
        return db

    def mcs_to_time(self,mcs,packet_size):
        time = 8*packet_size*0.000001/(self.rate_by_mcs[mcs-1]*self.bw)
        return time


    def rate_to_mcs(self,rate):
        rate2 = np.asarray(rate)
        mcs = np.digitize(rate2,self.rate_by_mcs) 
        mcs = np.minimum(mcs,10)
        return mcs


    def f0(self, state, action, pause=False):
        """
        objective function (total transmission time)
        Args:
            state (TYPE): channel state, N by state_dim matrix
            action (TYPE): power allocation, N by action_dim matrix

        Returns:
            TYPE: N by 1 matrix of negative total channel capacity
        """
        N = action.shape[0]
        total_cost = 0
        T = state.shape[2]
  
        if T < 2:

            rate_action = action[:,0:self.rate_dim,0]
            transmit_action = action[:,self.rate_dim:,0]
            mcs_action = self.rate_to_mcs(rate_action)


            channel_state = state[:,0:self.channel_state_dim,0]
            control_state = state[:,self.channel_state_dim:,0]
            control_state_m = np.reshape(control_state,(N,self.num_users))
        
            cost_c = np.square(control_state_m * np.tile(self.Ac,(N,1)))
            cost_o = np.square(control_state_m * np.tile(self.Ao,(N,1))) 

            if self.sample_source == "Random":
                snr_state = self.snr_to_db(self.tx_power,channel_state)
            else:
                snr_state = channel_state

            if pause:
                pdb.set_trace()

            delivery_rates = transmit_action * self.packet_delivery_rate(snr_state,mcs_action,self.per_by_snr)

            total_cost = delivery_rates * cost_c + (1-delivery_rates) * cost_o
            

        else:
            T = state.shape[2]

            control_state = state[:,self.channel_state_dim:,:]
            control_state_m = np.reshape(control_state,(N,self.num_users,T))
        
            cost_c = np.square(control_state_m) 
            total_cost = np.sum(cost_c,axis=2)

        total_cost = np.sum(total_cost,axis=1)
        return np.reshape(total_cost,(-1,1))

    def get_time(self,action):
        """
        Compute transmission time for each user in each RU
        Inputs: State (N x state_dim)
        Outputs: Transmission time per RU (N x num_users x num_RUs)
        """

        rate_action = action[:,0:self.rate_dim]
        transmit_action = action[:,self.rate_dim:]

        mcs_action = self.rate_to_mcs(rate_action)
        total_time = transmit_action * self.mcs_to_time(mcs_action,self.packet_size)
        total_time = np.sum(total_time,axis=1)

        return total_time 

        
        
    def f1(self, state, action):
        """
        constraint functions (no intersecting channels)
        Args:
            state (TYPE): channel state, N by state_dim matrix
            action (TYPE): power allocation, N by action_dim matrix
        
        Returns:
            TYPE: N by constrain_dim matrix of power budget violations
        """

        N = state.shape[0]
        lhs = np.zeros((N,self.constraint_dim))

        for i in np.arange(self.constraint_dim):
            total_time = self.get_time(action[:,:,i])

           # lhs = 25000*(total_time - self.tmax)
            #lhs = np.zeros((N,self.constraint_dim))
            lhs[:,i] = 1000*(np.asarray(total_time > self.tmax).astype(int) - 0.05)
            #lhs[:,0:self.num_channels] = np.array([np.sum(ru_mat, axis=1) - 0*np.ones((N,self.num_channels))])

        return lhs

    
    def update_system(self, state, action,batch_size, rate_select=0):


        rate_action = action[:,0:self.rate_dim]
        transmit_action = action[:,self.rate_dim:]

        if rate_select:
            mcs_action = self.rate_to_mcs(rate_action)
        else:
            mcs_action = rate_action.astype(int)

      #  pdb.set_trace()

        channel_state = state[:,0:self.channel_state_dim]
        control_state = state[:,self.channel_state_dim:]

        control_state_m = np.reshape(control_state,(batch_size,self.num_users))

        if self.sample_source == "Random":
            snr_state = self.snr_to_db(self.tx_power,channel_state)
        else:
            snr_state = channel_state

        delivery_rates = transmit_action * self.packet_delivery_rate(snr_state,mcs_action,self.per_by_snr)

        q = np.random.binomial(1,delivery_rates)

        control_state_c = np.copy(control_state)
        for i in np.arange(batch_size):
            control_state_c[i,:] = q[i,:]*(self.Ac*control_state_c[i,:]) + (1-q[i,:])*(self.Ao*control_state_c[i,:])

        channel_state = np.random.exponential(self.mu, size=(batch_size, self.channel_state_dim))
        control_state_c = np.minimum(np.maximum(control_state_c,-self.bound),self.bound)
        new_state = np.concatenate([channel_state,control_state_c],axis=1)

        return new_state

    def calls(self,state, tmax=None):

        if tmax==None:
            tmax = self.tmax

        transmit_action = np.zeros((1,self.num_users))
        mcs_action = np.ones((1,self.num_users))

        rho = .95
        q_min = (np.square(self.Ao) - rho) / (np.square(self.Ao) - np.square(self.Ac))
        q_min = np.maximum(0,np.minimum(q_min,1))

        v = np.exp(q_min-1)
        p_min = q_min/v

        draws = np.random.binomial(1,v)
        all_to_transmit = np.random.permutation(np.where(draws)[0])
        idx = 0
        num_to_transmit = np.size(all_to_transmit)

        if num_to_transmit > 0:
        
            N = state.shape[0]
            total_time = 0

            channel_state = state[0,0:self.channel_state_dim]



            if self.sample_source == "Random":
                snr_state = self.snr_to_db(self.tx_power,channel_state)
            else:
                snr_state = channel_state
            col_state = self.db_to_col(snr_state)
            p2 = 1 - self.per_by_snr[:,col_state]

        
            max_mcs = np.maximum(np.sum(p2[:,all_to_transmit[idx]]>=p_min[all_to_transmit[idx]]),1)
            mcs_action[0,all_to_transmit[idx]] = max_mcs
            time_needed = self.mcs_to_time(max_mcs,self.packet_size)
            total_time += time_needed

            while (total_time <= tmax) & (np.sum(transmit_action) < num_to_transmit):
                transmit_action[0,all_to_transmit[idx]] = 1.0
                idx = (idx + 1)%num_to_transmit
                max_mcs = np.maximum(np.sum(p2[:,all_to_transmit[idx]]>=p_min[all_to_transmit[idx]]),1)
                mcs_action[0,all_to_transmit[idx]] = max_mcs
                time_needed = self.mcs_to_time(max_mcs,self.packet_size)
                total_time += time_needed


        return np.concatenate([mcs_action, transmit_action], axis=1)

    def priority_ranking(self,state,tmax=None):

        if tmax==None:
            tmax = self.tmax

        p_min = 0.95

        control_state = state[0,self.channel_state_dim:]

        all_to_transmit = np.argsort(-abs(control_state))
        idx = 0
        num_to_transmit = np.size(all_to_transmit)
        
        N = state.shape[0]
        total_time = 0

        channel_state = state[0,0:self.channel_state_dim]

        transmit_action = np.zeros((1,self.num_users))
        mcs_action = np.ones((1,self.num_users))

        if self.sample_source == "Random":
            snr_state = self.snr_to_db(self.tx_power,channel_state)
        else:
            snr_state = channel_state


        col_state = self.db_to_col(snr_state)
        p2 = 1 - self.per_by_snr[:,col_state]

        
        max_mcs = np.maximum(np.sum(p2[:,all_to_transmit[idx]]>=p_min),1)
        mcs_action[0,all_to_transmit[idx]] = max_mcs
        time_needed = self.mcs_to_time(max_mcs,self.packet_size)
        total_time += time_needed

        while (total_time <= tmax) & (np.sum(transmit_action) < num_to_transmit):
            transmit_action[0,all_to_transmit[idx]] = 1.0
            idx = (idx + 1)%num_to_transmit
            max_mcs = np.maximum(np.sum(p2[:,all_to_transmit[idx]]>=p_min),1)
            mcs_action[0,all_to_transmit[idx]] = max_mcs
            time_needed = self.mcs_to_time(max_mcs,self.packet_size)
            total_time += time_needed

        return np.concatenate([mcs_action, transmit_action], axis=1)


    def round_robin(self,state, tmax=None):

        if tmax==None:
            tmax = self.tmax

        p_min = 0.95
        
        N = state.shape[0]
        total_time = 0

        channel_state = state[0,0:self.channel_state_dim]

        transmit_action = np.zeros((1,self.num_users))
        mcs_action = np.ones((1,self.num_users))

        if self.sample_source == "Random":
            snr_state = self.snr_to_db(self.tx_power,channel_state)
        else:
            snr_state = channel_state
        col_state = self.db_to_col(snr_state)
        p2 = 1 - self.per_by_snr[:,col_state]

        
        max_mcs = np.maximum(np.sum(p2[:,self.to_transmit]>=p_min),1)
        mcs_action[0,self.to_transmit] = max_mcs
        time_needed = self.mcs_to_time(max_mcs,self.packet_size)
        total_time += time_needed

        while (total_time <= tmax) & (np.sum(transmit_action) < self.num_users):
            transmit_action[0,self.to_transmit] = 1.0
            self.to_transmit = (self.to_transmit + 1)%self.num_users
            max_mcs = np.maximum(np.sum(p2[:,self.to_transmit]>=p_min),1)
            mcs_action[0,self.to_transmit] = max_mcs
            time_needed = self.mcs_to_time(max_mcs,self.packet_size)
            total_time += time_needed


        return np.concatenate([mcs_action, transmit_action], axis=1)


if __name__ == '__main__':

    mu = 1 # parameter for exponential distribution of wireless channel distribution

    # load channel info
    DD = scipy.io.loadmat("channel_info_short_python.mat")
    CSI = DD['H2'] - 10

    num_users = 5 # number of devices 
    tmax = .0005  # latenct bound

    # bounds for data rate policy ###
    lower_bound = 1.6
    upper_bound = 5.0

    # learning rates
    theta_lr = 5e-5
    lambda_lr = .01

     # Initialize scheduling system
    sys = WirelessSchedulingSystem_OFDMA(num_users, tmax=tmax, mu=mu, CSI=CSI)

    N = 10
    state = sys.sample(N)
    rate_action = np.random.uniform(2,5,size=(N,num_users))
    
    transmit_action = np.random.binomial(1,0.5,size=(N,sys.transmit_dim))
    action = np.concatenate([rate_action, transmit_action], axis=1)

    state = np.expand_dims(state,axis=2)
    action = np.expand_dims(action,axis=2)
    print(action.shape)

    f2 = sys.f0(state,action,pause=True)

    action = np.array([[0, 1, 8, 3, 2], [3,1, 4, 5,2]])

    sys.get_binary(action)
    pdb.set_trace()
