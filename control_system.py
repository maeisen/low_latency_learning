import numpy as np
import scipy.io
from functools import partial

class ProbabilisticSystem(object):
    def __init__(self, state_dim, action_dim, constraint_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim

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


class WirelessSchedulingSystem(ProbabilisticSystem):
    def __init__(self, num_users, p, Ao=None, Ac=None, W=None, mu=2, S=1, num_rus = 24,num_channels=9):
        super().__init__(num_users*num_rus*10 + num_users*p, num_users*num_channels + num_users, num_users+S*num_channels)

        self.channel_state_dim = num_users*num_rus*10
        self.control_state_dim = num_users*p
        self.ru_dim = num_users*num_channels
        self.mcs_dim = num_users

        self.p = p

        self.pmax = 1
        self.mu = mu
        self.num_users = num_users
        self.num_rus = num_rus
        self.S = S
        self.num_channels = num_channels


        self.rate_by_mcs = [1.6,2.4,3.3,4.9,6.5,7.3,8.1,9.8,10.8,12.2,13.5]

        T = scipy.io.loadmat('snr_info.mat')
        self.per_by_snr = T.get('mcs_snr_per_wcl_20B')

        if Ac == None:
            self.Ac = 0.6*np.random.random_sample((1, num_users))
        else:
            if (Ac.shape != (1, num_users)):
                raise Exception("Ac is not the correct shape")
            self.Ac = Ac

        if Ao == None:
            self.Ao = (1.4-1)*np.random.random_sample((1, num_users))+1
        else:
            if (Ao.shape != (1, num_users)):
                raise Exception("Ao is not the correct shape")
            self.Ao = Ao

        if W == None:
            self.W = np.ones((1, num_users))
        else:
            if (W.shape != (1, num_users)):
                raise Exception("W is not the correct shape")
            self.W = W

    def sample(self, batch_size):
        return self._exponential_sample(batch_size)

    def snr_to_col(self,snr_value):
        snr_value = np.minimum(snr_value,35)
        snr_value = np.maximum(snr_value,-35)

        col = np.floor((snr_value+35)/0.1+1)-1
        return col

    def packet_delivery_rate(self,snr_value,mcs,per_by_snr):
        col = self.snr_to_col(snr_value)
        pdr = 1 - per_by_snr[mcs-1,col]
        return pdr

    def f0(self, state, action):
        """
        objective function for wireless capacity
        inputs are state (channel state) and action (power allocation),
        outputs capacity


        Args:
            state (TYPE): channel state, N by state_dim matrix
            action (TYPE): power allocation, N by action_dim matrix

        Returns:
            TYPE: N by 1 matrix of negative total channel capacity
        """
        # snr_value = channel_state[:,action,mcs]
        # packet_error_rate = self.packet_error_rate(snr_value,mcs,self.per_by_snr)
        # lyop = packet_error_rate*(control_state**2)*self.Ac**2 + (1-packet_error_rate)*(control_state**2)*self.Ao**2 + self.W
        # total_cap = -np.sum(lyop, axis=1)
        # total_cap = np.reshape(total_cap, (-1, 1))
        # return total_cap

        mcs = action[:,0:self.mcs_dim]
        rus = action[:,self.mcs_dim:]

        

        N = action.shape[0]
        
        packet_size = 100
        bw_per_user = np.sum(np.reshape(rus,(N,self.num_users,self.num_channels)),axis=2)
        
        time_per_user = np.zeros((N,self.num_users))
        for i in np.arange(self.num_users):
            if bw_per_user[i]>0:
                time_per_user[:,i] = 8*packet_size*0.000001/(self.rate_by_mcs[mcs[:,i]-1]*bw_per_user[:,i])   
        
        total_time = np.sum(time_per_user,axis=1)
        return total_time

    def get_binary(self, action):

        a2 = action
        f = np.zeros(shape=(self.action_dim,9))
        index = 1
        for i in np.arange(0,len(a2)):
            x = a2[i]
            condlist = [np.equal(x,0),np.equal(x,1),np.equal(x,2),np.equal(x,3),np.equal(x,4),np.equal(x,5),np.equal(x,6),np.equal(x,7),np.equal(x,8),np.equal(x,9),np.equal(x,10),np.equal(x,11),np.equal(x,12),np.equal(x,13),np.equal(x,14),np.equal(x,15),np.equal(x,16),np.equal(x,17),np.equal(x,18),np.equal(x,19),np.equal(x,20),np.equal(x,21),np.equal(x,22),np.equal(x,23),np.equal(x,24)]
            choiselst = ([[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]])
            f[i] = np.select(condlist,choiselst)
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

        channel_state_mat = np.reshape(channel_state,(N,self.num_users,self.num_rus,10))
        control_state_mat = np.reshape(channel_state,(N,self.num_users,p))

        time_per_user_per_ru = np.zeros((N,self.num_users,self.num_rus))
        for i in self.num_users:
            for j in self.num_rus:
                time_per_user_per_ru[:,i,j] = 

        return time_per_user_per_ru 

        
        
    def f1(self, state, action):
        """
        constraint functions for wireless capacity
        inputs are state (channel state) and action (power allocation)
        outputs budget

        Args:
            state (TYPE): channel state, N by state_dim matrix
            action (TYPE): power allocation, N by action_dim matrix
        
        Returns:
            TYPE: N by constrain_dim matrix of power budget violations
        """

        mcs = action[:,0:self.mcs_dim]
        rus = action[:,self.mcs_dim:]
        ru_mat = np.reshape(rus,(N,self.num_users,self.num_channels))
        N = np.size(state,0)

        lhs = np.zeros((N,self.constraint_dim))
        lhs[:,0:self.num_channels] = np.array([np.sum(ru_mat, axis=2) - np.ones(N,self.num_channels)])


        channel_state = state[:,0:self.channel_state_dim]
        control_state = state[:,self.channel_state_dim:]
        state_mat = np.reshape(channel_state,(N,self.num_users,self.num_rus,10))
        for i in np.arange(self.num_users):
            lhs[:,self.num_channels+1] = 

        return lhs
