import numpy as np
import scipy.io

def snr_to_db(tx_power,noise_snr):
    db = 10*np.log10(tx_power/noise_snr)
    return db

def db_to_col(snr_value):
    snr_value = np.minimum(snr_value,35)
    snr_value = np.maximum(snr_value,-35)

    col = np.floor((snr_value+35)/0.1+1)-1
    return col.astype(int)

def mcs_to_time(rate_by_mcs,mcs,bw,packet_size):
	time = 8*packet_size*0.000001/(rate_by_mcs[mcs]*np.tile(bw,(mcs.shape[0],1)))
	return time


if __name__ == '__main__':

	num_users = 6
	num_channels = 9
	num_rus = 24
	T = scipy.io.loadmat('snr_info.mat')
	per_by_snr = T.get('mcs_snr_per_wcl_20B')
	rate_by_mcs = np.array([1.6,2.4,3.3,4.9,6.5,7.3,8.1,9.8,10.8,12.2,13.5])
	bw_by_ru = [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,4,4,4,4,4,4,10]
	N = 12

	rho = 0.95

	channel_state_mat = np.random.exponential(1,(N,num_users,num_rus))
	pp = snr_to_db(2,channel_state_mat)
	channel_state_col = db_to_col(pp)
	mcs_per_user_per_ru = np.zeros((N,num_users,num_rus))
	time_per_user_per_ru = np.zeros((N,num_users,num_rus))

	Ac = 0.6*np.random.random_sample(num_users)
	Ao = (1.4-1)*np.random.random_sample(num_users)+1

	pp = 1 - per_by_snr[:,channel_state_col]

	#print(pp[:,1,1,:])
	for i in np.arange(num_users):
		p_min = (np.power(Ao[i],2)-rho) / (np.power(Ao[i],2) - np.power(Ac[i],2))
		print(p_min)
		mcs_per_user_per_ru[:,i,:] = np.maximum(np.sum(pp[:,:,i,:]>=p_min,axis=0),1)-1
		time_per_user_per_ru[:,i,:] = mcs_to_time(rate_by_mcs,mcs_per_user_per_ru[:,i,:].astype(int),bw_by_ru,100)


	print(pp[:,1,1,:]) 
	print(mcs_per_user_per_ru[1,1,:])
	print(time_per_user_per_ru[1,1,:])
	#print(pdr_per_ru)
