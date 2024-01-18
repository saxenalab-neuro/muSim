import numpy as np
from scipy.io import loadmat
import pickle
import os

n_train_conds = 5
n_test_conds = 1

train_conds = [702, 803, 932, 1106, 1319]
test_conds = [619]

train_timepoints = [[25+3*702,25+4*702], [25+3*803,25+4*803], [25+3*932,25+4*932], [25+2*1106,25+3*1106], [25+2*1319,25+3*1319]]
test_timepoints = [[25+2*619,25+3*619,]]

RNN_train_timepoints = [[100+3*702,100+4*702], [100+3*803,100+4*803], [100+3*932,100+4*932], [100+2*1106,100+3*1106], [100+1*1319,100+2*1319]]
RNN_test_timepoints = [[100+3*619,100+4*619]]

# ----------------- nuSim neural activities for training and testing conditions ---------------------

#Load the nuSim neural activities for training
data = {}
for idx in range(n_train_conds):
	activity = np.load(f"./Training/nusim_activity_c{idx+1}.npy")
	
	#Select the corresponding training timepoints
	activity = activity[train_timepoints[idx][0]:train_timepoints[idx][1]]
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./nusim_train.pkl"):
	os.remove("./nusim_train.pkl")

#Save as nuSim_train
with open('nusim_train.pkl', 'wb') as f:
    pickle.dump(data, f)


#Load the nuSim neural activities for testing
data = {}
for idx in range(n_test_conds):
	activity = np.load(f"./Testing/nusim_activity_c{idx+1}.npy")
	
	#Select the corresponding training timepoints
	activity = activity[test_timepoints[idx][0]:test_timepoints[idx][1]]
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./nusim_test.pkl"):
	os.remove("./nusim_test.pkl")

#Save as nuSim_test
with open('nusim_test.pkl', 'wb') as f:
    pickle.dump(data, f)

# ------------------ Experimental neural activities for training and testing -----------------------------

# Load the experimental activities for training
data= {}

for idx in range(n_train_conds):
	activity = loadmat(f"./Training/exp_activity_c{idx+1}.mat")['activity']

	#Select the corresponding training timepoints
	activity = activity[:]
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./exp_train.pkl"):
	os.remove("./exp_train.pkl")

#Save as exp_train
with open('exp_train.pkl', 'wb') as f:
    pickle.dump(data, f)

# Load the experimental activities for testing
data= {}

for idx in range(n_test_conds):
	activity = loadmat(f"./Testing/exp_activity_c{idx+1}.mat")['activity']

	#Select the corresponding training timepoints
	activity = activity[:]
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./exp_test.pkl"):
	os.remove("./exp_test.pkl")

#Save as exp_test
with open('exp_test.pkl', 'wb') as f:
    pickle.dump(data, f)

# ----------------------- EMG for training and testing ------------------------------------------------------
# Load the experimental emg for training

data = {}

for idx in range(n_train_conds):
	activity = loadmat(f"./Training/emg_c{idx+1}.mat")[f'emg_{train_conds[idx]}']
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./emg_train.pkl"):
	os.remove("./emg_train.pkl")

#Save as emg_train
with open('emg_train.pkl', 'wb') as f:
    pickle.dump(data, f)

# Load the experimental emg for testing

data = {}

for idx in range(n_test_conds):
	activity = loadmat(f"./Testing/emg_c{idx+1}.mat")[f'emg_{test_conds[idx]}']
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./emg_test.pkl"):
	os.remove("./emg_test.pkl")

#Save as emg_test
with open('emg_test.pkl', 'wb') as f:
    pickle.dump(data, f)

# ------------------------ Kinematics for training and testing -------------------------------------------------

#Load the kinematics for training
data = {}

for idx in range(n_train_conds):
	activity = loadmat(f"./Training/kin_c{idx+1}.mat")['kin']
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./kin_train.pkl"):
	os.remove("./kin_train.pkl")

#Save the kin_train
with open('kin_train.pkl', 'wb') as f:
    pickle.dump(data, f)


#Load the kinematics for testing
data = {}
for idx in range(n_test_conds):
	activity = loadmat(f"./Testing/kin_c{idx+1}.mat")['kin']
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./kin_test.pkl"):
	os.remove("./kin_test.pkl")

#Save the kin_test
with open('kin_test.pkl', 'wb') as f:
    pickle.dump(data, f)


# ------------------------ RNN activities for training and testing -------------------------------------------------

#Load the RNN activities for training
data = {}

for idx in range(n_train_conds):
	activity = np.load(f"./Training/RNN_activity_c{idx+1}.npy")[0, RNN_train_timepoints[idx][0]:RNN_train_timepoints[idx][1], :]
	data[idx] = activity

#Check if the file already exists and delete it
if os.path.exists("./RNN_train.pkl"):
	os.remove("./RNN_train.pkl")

#Save the kin_train
with open('RNN_train.pkl', 'wb') as f:
    pickle.dump(data, f)


#Load the RNN activities for testing
data = {}
for idx in range(n_test_conds):
	activity = np.load(f"./Testing/RNN_activity_c{idx+1}.npy")[0, RNN_test_timepoints[idx][0]:RNN_test_timepoints[idx][1], :]
	data[idx] = activity
	
#Check if the file already exists and delete it
if os.path.exists("./RNN_test.pkl"):
	os.remove("./RNN_test.pkl")
	
#Save the kin_test
with open('RNN_test.pkl', 'wb') as f:
    pickle.dump(data, f)