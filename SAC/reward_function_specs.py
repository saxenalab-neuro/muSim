import numpy as np
from . import kinematics_preprocessing_specs

#Reward function is a function of the sensory_feedback (do indexing to get desired state features), 
#current simulator_state(also to avoid indexing the sensory_feedback in case of no delays) and current action a_t

musculo_tracking = kinematics_preprocessing_specs.musculo_tracking

#Whether or not to implement the minimum muscle effort constraint
min_muscle_constraint = False
#Specify the scaler to weight the muscle effort
muscle_cost_scaler = 1/50

#Reward Scaling Factor exponentially scales the distance between the body/end-effector and the target it has to track
#For a smaller threshold, use a higher reward scaling factor
reward_scaling_factor = 1000

#Threshold crossing penalty is imposed if any of the body/end-effector xyz pos goes outside the thresholding region
threshold_crossing_penalty = -5

def reward_function(state_td, data, model, action_t, threshold):
	
        xyz_coord_dists = []

        for musculo_body_tracking in musculo_tracking:
            musculo_body = data.xpos[model.body(musculo_body_tracking[0]).id].flat.copy()
            musculo_target = data.xpos[model.body(musculo_body_tracking[1]).id].flat.copy()

            current_dists = np.abs(musculo_body - musculo_target)
            xyz_coord_dists = [*xyz_coord_dists, *current_dists]

            #If any body goes out of the movement thresholding region return a very high penalty
            if (np.array(current_dists) > threshold).any():
                return threshold_crossing_penalty

        xyz_coord_dists = np.array(xyz_coord_dists)
        #Implement the exponential reward scaling 
        reward_exp = 1/(reward_scaling_factor**xyz_coord_dists)
        reward = np.sum(reward_exp)


        if min_muscle_constraint:
            muscle_cost = muscle_effort_cost(action_t)
            reward = reward - muscle_cost

        return reward

def muscle_effort_cost(action_t):

        cost= muscle_cost_scaler * np.sum(np.abs(action_t))
        return cost