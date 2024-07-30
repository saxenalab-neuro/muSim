import numpy as np
#Specifies the sensory feedback to the agent/network
#True, if this feedback should be included in state feedback to the agent's network/controller
#False, if this feedback should not be included in the state feedback to the agent's network/controller

#Proprioceptive feedback consists of muscle lengths and velocities
proprioceptive_feedback = True

#Muscle forces consist of appled muscle forces 
muscle_forces = False

#Joint feedback consists of joint positions and velocities
joint_feedback = False 

#Visual feedback consists of x/y/z coordinates of the specified bodies in the model
#If visual_feedback is True, specify the names of the bodies from musculoskeletal_model.xml for which the feedback should be included
visual_feedback = False 

#Append the musculo bodies from which you 
#This list can consist of targets as specified in kin_train/kin_test.pkl
#Append targetn-1 for visual feedback from targets
#'target0' corresponds to the visual feedback from the first target, target1 to the second target and so on
visual_feedback_bodies = ['hand', 'target0'] 

###--------- DO NOT change this -------------------
if visual_feedback == True:
	assert len(visual_feedback_bodies) != 0 
### -----------------------------------------------

#Specify the names of the bodies as tuples for which the visual distance should be included in the feedback
#Leave blank if the visual distance is not to be included in the feedback
#Absoulte distance between the bodies will be included
visual_distance_bodies = [('hand', 'target0')] 

#Specify the names of the bodies for which the visual velocity should be included in the feedback
#Leave blank if the visual velocity is not to be included in the feedback
#Appends the absolute musculo body velocity, e.g. visual_velocity = ['hand', 'target0'] 
#will include the xyz velocities of hand and target0
visual_velocity = []

#Specify the delay in the sensory feedback in terms of the timepoints
sensory_delay_timepoints = 0 


#Sensory feedback elimination specs: when mode in configs.py is "SFE"
#Feedback to eliminate can include ["proprioceptive", "muscle_forces", "joint_feedback", "visual_position", "visual_distance", "visual_velocity", "task_scalar", "recurrent_connections"]
sf_elim = ["proprioceptive"] 

#Sensory feedback perturbation when the mode in configs.py is "sensory_pert"
#Shape of the vectors, 
muscle_lengths_pert = []
muscle_velocities_pert = []
muscle_forces_pert = []
joint_positions_pert = []
joint_velocities_pert = []
visual_position_pert = []
visual_velocity_pert = []
visual_distance_pert = []


#Functions to process the sensory feedback from the environment before it enters the uSim controller
def process_proprioceptive(muscle_lengths, muscle_velocities):
	
	#Input: list objects
	#Imp: First convert the input to numpy arrays, do the processing, and then convert the output
	#back to list object before returning for all sensory feedback functions. 

	return muscle_lengths, muscle_velocities

def process_muscle_forces(muscle_forces):
	
	return muscle_forces

def process_joint_feedback(sensory_joint_positions, sensory_joint_velocities):

	return sensory_joint_positions, sensory_joint_velocities

def process_visual_position(visual_xyz_coords):

	return visual_xyz_coords

def process_visual_distance(visual_xyz_distance):

	visual_xyz_distance = np.array(visual_xyz_distance)
	#process using the saturating non-linearity
	visual_xyz_distance = np.tanh(visual_xyz_distance*100).tolist()

	return visual_xyz_distance

def process_visual_velocity(visual_xyz_velocity):

	return visual_xyz_velocity


### -------------------------------------------------------- ----------   ####
## DO NOT change these functions -------------------------------------------
#Functions to process the sensory feedback for sensory pert
def process_proprioceptive_pert(muscle_lengths, muscle_velocities):
	
	muscle_lengths = np.array(muscle_lengths)
	muscle_velocities = np.array(muscle_velocities)

	if len(muscle_lengths_pert) != 0:
		muscle_lengths += muscle_lengths_pert

	if len(muscle_velocities_pert) != 0:
		muscle_velocities += muscle_velocities_pert

	return muscle_lengths.tolist(), muscle_velocities.tolist()

def process_muscle_forces_pert(muscle_forces):

	muscle_forces = np.array(muscle_forces)

	if len(muscle_forces_pert) != 0:
		muscle_forces += muscle_forces_pert
	
	return muscle_forces.tolist()

def process_joint_feedback_pert(sensory_joint_positions, sensory_joint_velocities):

	sensory_joint_positions = np.array(sensory_joint_positions)
	sensory_joint_velocities = np.array(sensory_joint_velocities)

	if len(joint_positions_pert) != 0:
		sensory_joint_positions += joint_positions_pert

	if len(joint_velocities_pert) != 0:
		sensory_joint_velocities += joint_velocities_pert

	return sensory_joint_positions.tolist(), sensory_joint_velocities.tolist()

def process_visual_position_pert(visual_xyz_coords):

	visual_xyz_coords = np.array(visual_xyz_coords)

	if len(visual_position_pert) != 0:
		
		visual_xyz_coords += visual_position_pert

	return visual_xyz_coords.tolist()

def process_visual_distance_pert(visual_xyz_distance):

	visual_xyz_distance = np.array(visual_xyz_distance)

	if len(visual_distance_pert) != 0:
		
		visual_xyz_distance += visual_distance_pert

	return visual_xyz_distance.tolist()

def process_visual_velocity_pert(visual_xyz_velocity):

	visual_xyz_velocity = np.array(visual_xyz_velocity)

	if len(visual_velocity_pert) != 0:
		visual_xyz_velocity += visual_velocity_pert

	return visual_xyz_velocity.tolist()