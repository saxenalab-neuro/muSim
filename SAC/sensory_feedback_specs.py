import numpy as np
from . import perturbation_specs

#Add perturbations to the sensory feedback
stim_feedback_pert = perturbation_specs.stim_feedback_pert
muscle_lengths_pert = perturbation_specs.muscle_lengths_pert
muscle_velocities_pert = perturbation_specs.muscle_velocities_pert
muscle_forces_pert = perturbation_specs.muscle_forces_pert
joint_positions_pert = perturbation_specs.joint_positions_pert
joint_velocities_pert = perturbation_specs.joint_velocities_pert
visual_position_pert = perturbation_specs.visual_position_pert
visual_velocity_pert = perturbation_specs.visual_velocity_pert
visual_distance_pert = perturbation_specs.visual_distance_pert


#Functions to process the sensory feedback from the environment before it enters the uSim controller
def process_stimulus(stim_feedback):
	
	#Input: list objects
	#Imp: First convert the input to numpy arrays, do the processing, and then convert the output
	#back to list object before returning for all sensory feedback functions. 

	return stim_feedback

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
def process_stimulus_pert(stim_feedback, istep):

	#print dim
	istep -= 1
	if istep == 0:
		print('Stimulus Feedback Dimension is', len(stim_feedback))

	stim_feedback = np.array(stim_feedback)

	if len(stim_feedback_pert) != 0:
		stim_feedback += stim_feedback_pert[istep, :]

	return stim_feedback.tolist()

def process_proprioceptive_pert(muscle_lengths, muscle_velocities, istep):

	#print dim
	istep -= 1
	if istep == 0:
		print('Muscle Lengths Dimension is', len(muscle_lengths))
		print('Muscle Velocities Dimension is', len(muscle_velocities))
	
	muscle_lengths = np.array(muscle_lengths)
	muscle_velocities = np.array(muscle_velocities)

	if len(muscle_lengths_pert) != 0:
		muscle_lengths += muscle_lengths_pert[istep, :]

	if len(muscle_velocities_pert) != 0:
		muscle_velocities += muscle_velocities_pert[istep, :]

	return muscle_lengths.tolist(), muscle_velocities.tolist()

def process_muscle_forces_pert(muscle_forces, istep):

	#print dim
	istep -= 1
	if istep == 0:
		print('Muscle Forces Dimension is', len(muscle_forces))

	muscle_forces = np.array(muscle_forces)

	if len(muscle_forces_pert) != 0:
		muscle_forces += muscle_forces_pert[istep, :]
	
	return muscle_forces.tolist()

def process_joint_feedback_pert(sensory_joint_positions, sensory_joint_velocities, istep):

	#print dim
	istep -= 1
	if istep == 0:
		print('Sensory Joint Positions Dimension is', len(sensory_joint_positions))
		print('Sensory Joint Velocities Dimension is', len(sensory_joint_velocities))

	sensory_joint_positions = np.array(sensory_joint_positions)
	sensory_joint_velocities = np.array(sensory_joint_velocities)

	if len(joint_positions_pert) != 0:
		sensory_joint_positions += joint_positions_pert[istep, :]

	if len(joint_velocities_pert) != 0:
		sensory_joint_velocities += joint_velocities_pert[istep, :]

	return sensory_joint_positions.tolist(), sensory_joint_velocities.tolist()

def process_visual_position_pert(visual_xyz_coords, istep):

	#print dim
	istep -= 1
	if istep == 0:
		print('Visual Positions Dimension is', len(visual_xyz_coords))

	visual_xyz_coords = np.array(visual_xyz_coords)

	if len(visual_position_pert) != 0:
		
		visual_xyz_coords += visual_position_pert[istep, :]

	return visual_xyz_coords.tolist()

def process_visual_distance_pert(visual_xyz_distance, istep):

	#print dim
	istep -= 1
	if istep == 0:
		print('Visual Distance Dimension is', len(visual_xyz_distance))

	visual_xyz_distance = np.array(visual_xyz_distance)

	if len(visual_distance_pert) != 0:
		
		visual_xyz_distance += visual_distance_pert[istep, :]

	return visual_xyz_distance.tolist()

def process_visual_velocity_pert(visual_xyz_velocity, istep):

	#print dim
	istep -= 1
	if istep == 0:
		print('Visual Velocity Dimension is', len(visual_xyz_velocity))

	visual_xyz_velocity = np.array(visual_xyz_velocity)

	if len(visual_velocity_pert) != 0:
		visual_xyz_velocity += visual_velocity_pert[istep, :]

	return visual_xyz_velocity.tolist()