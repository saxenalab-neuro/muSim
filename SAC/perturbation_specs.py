import numpy as np

#Sensory feedback elimination specs: when mode in configs.py is "SFE"
#Feedback to eliminate can include ["proprioceptive", "muscle_forces", "joint_feedback", "visual_position", "visual_distance", "visual_velocity", "task_scalar", "recurrent_connections", "stimulus"]
sf_elim = ["task_scalar"]

#Sensory feedback perturbation when the mode in configs.py is "sensory_pert"
#The perturbation vectors should be np.ndarrays
#Shape of the vectors: [args.timestep_limit, n_features_in_sensory_part]

#Add perturbation to the stimulus feedback
stim_feedback_pert = []

#Add perturbation to muscle lengths
muscle_lengths_pert = []

#Add perturbation to muscle velocities
muscle_velocities_pert = []

#Add perturbation to muscle forces
muscle_forces_pert = []

#Add perturbation to joint positions
joint_positions_pert = []

#Add perturbation to joint velocities
joint_velocities_pert = []

#Add perturbation to visual positions
visual_position_pert = []

#Add perturbation to visual velocities
visual_velocity_pert = []

#Add perturbation to visual distances
visual_distance_pert = []

#Neural perturbation when the mode in configs.py is "neural_pert"
#The neural perturbation is added to the neurons of the uSim/nuSim neurons/units
#It should be a numpy array of the shape [timepoints, n_hidden_units (for uSim controller/policy RNN)]
#If timepoints < args.timestep_limit, the neural perturbation will keep on repeating until the episode ends

neural_pert = np.ones((100, 256))

