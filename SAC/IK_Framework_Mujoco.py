#Minimal implementation of the RL_Framework_Mujoco environment for CMA-ES and IK Optimization
#for finding the inital pose of the musculoskeletal model
#and for trajectory visualization

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import pickle

import numpy as np
from gym import utils
from . import sensory_feedback_specs
from . import kinematics_preprocessing_specs
from mujoco import viewer

import ipdb

import mujoco

DEFAULT_SIZE = 500

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """
    def __init__(self, model_path, condition_to_sim, cond_tpoint, args):

        self.model_path = model_path
        self.initial_pose_path = args.initial_pose_path
        self.kinematics_path = args.kinematics_path
        self.cond_to_sim = condition_to_sim
        self.cond_tpoint = cond_tpoint

         #setup the model
        self.model = mujoco.MjModel.from_xml_path(model_path)

        #setup the simulator and data object
        self.data = mujoco.MjData(self.model)

        #Save all the sensory feedback specs for use in the later functions
        self.sfs_proprioceptive_feedback = args.proprioceptive_feedback
        self.sfs_muscle_forces = args.muscle_forces
        self.sfs_joint_feedback = args.joint_feedback
        self.sfs_visual_feedback = args.visual_feedback
        self.sfs_visual_feedback_bodies = args.visual_feedback_bodies
        self.sfs_visual_distance_bodies = args.visual_distance_bodies
        self.sfs_visual_velocity = args.visual_velocity
        self.sfs_sensory_delay_timepoints = args.sensory_delay_timepoints

        #Load the kin_train
        # Load the experimental kinematics x and y coordinates from the data
        with open(self.kinematics_path + '/kinematics.pkl', 'rb') as f:
            kin_train_test = pickle.load(f)

        kin_train = kin_train_test['train'] #[num_conds][num_targets, num_coords, timepoints]

        #Randomly sample the number of targets from a condition = 0
        num_targets = kin_train[0].shape[0]

        #Find the qpos corresponding to the musculoskeletal model and targets
        self.qpos_idx_musculo = np.array(list(range(0, self.model.nq)))
        self.qpos_idx_targets = []
        for musculo_targets in kinematics_preprocessing_specs.musculo_target_joints:
            joint_id = self.model.joint(musculo_targets).qposadr
            self.qpos_idx_targets.append(joint_id)
        	
        #Delete the corresponding index from the qpos_idx_musculo
        self.qpos_idx_musculo = np.delete(self.qpos_idx_musculo, self.qpos_idx_targets).tolist()

        #Set the simulation timestep
        if args.sim_dt != 0:
            self.model.opt.timestep = args.sim_dt


        self.n_fixedsteps = args.n_fixedsteps
        self.radius = args.trajectory_scaling

        self.center = args.center

        #Kinematics preprocessing for training and testing kinematics
        #Preprocess training kinematics
        for i_target in range(kin_train[0].shape[0]):
            for i_cond in range(len(kin_train)):
                for i_coord in range(kin_train[i_cond].shape[1]):
                    kin_train[i_cond][i_target, i_coord, :] = kin_train[i_cond][i_target, i_coord, :] / self.radius[i_target]
                    kin_train[i_cond][i_target, i_coord, :] = kin_train[i_cond][i_target, i_coord, :] + self.center[i_target][i_coord]


        self.kin_train = kin_train
        self.kin_to_sim = self.kin_train

        self.current_cond_to_sim = self.cond_to_sim

        self.upd_theta(self.cond_tpoint)


        self.viewer = None 
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        #init_qpos and init_qvel do not contain target qpos
        #and contain only the musculo qpos
        #If the initial qpos and qvel are provided by the user
        if path.isfile(self.initial_pose_path + '/qpos.npy'):
            init_qpos = np.load(self.initial_pose_path + '/qpos.npy')
            init_qvel = np.load(self.initial_pose_path + '/qvel.npy')
        
        #else use the default initial pose of xml model
        else:
            init_qpos = self.data.qpos.flat.copy()[self.qpos_idx_musculo]
            init_qvel = self.data.qvel.flat.copy()[self.qpos_idx_musculo]

        #Get the qpos of musclo + targets
        musculo_qpos = self.data.qpos.flat.copy()
        musculo_qvel = self.data.qvel.flat.copy()

        #Set the musculo part to the saved initial qpos and qvel
        musculo_qpos[self.qpos_idx_musculo] = init_qpos

        self.init_qpos = musculo_qpos

        #Set the initial state to init_qpos
        #Set the state to the initial pose
        self.set_state(self.init_qpos)

        if self.sfs_visual_feedback == True:

            #Save the xpos of the musculo bodies for visual vels
            if len(self.sfs_visual_velocity) != 0:
                self.prev_body_xpos = []
                for musculo_body in self.sfs_visual_velocity:
                    body_xpos = self.data.xpos[self.model.body(musculo_body).id]
                    self.prev_body_xpos = [*self.prev_body_xpos, *body_xpos]

    #Return the qpos of only the musculoskeletal bodies and not the targets
    def get_musculo_state(self):
    	qpos_all = self.data.qpos.flat.copy()
    	qpos_musculo = qpos_all[self.qpos_idx_musculo]

    	return qpos_musculo
   

    def set_state(self, qpos):
        assert qpos.shape == (self.model.nq, )

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qpos * 0
        mujoco.mj_forward(self.model, self.data)

    def set_state_musculo(self, qpos):
        qpos_all = self.data.qpos.flat.copy()
        qpos_all[self.qpos_idx_musculo] = qpos

        self.data.qpos[:] = qpos_all
        self.data.qvel[:] = qpos_all * 0
        mujoco.mj_forward(self.model, self.data)


    def get_obs_musculo_bodies(self):
    	#Returns the current xyz coords of the musculo bodies to be tracked
    	musculo_body_state = []
    	for musculo_body in kinematics_preprocessing_specs.musculo_tracking:
    		current_xyz_coord = self.data.xpos[self.model.body(musculo_body[0]).id].flat.copy()
    		musculo_body_state.append(current_xyz_coord)

    	return np.array(musculo_body_state)  #[n_musculo_bodies, 3]


    def get_obs_targets(self):
    	#Returns the current xyz coords of the targets to be tracked
    	musculo_target_state = []
    	for musculo_body in kinematics_preprocessing_specs.musculo_tracking:
    		current_xyz_coord = self.data.xpos[self.model.body(musculo_body[1]).id].flat.copy()
    		musculo_target_state.append(current_xyz_coord)

    	return np.array(musculo_target_state)  #[n_musculo_targets, 3]

    def set_cond_to_simulate(self, i_condition, cond_timepoint):

        #update the state variables
        self.cond_to_sim = i_condition
        self.cond_tpoint = cond_timepoint

        self.current_cond_to_sim = self.cond_to_sim

        self.upd_theta(self.cond_tpoint)


    @property
    def dt(self):
        return self.model.opt.timestep 

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=0,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).sync()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = viewer.launch_passive(self.model, self.data)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name).copy()


class Muscle_Env(MujocoEnv):

    def __init__(self, model_path, condition_to_sim, cond_tpoint, args):
        MujocoEnv.__init__(self, model_path, condition_to_sim, cond_tpoint, args)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def upd_theta(self, cond_timepoint):

        coords_to_sim = self.kin_to_sim[self.current_cond_to_sim] #[num_targets, num_coords, timepoints]

        assert cond_timepoint < coords_to_sim.shape[-1]

        crnt_qpos = self.data.qpos.copy()

        for i_target in range(self.kin_to_sim[self.current_cond_to_sim].shape[0]):
            if kinematics_preprocessing_specs.xyz_target[i_target][0]:
                x_joint_idx= self.model.joint(f"box:x{i_target}").qposadr
                crnt_qpos[x_joint_idx] = coords_to_sim[i_target, 0, cond_timepoint]


            if kinematics_preprocessing_specs.xyz_target[i_target][1]:
                y_joint_idx= self.model.joint(f"box:y{i_target}").qposadr
                crnt_qpos[y_joint_idx] = coords_to_sim[i_target, kinematics_preprocessing_specs.xyz_target[i_target][0], cond_timepoint]

            if kinematics_preprocessing_specs.xyz_target[i_target][2]:
                z_joint_idx= self.model.joint(f"box:z{i_target}").qposadr
                crnt_qpos[z_joint_idx] = coords_to_sim[i_target, kinematics_preprocessing_specs.xyz_target[i_target][0] + kinematics_preprocessing_specs.xyz_target[i_target][1], cond_timepoint]


        #Now set the state
        self.set_state(crnt_qpos)


    def _get_obs(self):
        
        sensory_feedback = []
        if self.sfs_proprioceptive_feedback == True:
            muscle_lens = self.data.actuator_length.flat.copy()
            muscle_vels = self.data.actuator_velocity.flat.copy()

            #process through the given function for muscle lens and muscle vels
            muscle_lens, muscle_vels = sensory_feedback_specs.process_proprioceptive(muscle_lens, muscle_vels)
            sensory_feedback = [*sensory_feedback, *muscle_lens, *muscle_vels]


        if self.sfs_muscle_forces == True:
            actuator_forces = self.data.qfrc_actuator.flat.copy()

            #process
            actuator_forces = sensory_feedback_specs.process_muscle_forces(actuator_forces)
            sensory_feedback = [*sensory_feedback, *actuator_forces]


        if self.sfs_joint_feedback == True:
            sensory_qpos = self.data.qpos.flat.copy()
            sensory_qvel = self.data.qvel.flat.copy()

            sensory_qpos, sensory_qvel = sensory_feedback_specs.process_joint_feedback(sensory_qpos, sensory_qvel)
            sensory_feedback = [*sensory_feedback, *sensory_qpos, *sensory_qvel]


        if self.sfs_visual_feedback == True:
            
            #Check if the user specified the musculo bodies to be included
            assert len(self.sfs_visual_feedback_bodies) != 0

            visual_xyz_coords = []
            for musculo_body in self.sfs_visual_feedback_bodies:
                visual_xyz_coords = [*visual_xyz_coords, *self.data.get_body_xpos(musculo_body)]

            visual_xyz_coords = sensory_feedback_specs.process_visual_position(visual_xyz_coords)
            sensory_feedback = [*sensory_feedback, *visual_xyz_coords]

        if len(self.sfs_visual_distance_bodies) != 0:
            visual_xyz_distance = []
            for musculo_tuple in self.sfs_visual_distance_bodies:
                body0_xyz = self.data.xpos[self.model.body(musculo_tuple[0]).id]
                body1_xyz = self.data.xpos[self.model.body(musculo_tuple[1]).id]
                tuple_dist = np.abs(body0_xyz - body1_xyz).tolist()
                visual_xyz_distance = [*visual_xyz_distance, *tuple_dist]

            #process
            visual_xyz_distance = sensory_feedback_specs.process_visual_distance(visual_xyz_distance)
            sensory_feedback = [*sensory_feedback, *visual_xyz_distance]

        #Save the xpos of the musculo bodies for visual vels
        if len(self.sfs_visual_velocity) != 0:

            #Find the visual vels after the simulation
            current_body_xpos = []
            for musculo_body in self.sfs_visual_velocity:
                body_xpos = self.data.xpos[self.model.body(musculo_body).id]
                current_body_xpos = [*current_body_xpos, *body_xpos]

            #Find the velocity
            visual_vels = (np.abs(np.array(self.prev_body_xpos) - np.array(current_body_xpos)) / self.dt).tolist()
            
            #process visual velocity feedback
            visual_vels = sensory_feedback_specs.process_visual_velocity(visual_vels)
            sensory_feedback= [*sensory_feedback, *visual_vels]

            #update the self.prev_body_xpos
            self.prev_body_xpos = current_body_xpos

        return np.array(sensory_feedback)