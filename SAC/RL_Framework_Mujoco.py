from collections import OrderedDict
import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import pickle

import numpy as np
from gym import utils
from . import sensory_feedback_specs, reward_function_specs
from . import kinematics_preprocessing_specs

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

import ipdb

DEFAULT_SIZE = 500

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """
    def __init__(self, model_path, initial_pose_path, kinematics_path, nusim_data_path, mode_to_sim, frame_skip):


        self.model_path = model_path
        self.initial_pose_path = initial_pose_path
        self.kinematics_path = kinematics_path
        self.nusim_data_path = nusim_data_path

        self.mode_to_sim = mode_to_sim
        self.frame_skip = frame_skip
        self.frame_repeat = kinematics_preprocessing_specs.frame_repeat
        self.model = mujoco_py.load_model_from_path(model_path)

        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

        #Set the simulation timestep
        if kinematics_preprocessing_specs.sim_dt != 0:
            self.model.opt.timestep = kinematics_preprocessing_specs.sim_dt

        # Load the experimental kinematics x and y coordinates from the data
        with open(self.kinematics_path + '/kinematics.pkl', 'rb') as f:
            kin_train_test = pickle.load(f)

        kin_train = kin_train_test['train'] #[num_conds][num_targets, num_coords, timepoints]
        kin_test = kin_train_test['test'] #[num_conds][num_targets, num_coords, timepoints]


        #Load the neural activities
        #[timepoints, n_neurons= 49]

        with open(self.nusim_data_path + '/neural_activity_train.pkl', 'rb') as f:
            na_train = pickle.load(f)
    
        with open(self.nusim_data_path + '/neural_activity_test.pkl', 'rb') as f:
            na_test = pickle.load(f)


        neural_activity_cum = []
        
        for i_condition in range(len(na_train)):

            #Now normalize the neural activity and append it
            na_c = na_train[i_condition] / np.max(na_train[i_condition])
            neural_activity_cum.append(na_c)

        self.n_fixedsteps = kinematics_preprocessing_specs.n_fixedsteps
        self.timestep_limit = kinematics_preprocessing_specs.timestep_limit
        self.radius = kinematics_preprocessing_specs.radius
        self.center = kinematics_preprocessing_specs.center

        #The threshold is varied dynamically in the step and reset functions 
        self.threshold_user = 0.064   #Previously it was 0.1
        
        #Setup coord_idx for setting the neural activity loss during nusim training
        self.coord_idx=0
        self.neural_activity_cum = neural_activity_cum


        #Kinematics preprocessing for training and testing kinematics
        #Preprocess training kinematics
        for i_target in range(kin_train[0].shape[0]):
            for i_cond in range(len(kin_train)):
                for i_coord in range(kin_train[i_cond].shape[1]):
                    kin_train[i_cond][i_target, i_coord, :] = kin_train[i_cond][i_target, i_coord, :] / self.radius[i_target]
                    kin_train[i_cond][i_target, i_coord, :] = kin_train[i_cond][i_target, i_coord, :] + self.center[i_target][i_coord]

        #Preprocess testing kinematics
        for i_target in range(kin_test[0].shape[0]):
            for i_cond in range(len(kin_test)):
                for i_coord in range(kin_test[i_cond].shape[1]):

                    kin_test[i_cond][i_target, i_coord, :] = kin_test[i_cond][i_target, i_coord, :] / self.radius[i_target]
                    kin_test[i_cond][i_target, i_coord, :] = kin_test[i_cond][i_target, i_coord, :] + self.center[i_target][i_coord]

        self.kin_train = kin_train 
        self.kin_test = kin_test 

        self.kin_to_sim = self.kin_train
        self.n_exp_conds = len(self.kin_to_sim)
        self.current_cond_to_sim = 0

        self.viewer = None 
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = np.load(initial_pose_path + '/initial_qpos_opt.npy')
        #Start the musculo model with zero initial qvels
        self.init_qvel = np.load(initial_pose_path + '/initial_qpos_opt.npy')*0

        self._set_action_space()
        
        self._set_observation_space(self._get_obs())

        self.seed()


    def update_kinematics_for_test(self):

        with open(self.nusim_data_path + '/neural_activity_train.pkl', 'rb') as f:
            na_train = pickle.load(f)
    
        with open(self.nusim_data_path + '/neural_activity_test.pkl', 'rb') as f:
            na_test = pickle.load(f)


        neural_activity_cum = []

        #First append the training conditions
        for i_condition in range(len(self.kin_train)):

            #Normalize the neural activity and append it
            na_c = na_train[i_condition] / np.max(na_train[i_condition])
            neural_activity_cum.append(na_c)

        #Then append the testing conditions
        for i_condition in range(len(self.kin_test)):

            #Normalize the neural activity and append it
            na_c = na_test[i_condition] / np.max(na_test[i_condition])
            neural_activity_cum.append(na_c)


        self.neural_activity_cum = neural_activity_cum

        #Simulate the environment on both the training and testing kinematics
        #First update the keys of self.kin_test
        for cond in range(len(self.kin_test)):
            self.kin_test[len(self.kin_train) + cond] = self.kin_test.pop(cond)
        
        #Update the kinematics to simulate
        self.kin_to_sim.update(self.kin_test)

        #Update the number of experimental conditions
        self.n_exp_conds = len(self.kin_to_sim)


    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self, cond_to_select):

        #Set the experimental condition for training
        
        self.current_cond_to_sim = cond_to_select
        self.neural_activity = self.neural_activity_cum[cond_to_select]

        #Set the high-level task scalar signal
        self.condition_scalar = (self.kin_to_sim[self.current_cond_to_sim].shape[-1] - 600) / (1319 - 600)
        #Set the max episode steps to reset after one cycle for multiple cycles
        self._max_episode_steps = self.kin_to_sim[self.current_cond_to_sim].shape[-1] + self.n_fixedsteps

        self.istep= 0
        self.coord_idx = 0
        self.theta= np.pi
        self.threshold= self.threshold_user
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq, ) and qvel.shape == (self.model.nv, )
        old_state= self.sim.get_state()

        new_state= mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                        old_state.act, old_state.udd_state)

        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip 

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:]= ctrl 
        for _ in range(n_frames):
            self.sim.data.ctrl[:]= ctrl
            self.sim.step()
            self.sim.forward()

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
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name).copy()

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])


class Muscle_Env(MujocoEnv):

    def __init__(self, model_path, initial_pose_path, kinematics_path, nusim_data_path, mode_to_sim, frame_skip):
        MujocoEnv.__init__(self, model_path, initial_pose_path, kinematics_path, nusim_data_path, mode_to_sim, frame_skip)

    def get_cost(self, action):
        scaler= 1/50
        act= np.array(action)
        cost= scaler * np.sum(np.abs(act))
        return cost

    def is_done(self):
        #Define the distance threshold termination criteria
        target_position= self.sim.data.get_body_xpos("target0").copy()
        hand_position= self.sim.data.get_body_xpos("hand").copy()
        
        criteria= hand_position - target_position

        if self.istep < self.timestep_limit:
            if np.abs(criteria[0]) > self.threshold or np.abs(criteria[1]) > self.threshold or np.abs(criteria[2]) > self.threshold:
                return True
            else:
                return False
        else:
            return True

    def step(self, action):
        self.istep += 1

        if self.istep > self.n_fixedsteps and self.istep < 100:
            self.threshold = 0.032
        elif self.istep >= 100 and self.istep<150:
            self.threshold = 0.016
        elif self.istep >=150:
            self.threshold = 0.008

        #Save the xpos of the musculo bodies for visual vels
        if len(sensory_feedback_specs.visual_velocity) != 0:
            prev_body_xpos = []
            for musculo_body in sensory_feedback_specs.visual_velocity:
                body_xpos = self.sim.data.get_body_xpos(musculo_body)
                prev_body_xpos = [*prev_body_xpos, *body_xpos]

        #Now carry out one step of the MuJoCo simulation
        self.do_simulation(action, self.frame_skip)

        #Currently the reward function is the function of the delayed state, current simulator state, action and threshold
        if sensory_feedback_specs.sensory_delay_timepoints != 0:
            reward= reward_function_specs.reward_function(self.state_to_return[-1], self.sim, action, self.threshold)
        else:
            #Pass a dummy variable for the delayed state feedback
            reward= reward_function_specs.reward_function(0, self.sim, action, self.threshold)
            
        cost= self.get_cost(action)
        final_reward= (5*reward) #- (0.5*cost)

        done= self.is_done()

        self.upd_theta()

        visual_vels = []
        #Find the visual vels after the simulation
        if len(sensory_feedback_specs.visual_velocity) != 0:
            current_body_xpos = []
            for musculo_body in sensory_feedback_specs.visual_velocity:
                body_xpos = self.sim.data.get_body_xpos(musculo_body)
                current_body_xpos = [*current_body_xpos, *body_xpos]

            #Find the velocity
            visual_vels = (np.abs(np.array(prev_body_xpos) - np.array(current_body_xpos)) / self.dt).tolist()

        ob= self._get_obs()
        
        #process visual velocity feedback
        if self.mode_to_sim in ["sensory_pert"]:
            visual_vels = sensory_feedback_specs.process_visual_velocity_pert(visual_vels)

        visual_vels = sensory_feedback_specs.process_visual_velocity(visual_vels)

        if self.mode_to_sim in ["SFE"] and "visual_velocity" in sensory_feedback_specs.sf_elim:
            obser= [*ob, *[ele*0 for ele in visual_vels]]
        else:
            obser= [*ob, *visual_vels]

        #Append the current observation to the start of the list
        #Return the last observation later on
        self.state_to_return.insert(0, obser)


        return self.state_to_return.pop(), final_reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):

        #Set the state to the initial pose
        self.set_state(self.init_qpos, self.init_qvel)

        #Now get the observation of the initial state and append zeros corresponding to the velocity of musculo bodies 
        #as specified in sensory_feedback_specs (len*3 for x/y/z vel for each musculo body)
        initial_state_obs = [*self._get_obs(), *np.zeros(len(sensory_feedback_specs.visual_velocity)*3)]

        #Maintain a list of state observations for implementing the state delay
        self.state_to_return = [[0]*len(initial_state_obs)] * sensory_feedback_specs.sensory_delay_timepoints
        #Insert the inital state obs to the start of the list
        self.state_to_return.insert(0, initial_state_obs)

        #Return the last element of the state_to_return 
        return self.state_to_return.pop()

    def _get_obs(self):
        
        sensory_feedback = []
        if sensory_feedback_specs.proprioceptive_feedback == True:
            muscle_lens = self.sim.data.actuator_length.flat.copy()
            muscle_vels = self.sim.data.actuator_velocity.flat.copy()

            #process through the given function for muscle lens and muscle vels
            if self.mode_to_sim in ["sensory_pert"]:
                muscle_lens, muscle_vels = sensory_feedback_specs.process_proprioceptive_pert(muscle_lens, muscle_vels)

            muscle_lens, muscle_vels = sensory_feedback_specs.process_proprioceptive(muscle_lens, muscle_vels)

            if self.mode_to_sim in ["SFE"] and "proprioceptive" in sensory_feedback_specs.sf_elim:
                sensory_feedback = [*sensory_feedback, *[ele*0 for ele in muscle_lens], *[ele*0 for ele in muscle_vels]]
            else:
                sensory_feedback = [*sensory_feedback, *muscle_lens, *muscle_vels]


        if sensory_feedback_specs.muscle_forces == True:
            actuator_forces = self.sim.data.qfrc_actuator.flat.copy()

            #process
            if self.mode_to_sim in ["sensory_pert"]:
                actuator_forces = sensory_feedback_specs.process_muscle_forces_pert(actuator_forces)

            actuator_forces = sensory_feedback_specs.process_muscle_forces(actuator_forces)
            
            if self.mode_to_sim in ["SFE"] and "muscle_forces" in sensory_feedback_specs.sf_elim:
                sensory_feedback = [*sensory_feedback, *[ele*0 for ele in actuator_forces]]
            else:
                sensory_feedback = [*sensory_feedback, *actuator_forces]


        if sensory_feedback_specs.joint_feedback == True:
            sensory_qpos = self.sim.data.qpos.flat.copy()
            sensory_qvel = self.sim.data.qvel.flat.copy()

            #process
            if self.mode_to_sim in ["sensory_pert"]:
                sensory_qpos, sensory_qvel = sensory_feedback_specs.process_joint_feedback_pert(sensory_qpos, sensory_qvel)

            sensory_qpos, sensory_qvel = sensory_feedback_specs.process_joint_feedback(sensory_qpos, sensory_qvel)
            
            if self.mode_to_sim in ["SFE"] and "joint_feedback" in sensory_feedback_specs.sf_elim:
                sensory_feedback = [*sensory_feedback, *[ele*0 for ele in sensory_qpos], *[ele*0 for ele in sensory_qvel]]
            else:
                sensory_feedback = [*sensory_feedback, *sensory_qpos, *sensory_qvel]


        if sensory_feedback_specs.visual_feedback == True:
            
            #Check if the user specified the musculo bodies to be included
            assert len(sensory_feedback_specs.visual_feedback_bodies) != 0

            visual_xyz_coords = []
            for musculo_body in sensory_feedback_specs.visual_feedback_bodies:
                visual_xyz_coords = [*visual_xyz_coords, *self.sim.data.get_body_xpos(musculo_body)]

            if self.mode_to_sim in ["sensory_pert"]:
                visual_xyz_coords = sensory_feedback_specs.process_visual_position_pert(visual_xyz_coords)

            visual_xyz_coords = sensory_feedback_specs.process_visual_position(visual_xyz_coords)
            
            if self.mode_to_sim in ["SFE"] and "visual_position" in sensory_feedback_specs.sf_elim:
                sensory_feedback = [*sensory_feedback, *[ele*0 for ele in visual_xyz_coords]]
            else:
                sensory_feedback = [*sensory_feedback, *visual_xyz_coords]

        if len(sensory_feedback_specs.visual_distance_bodies) != 0:
            visual_xyz_distance = []
            for musculo_tuple in sensory_feedback_specs.visual_distance_bodies:
                body0_xyz = self.sim.data.get_body_xpos(musculo_tuple[0])
                body1_xyz = self.sim.data.get_body_xpos(musculo_tuple[1])
                tuple_dist = (body0_xyz - body1_xyz).tolist()
                visual_xyz_distance = [*visual_xyz_distance, *tuple_dist]

            #process
            if self.mode_to_sim in ["sensory_pert"]:
                visual_xyz_distance = sensory_feedback_specs.process_visual_distance_pert(visual_xyz_distance)

            visual_xyz_distance = sensory_feedback_specs.process_visual_distance(visual_xyz_distance)

            if self.mode_to_sim in ["SFE"] and "visual_distance" in sensory_feedback_specs.sf_elim:
                sensory_feedback = [*sensory_feedback, *[ele*0 for ele in visual_xyz_distance]]
            else:
                sensory_feedback = [*sensory_feedback, *visual_xyz_distance]

        return np.array(sensory_feedback)

    def upd_theta(self):
        if self.istep <= self._max_episode_steps:
            if self.istep <= self.n_fixedsteps:
                self.tpoint_to_sim = 0
            else:
                self.tpoint_to_sim = int(((self.kin_to_sim[self.current_cond_to_sim].shape[-1]-1)/(self._max_episode_steps-self.n_fixedsteps)) * (self.istep - self.n_fixedsteps))
                
        else:
            self.tpoint_to_sim = int(((self.kin_to_sim[self.current_cond_to_sim].shape[-1]-1)/(self._max_episode_steps-self.n_fixedsteps)) * ((self.istep - self.n_fixedsteps) % (self._max_episode_steps - self.n_fixedsteps)))

        self.coord_idx = self.tpoint_to_sim

        coords_to_sim = self.kin_to_sim[self.current_cond_to_sim]
        
        crnt_state = self.sim.get_state()

        for i_target in range(self.kin_to_sim[self.current_cond_to_sim].shape[0]):
            if kinematics_preprocessing_specs.xyz_target[i_target][0]:
                x_joint_idx= self.model.get_joint_qpos_addr(f"box:x{i_target}")
                crnt_state.qpos[x_joint_idx] = coords_to_sim[i_target, 0, self.tpoint_to_sim]


            if kinematics_preprocessing_specs.xyz_target[i_target][1]:
                y_joint_idx= self.model.get_joint_qpos_addr(f"box:y{i_target}")
                crnt_state.qpos[y_joint_idx] = coords_to_sim[i_target, kinematics_preprocessing_specs.xyz_target[i_target][0], self.tpoint_to_sim]

            if kinematics_preprocessing_specs.xyz_target[i_target][2]:
                z_joint_idx= self.model.get_joint_qpos_addr(f"box:z{i_target}")
                crnt_state.qpos[z_joint_idx] = coords_to_sim[i_target, kinematics_preprocessing_specs.xyz_target[i_target][0] + kinematics_preprocessing_specs.xyz_target[i_target][1], self.tpoint_to_sim]


        #Now set the state
        self.set_state(crnt_state.qpos, crnt_state.qvel)
