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
from . import Utils

from .Utils import load_data, set_parameters

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

data_path = 'monkey/monkey_data_xycoord'

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
    def __init__(self, model_path, params_file_path, frame_skip):

        self.frame_skip= frame_skip
        self.model = mujoco_py.load_model_from_path(model_path)
        #optimize the model
        #Optimize the model parameters
        #with open(params_file_path, 'rb') as f:
        #    out= pickle.load(f)

        #out[0].set_values_to_model(self.model)

        self.sim= mujoco_py.MjSim(self.model)
        self.data= self.sim.data 

        #Load the kinematics (x and y coordinates from the data)
        x_coord_1319 = np.load(f'{data_path}/x_coord_1319.npy')
        y_coord_1319 = np.load(f'{data_path}/y_coord_1319.npy')

        x_coord_1106 = np.load(f'{data_path}/x_coord_1106.npy')
        y_coord_1106 = np.load(f'{data_path}/y_coord_1106.npy')

        x_coord_932 = np.load(f'{data_path}/x_coord_932.npy')
        y_coord_932 = np.load(f'{data_path}/y_coord_932.npy')

        x_coord_803 = np.load(f'{data_path}/x_coord_803.npy')
        y_coord_803 = np.load(f'{data_path}/y_coord_803.npy')

        x_coord_702 = np.load(f'{data_path}/x_coord_702.npy')
        y_coord_702 = np.load(f'{data_path}/y_coord_702.npy')

        x_coord_619 = np.load(f'{data_path}/x_coord_619.npy')
        y_coord_619 = np.load(f'{data_path}/y_coord_619.npy')

        #Meta parameters for the simulation
        self.n_fixedsteps= 5
        self.timestep_limit= (619 * 1) + self.n_fixedsteps
        # self._max_episode_steps= self.timestep_limit/ 2
        self._max_episode_steps= (619 * 1) + self.n_fixedsteps   #Do not matter. It is being set in the main.py where the total number of steps are being changed.
        self.radius= 0.038   #0.075
        self.theta= np.pi
        self.center= [0.06, 0.083]

        #The threshold is varied dynamically in the step and reset functions 
        self.threshold_user= 0.064   #Previously it was 0.1
        # self.threshold_user= 10.0 


        #Now change the x_coord and y_coord matrices to adjust for the self.radius and self.center
        d_radius = 1/self.radius
        x_coord_1319 = x_coord_1319/d_radius
        y_coord_1319 = y_coord_1319/d_radius 

        x_coord_1106 = x_coord_1106/d_radius
        y_coord_1106 = y_coord_1106/d_radius 

        x_coord_932 = x_coord_932/d_radius
        y_coord_932 = y_coord_932/d_radius 

        x_coord_803 = x_coord_803/d_radius
        y_coord_803 = y_coord_803/d_radius 

        x_coord_702 = x_coord_702/d_radius
        y_coord_702 = y_coord_702/d_radius 

        x_coord_619 = x_coord_619/d_radius
        y_coord_619 = y_coord_619/d_radius 

        self.x_coord_1319 = x_coord_1319 + self.center[0]
        self.y_coord_1319 = y_coord_1319 + self.center[1]

        self.x_coord_1106 = x_coord_1106 + self.center[0]
        self.y_coord_1106 = y_coord_1106 + self.center[1]

        self.x_coord_932 = x_coord_932 + self.center[0]
        self.y_coord_932 = y_coord_932 + self.center[1]

        self.x_coord_803 = x_coord_803 + self.center[0]
        self.y_coord_803 = y_coord_803 + self.center[1]

        self.x_coord_702 = x_coord_702 + self.center[0]
        self.y_coord_702 = y_coord_702 + self.center[1]

        self.x_coord_619 = x_coord_619 + self.center[0]
        self.y_coord_619 = y_coord_619 + self.center[1]

        self.x_coord = self.x_coord_619
        self.y_coord = self.y_coord_619

        self.viewer= None 
        self._viewers= {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        # self.init_qpos= self.sim.data.qpos.ravel().copy()
        # self.init_qvel= self.sim.data.qvel.ravel().copy()

        # self.init_qpos= np.load('./init_qpos.npy')
        # self.init_qvel= np.load('./init_qvel.npy')

        self.init_qpos= np.load('SAC/qpos_1.npy')
        self.init_qvel= np.load('SAC/qvel_1.npy')

        self._set_action_space()
        
        self._set_observation_space(self._get_obs())

        self.seed()

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

    def reset(self):
        self.istep= 0
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
            self.sim.step() 

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


class Arm_Env(MujocoEnv, utils.EzPickle):

    def __init__(self, model_path, params_file_path, frame_skip):
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, model_path, params_file_path, frame_skip)

    def get_cost(self, action):
        scaler= 1/50
        act= np.array(action)
        cost= scaler * np.sum(np.abs(act))
        return cost

    def get_reward(self):
        hand_pos= self.sim.data.get_body_xpos("hand").copy()
        target_pos= self.sim.data.get_body_xpos("target").copy()

        d_x= np.abs(hand_pos[0] - target_pos[0])
        d_y= np.abs(hand_pos[1] - target_pos[1])
        d_z= np.abs(hand_pos[2] - target_pos[2])

        #Check if the distance is greater than the distance threshold, which will terminate the environment.
        if d_x > self.threshold or d_y > self.threshold or d_z > self.threshold:
            return -5

        r_x= 1/(1000**d_x)
        r_y= 1/(1000**d_y)
        r_z= 1/(1000**d_z)

        reward= r_x + r_y + r_z

        return reward

    def is_done(self):
        #Define the distance threshold termination criteria
        target_position= self.sim.data.get_body_xpos("target").copy()
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

        if self.istep > self.n_fixedsteps and self.istep < 20:
            self.threshold = 0.032
        elif self.istep >= 20 and self.istep<30:
            self.threshold = 0.016
        elif self.istep >=30:
            self.threshold = 0.008

        prev_hand_xpos= self.sim.data.get_body_xpos("hand").copy()

        self.do_simulation(action, self.frame_skip)

        curr_hand_xpos= self.sim.data.get_body_xpos("hand").copy()
        prev_target_xpos = self.sim.data.get_body_xpos("target").copy()

        hand_vel= (curr_hand_xpos - prev_hand_xpos) / self.dt

        reward= self.get_reward()
        cost= self.get_cost(action)
        final_reward= (5*reward) - (0.5*cost)

        done= self.is_done()

        self.upd_theta()
        curr_target_xpos = self.sim.data.get_body_xpos("target").copy()
        target_vel = (curr_target_xpos - prev_target_xpos) / self.dt

        hand_vel = np.clip(hand_vel*10, -1.5, 1.5)
        target_vel = np.clip(target_vel*10, -1.5, 1.5)

        ob= self._get_obs()
        obser= [*ob, *hand_vel, *target_vel]

        return obser, final_reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos= self.init_qpos
        qvel= self.init_qvel

        self.set_state(qpos, qvel)

        return [*self._get_obs(), *[0, 0, 0], *[0,0,0]]

    def _get_obs(self):
        target_position= self.sim.data.get_body_xpos("target").copy()
        qposition= self.sim.data.qpos.flat.copy()
        qvelocity= self.sim.data.qvel.flat.copy()
        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        hand_position= self.sim.data.get_body_xpos("hand").copy()
        dist_hand_target= hand_position - target_position

        return np.concatenate((target_position,
        qposition,
        qvelocity,
        actuator_forces,
        hand_position,
        dist_hand_target))

    def upd_theta(self):
        if self.istep <= self._max_episode_steps:
            if self.istep <= self.n_fixedsteps:
                target_x = self.x_coord[0]
                target_y = self.y_coord[0]
            else:
                target_x = self.x_coord[int(((self.x_coord.shape[0]-1)/(self._max_episode_steps-self.n_fixedsteps)) * (self.istep - self.n_fixedsteps))]
                target_y = self.y_coord[int(((self.y_coord.shape[0]-1)/(self._max_episode_steps-self.n_fixedsteps)) * (self.istep - self.n_fixedsteps))]
        else:
            # target_x = self.x_coord[int(((self.x_coord.shape[0]-1)/(self._max_episode_steps)) * (self.istep % self._max_episode_steps))]
            # target_y = self.y_coord[int(((self.y_coord.shape[0]-1)/(self._max_episode_steps)) * (self.istep % self._max_episode_steps))]
            target_x = self.x_coord[int(((self.x_coord.shape[0]-1)/(self._max_episode_steps-self.n_fixedsteps)) * ((self.istep - self.n_fixedsteps) % (self._max_episode_steps - self.n_fixedsteps)))]
            target_y = self.y_coord[int(((self.y_coord.shape[0]-1)/(self._max_episode_steps-self.n_fixedsteps)) * ((self.istep - self.n_fixedsteps) % (self._max_episode_steps - self.n_fixedsteps)))]


        x_joint_i= self.model.get_joint_qpos_addr("box:x")
        y_joint_i= self.model.get_joint_qpos_addr("box:y")

        # target_x = (0.5*self.radius* np.cos(self.theta)) + self.center[0]
        # target_y = self.radius* -np.sin(self.theta) + self.center[1]

        crnt_state= self.sim.get_state()

        crnt_state.qpos[x_joint_i]= target_x 
        crnt_state.qpos[y_joint_i]= target_y

        self.set_state(crnt_state.qpos, crnt_state.qvel)