import datetime
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import itertools
import model_utils as model_utils
import pybullet as p
import pybullet_data
import yaml
import scipy.io
from pybullet_env import PyBulletEnv
import torch.nn.functional as F

import farms_pylog as pylog
try:
    from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
except ImportError:
    pylog.warning("farms-muscle not installed!")
from farms_container import Container

sphere_file = "../files/sphere_small.urdf"

class Mouse_Env(PyBulletEnv):

    def __init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset, vizualize, threshold, cost_scale, max_cycle_len):
        PyBulletEnv.__init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset, vizualize, threshold, cost_scale)

        u = self.container.muscles.activations
        self.max_cycle_len = max_cycle_len
        self.muscle_params = {}
        self.muscle_excitation = {}

        #####TARGET POSITION USING POINT IN SPACE: X, Y, Z#####
        ###x, y, z for initializing from hand starting position, target_pos for updating
        self.x_pos = [0]
        self.y_pos = p.getLinkState(self.model, 115)[0][1]
        self.z_pos = p.getLinkState(self.model, 115)[0][2]

        self.avg_vel = 1
        self.target_pos = [self.x_pos[0]/self.scale-self.offset, self.y_pos, self.z_pos]

        for muscle in self.muscles.muscles.keys():
               self.muscle_params[muscle] = u.get_parameter('stim_{}'.format(muscle))
               self.muscle_excitation[muscle] = p.addUserDebugParameter("flexor {}".format(muscle), 0, 1, 0.00)
               self.muscle_params[muscle].value = 0

    def reset(self, pose_file):

        self.istep = 0
        self.activations = []
        self.hand_positions = []

        model_utils.disable_control(self.model) #disables torque/position

        self.reset_model(pose_file) #resets model position
        self.container.initialize() #resets container
        self.muscles.setup_integrator() #resets muscles

        #resets target position
        self.target_pos = np.array([self.x_pos[0]/self.scale-self.offset, self.y_pos, self.z_pos])
        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        
    def reset_model(self, pose_file): 
        model_utils.reset_model_position(self.model, pose_file)

    def get_reward(self): 
        hand_pos = p.getLinkState(self.model, 115, computeForwardKinematics=True)[0] #(x, y, z)

        d_x = np.abs(hand_pos[0] - self.target_pos[0])
        d_y = np.abs(hand_pos[1] - self.target_pos[1])
        d_z = np.abs(hand_pos[2] - self.target_pos[2])

        distances = [d_x, d_y, d_z]

        if d_x > self.threshold_x or d_y > self.threshold_y or d_z > self.threshold_z:
            reward = -5
        else:
            r_x= 1/(1000**d_x)
            r_y= 1/(1000**d_y)
            r_z= 1/(1000**d_z)

            reward= r_x + r_y + r_z

        return reward, distances

    def is_done(self):
        hand_pos =  np.array(p.getLinkState(self.model, 115, computeForwardKinematics=True)[0]) #(x, y, z)
        criteria = hand_pos - self.target_pos

        if self.istep >= self._max_episode_steps:
            return True

        if np.abs(criteria[0]) > self.threshold_x or np.abs(criteria[1]) > self.threshold_y or np.abs(criteria[2]) > self.threshold_z:
            return True
        
        return False

    def update_target_pos(self):
        self.target_pos = np.array([self.x_pos[(self.istep)]/self.scale-self.offset, self.y_pos, self.z_pos])

        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        
    def get_joint_positions_and_velocities(self):
        joint_positions = []
        joint_velocities = []
        for i in range(len(self.ctrl)):
            joint_positions.append(p.getJointState(self.model, self.ctrl[i])[0])
            joint_velocities.append(p.getJointState(self.model, self.ctrl[i])[1]*.01)
        joint_positions = [*list(np.array(joint_positions)), *list(p.getLinkState(self.model, 115, computeForwardKinematics=True)[0])] #(x, y, z)
        joint_velocities = [*list(np.array(joint_velocities)), *list(p.getLinkState(self.model, 115, computeForwardKinematics=True, computeLinkVelocity=True)[6])]
        return joint_positions, joint_velocities

    def get_start_state(self):
        joint_positions, _ = self.get_joint_positions_and_velocities()
        _, distance = self.get_reward()
        #targ_vel_const = self.comp_targ_vel_const()
        return [*list(np.array(self.get_activations())), *list(np.array(joint_positions)), *[0.]*10, *list(np.array(self.target_pos)), 0., *list(np.array(distance))]

    def update_state(self, act, joint_positions, joint_velocities, target_velocity, distances):
        state = [*list(np.array(act)), *list(np.array(joint_positions)), *list(np.array(joint_velocities)), *list(np.array(self.target_pos)), target_velocity, *list(np.array(distances))]
        return state
    
    def comp_targ_vel_const(self, scaling=.25):
        return np.sinh(self.avg_vel*scaling)/np.cosh(self.avg_vel*scaling)
    
    def comp_targ_vel(self, prev_target):
        return (self.target_pos - prev_target) / .001
    
    def step(self, forces, timestep):

        if timestep < (self._max_episode_steps-1):
            self.istep += 1

        prev_target = self.target_pos
        self.update_target_pos()
        
        self.controller_to_actuator(forces)

        self.do_simulation()

        act = self.get_activations()

        reward, distances = self.get_reward()
        cost = self.get_cost(forces)
        final_reward= 5*reward - (self.forces_scale*cost) 

        done = self.is_done()
        target_vel = self.comp_targ_vel(prev_target)
        joint_positions, joint_velocities = self.get_joint_positions_and_velocities()
        #target_vel_const = self.comp_targ_vel_const()
        state = self.update_state(act, joint_positions, joint_velocities, target_vel[0], distances)

        return state, final_reward, done


########################## SIMULATED ENVIRONMENT #########################################


class Mouse_Env_Simulated(PyBulletEnv):

    def __init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset, vizualize, threshold, cost_scale):
        PyBulletEnv.__init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset, vizualize, threshold, cost_scale)

        u = self.container.muscles.activations
        self.muscle_params = {}
        self.muscle_excitation = {}

        self.z_offset = 6
        self.x_offset = 20
        self.start_interval = -np.pi / 2
        self.end_interval =  3 * np.pi / 2

        #####TARGET POSITION USING POINT IN SPACE: X, Y, Z#####
        ###x, y, z for initializing from hand starting position, target_pos for updating
        self.x_theta = np.linspace(self.start_interval, self.end_interval, self.timestep)
        self.x_pos = np.sin(self.x_theta[0])
        self.y_pos = p.getLinkState(self.model, 115)[0][1]
        self.z_theta = np.linspace(self.start_interval, self.end_interval, self.timestep)
        self.z_pos = np.cos(self.z_theta[0])

        self.target_pos = [(self.x_pos + self.x_offset) / self.scale, self.y_pos, (self.z_pos + self.z_offset) / self.scale]

        for muscle in self.muscles.muscles.keys():
               self.muscle_params[muscle] = u.get_parameter('stim_{}'.format(muscle))
               self.muscle_excitation[muscle] = p.addUserDebugParameter("flexor {}".format(muscle), 0, 1, 0.00)
               self.muscle_params[muscle].value = 0

    def reset(self, pose_file):

        self.istep = 0
        model_utils.disable_control(self.model) #disables torque/position
        self.reset_model(pose_file) #resets model position
        self.container.initialize() #resets container
        self.muscles.setup_integrator() #resets muscles
        #resets target position
        self.x_theta = np.linspace(self.start_interval, self.end_interval, self.timestep)
        self.x_pos = np.sin(self.x_theta[0])

        self.z_theta = np.linspace(self.start_interval, self.end_interval, self.timestep)
        self.z_pos = np.cos(self.z_theta[0])

        self.target_pos = [(self.x_pos + self.x_offset) / self.scale, self.y_pos, (self.z_pos + self.z_offset) / self.scale]

        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        
    def reset_model(self, pose_file): 
        model_utils.reset_model_position(self.model, pose_file)

    def get_reward(self): 
        hand_pos = p.getLinkState(self.model, 115, computeForwardKinematics=True)[0] #(x, y, z)

        d_x = np.abs(hand_pos[0] - self.target_pos[0])
        d_y = np.abs(hand_pos[1] - self.target_pos[1])
        d_z = np.abs(hand_pos[2] - self.target_pos[2])

        distances = [d_x, d_y, d_z]

        if d_x > self.threshold_x or d_y > self.threshold_y or d_z > self.threshold_z:
            reward = -5
        else:
            r_x= 1/(1000**d_x)
            r_y= 1/(1000**d_y)
            r_z= 1/(1000**d_z)

            reward= r_x + r_y + r_z

        return reward, distances

    def is_done(self):
        hand_pos =  np.array(p.getLinkState(self.model, 115, computeForwardKinematics=True)[0]) #(x, y, z)
        criteria = hand_pos - self.target_pos

        if self.istep >= self._max_episode_steps:
            return True

        if np.abs(criteria[0]) > self.threshold_x or np.abs(criteria[1]) > self.threshold_y or np.abs(criteria[2]) > self.threshold_z:
            return True
        
        return False

    def update_target_pos(self):
        self.target_pos = [(np.sin(self.x_theta[self.istep]) + self.x_offset) / self.scale, self.y_pos, (np.cos(self.z_theta[self.istep]) + self.z_offset) / self.scale]

        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))

    def get_joint_positions_and_velocities(self):
        joint_positions = []
        joint_velocities = []
        for i in range(len(self.ctrl)):
            joint_positions.append(p.getJointState(self.model, self.ctrl[i])[0])
            joint_velocities.append(p.getJointState(self.model, self.ctrl[i])[1]/100)
        joint_positions = [*list(np.array(joint_positions)), *list(p.getLinkState(self.model, 115, computeForwardKinematics=True)[0])] #(x, y, z)
        joint_velocities = [*list(np.array(joint_velocities)), *list(p.getLinkState(self.model, 115, computeForwardKinematics=True, computeLinkVelocity=True)[6])]
        return joint_positions, joint_velocities
        
    def update_state(self, act, joint_positions, joint_velocities, target_pos, target_velocity, distances):
        state = [*list(np.array(act)), *list(np.array(joint_positions)), *list(np.array(joint_velocities)), *list(np.array(target_pos)), *list(np.array(target_velocity)), *list(np.array(distances))]
        return state

    def get_cur_state(self):

        joint_positions, _ = self.get_joint_positions_and_velocities()
        _, distance = self.get_reward()
        return [*list(np.array(self.get_activations())), *list(np.array(joint_positions)), *[0.]*10, *list(np.array(self.target_pos)), *[0.]*3, *list(np.array(distance))]
    
    def step(self, forces, timestep):

        if timestep < (self._max_episode_steps-1):
            self.istep += 1

        prev_target = self.target_pos
        self.update_target_pos()
        curr_target = self.target_pos
        
        self.controller_to_actuator(forces)

        self.do_simulation()

        act = self.get_activations()

        reward, distances = self.get_reward()
        cost = self.get_cost(forces)
        final_reward= 5*reward - (self.forces_scale*cost) 

        done = self.is_done()
        target_vel = (curr_target - prev_target) / .001
        joint_positions, joint_velocities = self.get_joint_positions_and_velocities()
        state = self.update_state(act, joint_positions, joint_velocities, curr_target, target_vel[0], distances)

        return state, final_reward, done