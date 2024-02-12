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

import farms_pylog as pylog
try:
    from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
except ImportError:
    pylog.warning("farms-muscle not installed!")
from farms_container import Container

sphere_file = "../files/sphere_small.urdf"

class PyBulletEnv(gym.Env):
    def __init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset, vizualize, threshold, cost_scale):
        #####BUILDS SERVER AND LOADS MODEL#####
        if(vizualize):
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81) #normal gravity
        self.plane = p.loadURDF("plane.urdf") #sets floor
        self.model = p.loadSDF(model_path)[0] #resizes, loads model, returns model id
        self.model_offset = model_offset
        p.resetBasePositionAndOrientation(self.model, self.model_offset, p.getQuaternionFromEuler([0, 0, 80.2])) #resets model position
        self.use_sphere = False

        # This might need to change for simulated
        # This is for the target positon only
        self.scale = 21 
        self.offset = -0.713 

        self.muscle_config_file = muscle_config_file
        self.joint_id = {}
        self.link_id = {}
        self.joint_type = {}
        self.activations = []
        self.hand_positions = []

        self.forces_scale = cost_scale
        self.threshold = threshold

        self.threshold_x = self.threshold 
        self.threshold_y = self.threshold
        self.threshold_z = self.threshold

        if self.use_sphere:
            self.sphere = p.loadURDF("sphere_small.urdf", globalScaling=.1) #visualizes target position

        self.ctrl = ctrl #control, list of all joints in right arm (shoulder, elbow, wrist + metacarpus for measuring hand pos)
        self.pose_file = pose_file
        
        #####MUSCLES + DATA LOGGING#####
        self.container = Container(max_iterations=int(100000))

        # Physics simulation to namespace
        self.sim_data = self.container.add_namespace('physics')

        self.initialize_muscles()
        model_utils.reset_model_position(self.model, self.pose_file)
        self.container.initialize()
        #self.muscles.setup_integrator()

        #####META PARAMETERS FOR SIMULATION#####
        self.n_fixedsteps = 0
        self._max_episode_steps = timestep #Does not matter. It is being set in the main.py where the total number of steps are being changed.
        self.frame_skip= frame_skip

        p.resetDebugVisualizerCamera(0.3, 15, -10, [0, 0.21, 0])
        self.muscle_list = ['AN', 'BBL', 'BBS', 'BRA', 'COR', 'ECRB', 'ECRL', 'ECU', 'EIP1', 'EIP2', 'FCR', 'FCU', 'PLO', 'PQU', 'PTE', 'TBL', 'TBM', 'TBO']
        self.action_space = spaces.Box(low=np.ones(18), high=np.ones(18), dtype=np.float32)
        self.seed()

    def get_cost(self, forces):
        scaler= 1/50
        cost = scaler * np.sum(np.abs(forces))
        return cost

    def controller_to_actuator(self, forces):
        for i, muscle in enumerate(self.muscle_list):
            self.container.muscles.activations.set_parameter_value(f"stim_RIGHT_FORE_{muscle}", forces[i])

    def get_activations(self):
        activations = []
        for muscle in self.muscle_list:
            activations.append(self.container.muscles.states.get_parameter_value(f'activation_RIGHT_FORE_{muscle}'))
        return activations

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_ids(self):
        return self.client, self.model
    
    #####OVERWRITTEN IN CHILD CLASS#####
    def reset_model(self, pose_file):
        raise NotImplementedError

    def initialize_muscles(self):
        self.muscles = MusculoSkeletalSystem(self.container, 1e-3, self.muscle_config_file)

    def do_simulation(self):
        self.muscles.step()
        self.container.update_log()
        p.stepSimulation()
    
    #####DISCONNECTS SERVER#####
    def close(self):
        p.disconnect(self.client)