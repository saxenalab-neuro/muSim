import yaml
import pybullet as p
import numpy as np

arm_indexes = [104, 105, 106, 107, 108, 110, 111, 112]
#RShoulder_rotation - 104, looks to move right to left circular, .00025
#RShoulder_adduction - 105, looks to move right to left circular, .002
#RShoulder_flexion - 106, moves up and down, .0003 
#RElbow_flexion - 107, awkward-no movement, .0003
#RElbow_supination - 108, similar to right to left shoulder movement
#RWrist_adduction - 110 "", .002
#RWrist_flexion - 111 .0002
#RMetacarpus1_flextion - 112, use link (carpus)

def disable_control(modelId):
    num_joints = p.getNumJoints(modelId)
    p.setJointMotorControlArray(
        modelId,
        np.arange(num_joints),
        p.VELOCITY_CONTROL,
        targetVelocities=np.zeros((num_joints,)),
        forces=np.zeros((num_joints,))
    )
    p.setJointMotorControlArray(
        modelId,
        np.arange(num_joints),
        p.POSITION_CONTROL,
        forces=np.zeros((num_joints,))
    )
    p.setJointMotorControlArray(
        modelId,
        np.arange(num_joints),
        p.TORQUE_CONTROL,
        forces=np.zeros((num_joints,))
    )

def initialize_joint_list(num_joints):
    joint_list =[]
    for joint in range(num_joints):
        joint_list.append(joint)
    return joint_list

def generate_joint_id_to_name_dict(modelId):
    joint_Dictionary ={}
    for i in range(p.getNumJoints(modelId)):
        joint_Dictionary[i] = p.getJointInfo(modelId, i)[1].decode('UTF-8') 
    return joint_Dictionary

def generate_name_to_joint_id_dict(modelId):
    name_Dictionary ={}
    for i in range(p.getNumJoints(modelId)):
        name_Dictionary[p.getJointInfo(modelId, i)[1].decode('UTF-8')] = i
    return name_Dictionary

def initialize_position(modelId, pose_file, joint_list):
    with open(pose_file) as stream:
        data = yaml.load(stream, Loader=yaml.SafeLoader)
        data = {k.lower(): v for k, v in data.items()}
    for joint in joint_list:
        #joint_name =p.getJointInfo(mouseId, joint)[1] 
        _pose = np.deg2rad(data.get(p.getJointInfo(modelId, joint)[1].decode('UTF-8').lower(), 0))#decode removes b' prefix
        p.resetJointState(modelId, joint, targetValue=_pose)

def reset_model_position(model, pose_file): 
        joint_list = []
        for joint in range(p.getNumJoints(model)):
            joint_list.append(joint)
        with open(pose_file) as stream:
            data = yaml.load(stream, Loader=yaml.SafeLoader)
            data = {k.lower(): v for k, v in data.items()}
        for joint in joint_list:
            _pose = np.deg2rad(data.get(p.getJointInfo(model, joint)[1].decode('UTF-8').lower(), 0)) #decode removes b' prefix
            p.resetJointState(model, joint, targetValue=_pose)

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z) #rho
    el = np.arctan2(z, hxy) #theta
    az = np.arctan2(y, x) #phi
    return az, el, r

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def get_speed(curr, prev, time=.001):
    return abs(curr-prev) / time