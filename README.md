# 𝜂𝜇Sim
![Fig1_github](https://github.com/saxenalab-neuro/muSim/assets/77393494/aefcb769-7427-4654-be72-08e1d6f59642)


Training LSTMs and ANNs to perform tasks with musculoskeletal models. 
Environments include monkey model performing cycling.

Link to corresponding paper (https://www.biorxiv.org/content/10.1101/2024.02.02.578628v1)

## Installation

We highly recommend a linux system for easy installation.

First you will need to install Mujoco (older version). Please make sure that Anaconda as well as git are also installed on your system.

1. Download the library using this link: https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

2. Create a hidden folder in your root directory called .mujoco as such (replacing the path with the path on your computer): 
    
    `mkdir /home/username/.mujoco`

3. Extract the downloaded library into the newly created hidden folder:

    `tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/`

4. Open your .bashrc file in your root/home directory:

    `nano .bashrc`

5. Once in the .bashrc file, add the following line replacing the path with your true home directory:

    `export LD_LIBRARY_PATH=/home/user-name/.mujoco/mujoco210/bin`

6. If your system has an nvidia GPU, add this line as well:

    `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia `

7. Save, close, then source the .bashrc using the following command:

    `source ~/.bashrc`

8. Reboot your system to ensure changes are made

9. Create a new environment using conda:

    `conda env create --name mujoco_env --file=requirements.yml`

10. Activate the conda environment:

    `conda activate mujoco_env`


**If facing errors, installing the following libraries may help**

1. If you are on linux (and may apply to Mac as well), there will likely be additional packages necessary. Here is a list of possible packages:

    * patchelf
    * python3-dev
    * build-essential
    * libssl-dev
    * libffi-dev
    * libxml2-dev
    * libxslt1-dev
    * zlib1g-dev
    * libglew1.5
    * libglew-dev
    * libosmesa6-dev
    * libgl1-mesa-glx
    * libglfw3

    If facing errors, adding these packages may help.

2. Lastly, within the conda environment there are additional packages necessary to ensure the training can run:

    * cython
    * matplotlib
    * scipy
    * torch
    * PyYaml
    * configargparse
    * numpy
    * gym
    * pandas
    * pyquaternion
    * scikit-video

## Basic Usage for the Monkey Cycling Task

<p align="center"> <img src="https://github.com/saxenalab-neuro/muSim/assets/77393494/2073cc37-c44a-4558-82ae-a0c54e5573c4" width="50%" height="50%"> </p>

1. To train the controller, run the following in terminal:

    `python main.py --config configs/configs.txt`

    This will save the controller in the ./checkpoint file with training iterations. The highest reward should reach >= 55000 for kinematic accuracy.

   The episode reward with iterations should look like this:
   (There may be slight variations due to random seed but trend should look similar)
   
   <p align="center"> <img src="https://github.com/saxenalab-neuro/muSim/assets/77393494/d3a7578c-035d-4a8c-b87b-853e3d03187c" width="50%" height="50%"> </p>

## General Usage

**Inputs:**

**Musculoskeletal Model:**

1. Musculoskeletal Model: Save the MuJoCo musculoskeletal model in “./musculoskeletal_model/” as musculoskeletal_model.xml alongwith the Geometry files

(The path to the musculoskeletal_model.xml can also be specified in the configs.txt file with *musculoskeletal_model_path* param, if not using the above default path)

2. For conversion of musculoskeletal model from OpenSim to MuJoCo, please refer to MyoConverter: https://github.com/MyoHub/myoconverter

**Experimental Kinematics:**

1. Save the experimental kinematics in ‘./kinematics_data/kinematics.pkl’ as a Python dict object with the following format:
    
    dict{
    

```
    <'marker_names'> : <['marker_name_1', ..., 'marker_name_n']>,

    <'train'> : <dict_train>,

    <'test'> : <dict_test>
```

}

1. <’marker_names’> contain a list of names of the experimental markers that were recorded. The marker_name must correspond to a body name in the musculoskeletal model xml file. 
2. <dict_train> and <dict_test> are Python dictionary objects that contain kinematics in the following format

{ 
```
<key> :   <value>,

<key>:   <value>,

.
.
.

<key>: <value>
```

}

<key: int> is the integer index of the corresponding condition. (starts from 0 for the first condition for both the training and testing conditions) 

<value: numpy.ndarray> contains the kinematics for the corresponding condition with shape: [num_markers/targets, num_coordinates = 3, timepoints]. 

num_markers are the number of experimental markers/bodies that are recorded. The order of the num_markers must correspond to the order in which the marker_names are listed. For example, if ‘marker_names’ = [’hand’, ‘elbow’], num_marker= 0 should contain the experimental kinematics for hand and num_marker=1 should contain the experimental kinematics for elbow.

num_coordinates are the x [-->], y[↑] and z[ out of page] coordinates. Values of NaN for any coordinate will keep that coordinate locked. 

An example for saving the experimental kinematics for the cycling task is given in ./exp_kin_cycling/saving_exp_kin.ipynb

(The path to kinematics.pkl file can also be specified using *kinematics_path* param in configs.txt file) 

**Neural Data (optional):**

1. Save the recorded neural data for the training and testing conditions in ‘./nusim_neural_data/neural_activity.pkl’ as a Python dict object:
dict{

```

    <'train'> : <dict_train>,

    <'test'> : <dict_test>
```

}

1. <dict_train> and <dict_test> are Python dictionary objects that contain the neural data in the following format:

<key: int> is the integer index of the corresponding condition as in the kinematics file.

<value: numpy.ndarray> is the numpy array that contains recorded firing rates with the following shape: [timepoints, num_neurons]. num_neurons are the total number of recorded neurons.

Note: If this step is omitted, various post-processing analyses which require recorded neural data such as CCA, will not run. nuSim training will also not proceed.

(nusim_data_path can also be specified in the configs.txt file)

**Stimulus Data (optional):**

Provide any experimental stimulus data in ‘./stimulus_data/stimulus_data.pkl’ as a Python dict object. 

dict{

```

    <'train'> : <dict_train>,

    <'test'> : <dict_test>
```

}

1. <dict_train> and <dict_test> are Python dictionary objects that contain the experimental stimulus data in the following format:

<key: int> is the integer index of the corresponding condition as in the kinematics file.

<value: numpy.ndarray> is the numpy array that contains recorded stimulus data with the following shape: [timepoints, num_features]. num_features are the corresponding features in that stimulus.

**Initial Pose (optional):**

Save the initial pose (containing the qpos and qvel) as numpy arrays in ‘./inital_pose/’ as qpos.npy and qvel.npy with shape [nq, ]. nq is the number of joints in the xml model.

This step is optional. If omitted, the default initial pose for xml model will be used for CMA-ES and IK.

(initial_pose_path can also be specified in the configs.txt file)

**Specifications:**

Provide the parameters for various modules using the ‘./configs/configs.txt’ file. The details of each parameter/specification is given in the configs.txt file.

**General Usage:**

**Inverse Kinematics:**

1. **Append the xml model with targets:**

Run:

`python append_musculo_targets.py`

This will append targets to the musculoskeletal xml file that will follow the preprocessed markers kinematics during simulation.

2. **Find the initial pose for xml model using CMA-ES and Inverse Kinematics:**

a. Run the following command in the terminal:

`python find_init_pose.py --config configs/configs.txt --visualize True`

This will use inverse kinematics (IK) to find the initial pose for the xml model to match the initial timepoint of the target kinematics.

If you see the output, ‘Initial Pose found and saved’, skip 1b.

b. Run:

`python find_init_pose_ik_cma.py --config configs/configs.txt --visualize True`

This willl use CMA-ES optimization with IK to find a good initial pose for the xml model. 

If you see, ‘Initial Pose found and saved using CMA-ES and Inverse Kinematics’, proceed to the next step. 
    
Otherwise, provide a good inital pose for the xml model that preferably starts nearer to the inital marker/target position.
    
3. **Visualize the targets/markers trajectories using randomly initialized uSim network:**

Run

`python main --config configs/configs.txt --visualize True --mode test`

This will visualize the target trajectories using a randomly initialized uSim controller network. Make sure target trajectories look as desired. Otherwise, change the kinematics preprocessing parameters (e.g. trajectory_scaling, center) in the ./configs/configs.txt file.

4. **Visualize the musculoskeletal model trajectory and save the corresponding sensory feedback:**

Run:

`python visualize_trajectories_ik.py --config configs/configs.txt --visualize True`
    
    
This will visualize the xml model following/tracking the training target trajectories. Before proceeding, make sure that the target trajectories are feasible and lie within the bounds of the xml model. Otherwise, adjust the target trajectories using the kinematics preprocessing parameters in configs.txt file.
    
This will also save the generated sensory feedback in ‘./test_data/sensory_feedback_ik.pkl’ as Python dict object: 

<key: int> corresponds to the integer index of the corresponding training condition

<value: numpy.ndarray> with shape: [timepoints, num_of_state_feedback_variables]

This can be used to get Proprioception for training neural networks.

**Training the uSim Controller using DRL:**

**(Make sure DRL/SAC related parameters are specified correctly in the configs.txt file)**

1. To train the uSim controller using the provided DRL algorithm, run:

`python main.py --config configs/configs.txt`
    
2. To continue the training from the previous session, run:

`python main.py --config configs/configs.txt --load_saved_nets_for_training True`

**Testing the uSim Controller:**

To test the trained uSim controller, run:

`python main.py --config configs/configs.txt --mode test --visualize True`

This will visualize the xml model performing movements for training and testing conditions using the trained uSim controller. 

This will also save the files used for post training analyses.

**Post Training Analyses:**

After training, the following modules are used for various analyses. All these modules are in ‘./Analysis’

1. **Kinematics Visualization:**

To visualize the kinematics for the training and testing conditions, see visualize_kinematics.ipynb

2. **PCA:**

To visualize the uSim controller’s population trajectories in PCA subspace, run:

`python collective_pca.py`

3. **Canonical Correlation Analysis (CCA):**

see CCA.ipynb

4. **Linear Regression Analysis (LRA):**

see LRA.ipynb

5. **Procrustes:**

see procrustes.ipynb

6. **Fixed Point (FP) Analysis:**

Run

`python find_fp.py`

The fixed point analysis is based on the original implementation: https://github.com/mattgolub/fixed-point-finder. Refer to the github repo for further information.

7. **Rotational Dynamics: (requires MATLAB)**

See and run jpca_nusim.m

Note: jPCA analysis is based on MM Churchland’s original implementation. Please see it for further details (https://www.dropbox.com/scl/fo/duf5zbwcibsux467c6oc9/AIN-ZiFsy2Huyh8h7VMdL7g?rlkey=3o5axmq5hirel4cij7g64jc0r&e=1&dl=0)

**Important for jPCA analysis:**

1. Make sure that ./Analyses/jPCA_ForDistribution is included in the MATLAB path alongwith all sub-directories

2. Make sure that usim test_data folder is included in the MATLAB path. test_data folder is where the jpca data is saved during usim test

**Perturbation Analyses:**

**Selective Feedback Elimination (SFE):**

Specify the part of the sensory feedback to be eliminated in ./SAC/perturbation_specs.py using *sf_elim* variable. Run:

`python main --config configs/configs.txt --mode SFE --visualize True`

**Sensory Perturbation:**

Specify the perturbation vector to be added to the selected sensory feedback in ./SAC/perturbation_specs.py, e.g. *muscle_lengths_pert*. Run:

`python main.py --config configs/configs.txt --mode sensory_pert --visualize True`

**Neural Perturbation:**

The neural perturbation will add the given perturbation to the nodes of the uSim/nuSim controller’s RNN.

Specify the neural perturbation vector in perturbation_specs.py using *neural_pert* variable. Run:

`python main.py --config configs/configs.txt --mode neural_pert --visualize True`

**Change Musculoskeletal Properties:**

To test the trained uSim controller under changed musculoskeletal properties:

1. Go to the folder ‘./musculoskeletal_model/’. Copy and paste the xml model ‘musculo_targets.xml’. Rename the copied model as ‘musculo_targets_pert.xml’.

2. Change the desired musculoskeletal properties in xml model ‘musculo_targets_pert.xml’.

3. Run:

`python main.py --config configs/configs.txt --mode musculo_properties --visualize True`

All the above perturbation analyses will change the post training analyses files in place. To run the post training analyses after perturbation see Post Training Analyses section.
