# 𝜂𝜇Sim
![Fig1_github](https://github.com/saxenalab-neuro/muSim/assets/77393494/aefcb769-7427-4654-be72-08e1d6f59642)


Training LSTMs and ANNs to perform tasks with musculoskeletal models. 
Environments include monkey model performing cycling.

Link to corresponding paper (https://www.biorxiv.org/content/10.1101/2024.02.02.578628v1)

## Installation


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

9. To create a conda environment with dependencies installed automatically use:

    `conda env create --name nuSim --file=nuSim_env.yml`

10. Activate the conda environment using:

    `conda activate nuSim`

**Note:** **Skip the following steps, if installed environment using the yml file.**
**If facing errors, manually installing the following dependencies may help**

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

## Usage

To train an agent, run 

`python main.py --config configs/config_file_name.txt` 

with the corresponding config file of your choice. The corresponding commands are provided in the config.py file. To ensure your model saves (checkpoint containing parameters and optimizer states), create a folder with the name of your choosing in which to save the state_dict. In your configuration file 

`configs/config_file_name.txt`, 

add your root directory `root_dir = your_root_name`, name of the folder you created `checkpoint_folder = your_checkpoint_folder`, as well as the name of the file to save the model `checkpoint_file = model_name`. Your model will then be saved for training and can be tested on afterwards by setting `mode = "test"` in the config file, while making sure to specify how to save your testing data (kinematics, rnn activity, etc.) by selecting `test_data_filename = filename`. While training, a statistics file will be saved in the project folder as well, containing rewards and agent losses.

## Neural Analysis

For CCA and PCA, run

`python ./Neural_Analysis/CCA.py`

For Linear Regression Analysis (LRA), run

`python ./Neural_Analysis/LRA.py`

To perform the neural analysis on your dataset, please consult ./Neural_Analysis/Training/Saving Neural Activities.txt for saving data format
