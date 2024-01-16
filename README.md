# 𝜂𝜇Sim

![Fig1_github](https://github.com/saxenalab-neuro/Biomechanical-DRL/assets/77393494/a191621f-8f0d-45ed-8206-d4c7a452ca1b)

Training LSTMs and ANNs to perform tasks with musculoskeletal models. 
Environments include human arms performing reaching and a monkey model performing cycling.

Link to corresponding EMBC paper (https://ieeexplore.ieee.org/document/9871085)

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

    `echo -e 'export LD_LIBRARY_PATH=/home/user-name/.mujoco/mujoco210/bin `

6. If your system has an nvidia GPU, add this line as well:

    `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia `

7. Save, close, then source the .bashrc using the following command:

    `source ~/.bashrc`

8. Create a conda environment with the name of your choosing, then activate that environment and install mujoco-py as such:

    `pip install -U 'mujoco-py<2.2,>=2.1'`

9. Reboot your system to ensure changes are made

10. If you are on linux (and may apply to Mac as well), there will likely be additional packages necessary. Here is a list of possible packages:

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

11. Lastly, within the conda environment there are additional packages necessary to ensure the training can run:

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

add your root directory `root_dir = your_root_name`, name of the folder you created `checkpoint_folder = your_checkpoint_folder`, as well as the name of the file to save the model `checkpoint_file = model_name`. Your model will then be saved for training and can be tested on afterwards by setting `test = True` in the config file. While training, a statistics file will be saved in the project folder as well, containing rewards and agent losses.



