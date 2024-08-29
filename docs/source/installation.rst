Installation
============

We highly recommend a linux system for easy installation.

First you will need to install Mujoco (older version). Please make sure that Anaconda as well as git are also installed on your system.

1. Download the library using this link: https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

2. Create a hidden folder in your root directory called .mujoco as such (replacing the path with the path on your computer): 
    
    ``mkdir /home/username/.mujoco``

3. Extract the downloaded library into the newly created hidden folder:

    ``tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/``

4. Open your .bashrc file in your root/home directory:

    ``nano .bashrc``

5. Once in the .bashrc file, add the following line replacing the path with your true home directory:

    ``export LD_LIBRARY_PATH=/home/user-name/.mujoco/mujoco210/bin``

6. If your system has an nvidia GPU, add this line as well:

    ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia``

7. Save, close, then source the .bashrc using the following command:

    ``source ~/.bashrc``

8. Reboot your system to ensure changes are made

9. Create a new environment using conda:

    ``conda env create --name nusim --file=requirements.yml``

10. Activate the conda environment:

    ``conda activate nusim``


If facing errors
----------------

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




