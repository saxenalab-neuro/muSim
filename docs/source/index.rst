Welcome to 𝜂𝜇Sim's documentation!
===================================

**𝜂𝜇Sim** is a methodology for predicting motor related
neural activity using recurrent neural newtorks (RNNs) and 
complex musculoskeletal systems. This repo contains code to train
multiple musculoskeletal systems using deep reinforcement learning (DRL)
and RNNs in order to capture the underlying dynamics used by the motor cortex 
to generate movement. We provide a system to properly instantiate musculoskeletal 
models using Mujoco and train them to reproduce experimentally recorded kinematics. 
We additionally implement several statistical tests to compare trained the RNN's 
activity with user's provided experimental neural data to compare their representations.

Check out the :doc:`installation` section to install necessary requirements, and :ref:`usage` 
for setting up, traininig, and evaluating the resulting DRL algorithm.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   installation
   usage
