"""Make cue.pkl for the monkey cycling data.

What the cue is:
    One number that tells the network which movement (condition) it is doing.
    The network gets it as its last input at every step.

Why this script exists:
    The cue used to be worked out inside the code from the movement's length,
    with two numbers from the cycling data typed in:

        cue = (time points - 600) / (1319 - 600)

    Now the user gives the cue in its own file, cue_data/cue.pkl. This script
    writes that file for the cycling data, using the same formula, so the
    cycling results stay exactly as they were. It is kept in the repository to
    show where the numbers in cue.pkl come from.

    For your own data you do not need this script. Save your own cue.pkl in
    the format below, with any cue values you like.

Format of cue.pkl (the same layout as kinematics.pkl):
    {'train': {0: cue, 1: cue, ...}, 'test': {0: cue, 1: cue, ...}}
    Each cue is a 1-D numpy array with one value per time point, so it is as
    long as that movement's kinematics. For the cycling data every cue is one
    number repeated (e.g. 0.1419 written 702 times).

Run from the repository root:
    python cue_data/make_cycling_cue.py
"""

import os
import pickle

import numpy as np

KINEMATICS_FILE = os.path.join('kinematics_data', 'kinematics.pkl')
CUE_FILE = os.path.join('cue_data', 'cue.pkl')

# The shortest-allowed and longest movement lengths of the cycling data, as
# they were typed into SAC/RL_Framework_Mujoco.py before this file existed.
SHORTEST, LONGEST = 600, 1319

with open(KINEMATICS_FILE, 'rb') as f:
    kinematics = pickle.load(f)

cue = {}
for split in ('train', 'test'):
    cue[split] = {}
    for cond, movement in kinematics[split].items():
        timepoints = movement.shape[-1]     # movement is [targets, 3, time points]
        value = (timepoints - SHORTEST) / (LONGEST - SHORTEST)
        cue[split][cond] = np.full(timepoints, value)
        print(f'{split} movement {cond}: {timepoints} time points, cue {value:.4f}')

with open(CUE_FILE, 'wb') as f:
    pickle.dump(cue, f)
print(f'Saved {CUE_FILE}')
