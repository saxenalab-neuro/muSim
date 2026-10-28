# Builds the muSim environment and reports what it sees in its first two steps.
#
# Used by scenarios.py (Scenario.probe) to check custom setups quickly,
# without training. It builds the environment exactly the way simulate.py does,
# takes two steps from the start of condition 0 with all muscles relaxed, and
# prints the results as one line: "PROBE_RESULT {...json...}".
#
# Why a separate script: muSim reads its settings from the command line as
# soon as its modules are imported, so each custom setup has to run in its own
# Python process with its own arguments.
#
# Usage, from the repo root:  python tests/_probe_env.py <the same args as main.py>
import json
import os
import sys

import numpy as np

# Make the lab's modules importable (they expect to run from the repo root)
sys.path.insert(0, os.getcwd())
import config  # noqa: E402
from SAC import kinematics_preprocessing_specs  # noqa: E402
from SAC.RL_Framework_Mujoco import Muscle_Env  # noqa: E402

# Read the settings from the command line, as main.py does
args = config.config_parser().parse_args()

# Same model path trick as simulate.py: swap the file name for the model that
# includes the targets (this is also hardcode H6, which test_future.py checks)
env = Muscle_Env(args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 1, args)

# What the environment loaded: the training kinematics after scaling and
# centring, and the condition cue for every training condition
out = {"kin_train": {str(c): k.tolist() for c, k in env.kin_train.items()}, "cond_scalars": []}
for c in range(env.n_exp_conds):
    env.reset(c)
    out["cond_scalars"].append(env.condition_scalar)

# Two steps from the start of condition 0, with every muscle command at 0
env.reset(0)
rest = np.zeros(env.model.nu)
_, r1, d1, _ = env.step(rest)     # step 1 also moves the targets onto the data path
out.update(
    # Where the data says each target should start: [marker, x/y/z]
    kin_start=env.kin_to_sim[0][:, :, 0].tolist(),
    # Where the simulator actually put each target joint (e.g. "box:x0")
    target_qpos={j: float(env.data.qpos[env.model.joint(j).qposadr[0]])
                 for j in kinematics_preprocessing_specs.musculo_target_joints})
_, r2, d2, _ = env.step(rest)     # step 2 is scored against those positions
out.update(rewards=[float(r1), float(r2)], dones=[bool(d1), bool(d2)])

# The prefix lets the caller find this line even if MuJoCo prints warnings after it
print("PROBE_RESULT " + json.dumps(out))
