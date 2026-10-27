# Builds the environment the way simulate.py does, takes two steps from the
# start of condition 0, and prints what it saw as JSON on a PROBE_RESULT line.
# Usage: python tests/_probe_env.py <main.py args>
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import config  # noqa: E402
from SAC import kinematics_preprocessing_specs  # noqa: E402
from SAC.RL_Framework_Mujoco import Muscle_Env  # noqa: E402

args = config.config_parser().parse_args()
env = Muscle_Env(args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 1, args)

out = {"kin_train": {str(c): k.tolist() for c, k in env.kin_train.items()}, "cond_scalars": []}
for c in range(env.n_exp_conds):
    env.reset(c)
    out["cond_scalars"].append(env.condition_scalar)

env.reset(0)
rest = np.zeros(env.model.nu)
_, r1, d1, _ = env.step(rest)     # step 1 also moves the targets onto the data path
out.update(
    kin_start=env.kin_to_sim[0][:, :, 0].tolist(),
    target_qpos={j: float(env.data.qpos[env.model.joint(j).qposadr[0]])
                 for j in kinematics_preprocessing_specs.musculo_target_joints})
_, r2, d2, _ = env.step(rest)     # step 2 is scored against those positions
out.update(rewards=[float(r1), float(r2)], dones=[bool(d1), bool(d2)])
print("PROBE_RESULT " + json.dumps(out))
