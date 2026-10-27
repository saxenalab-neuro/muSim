"""Builds small custom setups (dataset, model, config, starting pose) in a
temporary folder, to try muSim on things other than the monkey cycling task."""
import copy
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

import musim_runner as mr

N_ARM_JOINTS = 7    # the monkey arm's own joints; target joints come after these


def cycling_data():
    with open(mr.REPO / "kinematics_data" / "kinematics.pkl", "rb") as f:
        kin = pickle.load(f)
    with open(mr.REPO / "nusim_neural_data" / "neural_activity.pkl", "rb") as f:
        neural = pickle.load(f)
    return copy.deepcopy(kin), copy.deepcopy(neural)


class Scenario:
    def __init__(self, tmp, kinematics, neural=None, rename_hand=None,
                 model_file="musculoskeletal_model.xml", model_subdir=".",
                 drop_settings=(), extra_args=()):
        self.tmp = Path(tmp)
        self.kinematics = kinematics

        data = self.tmp / "data"
        data.mkdir()
        with open(data / "kinematics.pkl", "wb") as f:
            pickle.dump(kinematics, f)
        neural_dir = self.tmp / "neural"
        neural_dir.mkdir()
        if neural is not None:
            with open(neural_dir / "neural_activity.pkl", "wb") as f:
                pickle.dump(neural, f)

        cfg = mr.write_config(self.tmp)
        lines = cfg.read_text().splitlines()
        cfg.write_text("\n".join(l for l in lines if l.split("=")[0].strip() not in drop_settings) + "\n")

        model_dir = self.tmp / "model" / model_subdir
        model_dir.mkdir(parents=True)
        xml = (mr.REPO / "musculoskeletal_model" / "musculoskeletal_model.xml").read_text()
        if rename_hand:
            xml = xml.replace('<body name="hand"', f'<body name="{rename_hand}"')
        self.model = model_dir / model_file
        self.model.write_text(xml)
        (model_dir / "Geometry").symlink_to(mr.REPO / "musculoskeletal_model" / "Geometry")

        # Starting pose: the arm's IK pose, with every target joint at 0
        n_target_joints = sum(int(np.isfinite(k[m, c]).all())
                              for k in [kinematics["train"][0]]
                              for m in range(k.shape[0]) for c in range(3))
        pose_dir = self.tmp / "pose"
        pose_dir.mkdir()
        arm = np.load(mr.GOLDEN / "initial_pose" / "initial_qpos_opt.npy")[:N_ARM_JOINTS]
        np.save(pose_dir / "initial_qpos_opt.npy", np.concatenate([arm, np.zeros(n_target_joints)]))

        self.args = ["--config", cfg, "--musculoskeletal_model_path", self.model,
                     "--kinematics_path", data, "--nusim_data_path", neural_dir,
                     "--initial_pose_path", pose_dir, *extra_args]
        mr.run("append_musculo_targets.py", *self.args)

    def probe(self):
        """Build the environment and take two steps. Returns what it saw."""
        out = mr.run("tests/_probe_env.py", *self.args)
        line = next(l for l in out.splitlines() if l.startswith("PROBE_RESULT "))
        return json.loads(line[len("PROBE_RESULT "):])

    def train(self, episodes=3):
        out = self.tmp / "train"
        (out / "stats").mkdir(parents=True)
        (out / "checkpoint").mkdir()
        mr.run("tests/_cpu_main.py", *self.args, "--total_episodes", episodes,
               "--statistics_folder", out / "stats", "--checkpoint_folder", out / "checkpoint")
        return np.load(out / "stats" / "stats_steps.npy")
