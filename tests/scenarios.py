"""Builds small custom muSim setups, to try things other than the monkey cycling task.

Used by test_future.py. A Scenario puts everything muSim needs into a
temporary folder (dataset, optional neural data, model, config, starting
pose), with whatever changes a test wants: a body renamed, a dataset without
neural recordings, a model file with another name, and so on. The repo's own
files are never changed.
"""
import copy
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

import musim_runner as mr

N_ARM_JOINTS = 7    # the monkey arm's own joints; target joints come after these


def cycling_data():
    """Fresh copies of the lab's cycling kinematics and neural data, so a test
    can change them freely: (kinematics, neural activity)."""
    with open(mr.REPO / "kinematics_data" / "kinematics.pkl", "rb") as f:
        kin = pickle.load(f)
    with open(mr.REPO / "nusim_neural_data" / "neural_activity.pkl", "rb") as f:
        neural = pickle.load(f)
    return copy.deepcopy(kin), copy.deepcopy(neural)


class Scenario:
    """One custom setup in a temporary folder.

    Arguments:
        tmp           the folder to build in (a test's tmp_path)
        kinematics    the dataset to use (usually cycling_data(), changed)
        neural        neural activity, or None for a dataset without recordings
        rename_hand   give the model's "hand" body a different name
        model_file    file name for the model (default is the lab's usual name)
        model_subdir  put the model in a subfolder with a different name
        drop_settings config settings to leave out
        extra_args    extra command-line settings

    Use .probe() for a quick look at the first two steps, or .train() for a
    short training run.
    """

    def __init__(self, tmp, kinematics, neural=None, rename_hand=None,
                 model_file="musculoskeletal_model.xml", model_subdir=".",
                 drop_settings=(), extra_args=()):
        self.tmp = Path(tmp)
        self.kinematics = kinematics

        # The dataset, in the same file format the lab uses
        data = self.tmp / "data"
        data.mkdir()
        with open(data / "kinematics.pkl", "wb") as f:
            pickle.dump(kinematics, f)
        # The neural data folder. Left empty when neural is None, which is
        # how muSim knows there are no recordings.
        neural_dir = self.tmp / "neural"
        neural_dir.mkdir()
        if neural is not None:
            with open(neural_dir / "neural_activity.pkl", "wb") as f:
                pickle.dump(neural, f)

        # The default config, minus any settings the test wants left out
        cfg = mr.write_config(self.tmp)
        lines = cfg.read_text().splitlines()
        cfg.write_text("\n".join(l for l in lines if l.split("=")[0].strip() not in drop_settings) + "\n")

        # The arm model, optionally with the hand body renamed and/or a different file name
        model_dir = self.tmp / "model" / model_subdir
        model_dir.mkdir(parents=True)
        xml = (mr.REPO / "musculoskeletal_model" / "musculoskeletal_model.xml").read_text()
        if rename_hand:
            xml = xml.replace('<body name="hand"', f'<body name="{rename_hand}"')
        self.model = model_dir / model_file
        self.model.write_text(xml)
        # The model refers to its mesh files by relative path, so link the folder in
        (model_dir / "Geometry").symlink_to(mr.REPO / "musculoskeletal_model" / "Geometry")

        # Starting pose: the arm's IK pose, with every target joint at 0
        # (one target joint per tracked coordinate, i.e. per non-NaN x/y/z row)
        n_target_joints = sum(int(np.isfinite(k[m, c]).all())
                              for k in [kinematics["train"][0]]
                              for m in range(k.shape[0]) for c in range(3))
        pose_dir = self.tmp / "pose"
        pose_dir.mkdir()
        arm = np.load(mr.GOLDEN / "initial_pose" / "initial_qpos_opt.npy")[:N_ARM_JOINTS]
        np.save(pose_dir / "initial_qpos_opt.npy", np.concatenate([arm, np.zeros(n_target_joints)]))

        # Command-line settings that point muSim at everything above
        self.args = ["--config", cfg, "--musculoskeletal_model_path", self.model,
                     "--kinematics_path", data, "--nusim_data_path", neural_dir,
                     "--initial_pose_path", pose_dir, *extra_args]
        # Add the target bodies to the model, as the lab's pipeline does
        mr.run("append_musculo_targets.py", *self.args)

    def probe(self):
        """Build the environment and take two steps (see _probe_env.py).
        Returns what it saw, as a dict."""
        out = mr.run("tests/_probe_env.py", *self.args)
        # Find our result line among anything else that was printed
        line = next(l for l in out.splitlines() if l.startswith("PROBE_RESULT "))
        return json.loads(line[len("PROBE_RESULT "):])

    def train(self, episodes=3):
        """A short training run on CPU. Returns how many steps each episode lasted."""
        out = self.tmp / "train"
        (out / "stats").mkdir(parents=True)
        (out / "checkpoint").mkdir()
        mr.run("tests/_cpu_main.py", *self.args, "--total_episodes", episodes,
               "--statistics_folder", out / "stats", "--checkpoint_folder", out / "checkpoint")
        return np.load(out / "stats" / "stats_steps.npy")
