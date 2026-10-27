"""Helpers that run the muSim scripts end to end and summarise their output.

muSim reads its settings from the command line at import time and uses paths
relative to the repo root, so each run is a subprocess started from the repo
root. Everything a run writes goes into a temporary folder.
"""
import hashlib
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden"

# Short seeded training run used for the golden output
TRAIN_EPISODES = 30
SAVE_ITER = 10

# Training setups with golden output: the lab's default config, plus options
# the default leaves off (extra command-line settings on top of the default)
TRAIN_VARIANTS = {
    "default": [],
    "nusim": ["--zeta_nusim", "10000"],            # match recorded neurons (1e4, per the comment in sac.py)
    "senses": ["--joint_feedback", "True", "--muscle_forces", "True",
               "--visual_feedback", "True"],       # joint angles, muscle forces, body positions
    "velocity_delay": ["--visual_velocity", "hand", "target0",
                       "--sensory_delay_timepoints", "3"],
}

STAT_NAMES = ["rewards", "steps", "policy_loss", "critic_loss"]
NET_NAMES = ["agent_state_dict", "critic_state_dict", "critic_target_state_dict"]
TEST_ARRAYS = ["emg", "rnn_activity", "rnn_input", "rnn_input_fp",
               "kinematics_mbodies", "kinematics_mtargets"]


def run(script, *args):
    """Run a repo script with the current Python, from the repo root."""
    cmd = [sys.executable, str(script), *map(str, args)]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "PYTHONHASHSEED": "0"})
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stderr[-3000:]}")
    return out.stdout


def write_config(tmp):
    """The lab's default config without `cuda = True`, which can't be switched
    off from the command line."""
    lines = (REPO / "configs" / "configs.txt").read_text().splitlines()
    cfg = Path(tmp) / "config.txt"
    cfg.write_text("\n".join(l for l in lines if not l.strip().startswith("cuda")) + "\n")
    return cfg


def build_model(tmp, cfg):
    """Copy the arm model into tmp and append the target bodies to it, as
    append_musculo_targets.py does. Returns the path to musculoskeletal_model.xml."""
    model_dir = Path(tmp) / "model"
    model_dir.mkdir()
    src = REPO / "musculoskeletal_model"
    shutil.copy(src / "musculoskeletal_model.xml", model_dir)
    (model_dir / "Geometry").symlink_to(src / "Geometry")
    model = model_dir / "musculoskeletal_model.xml"
    run("append_musculo_targets.py", "--config", cfg, "--musculoskeletal_model_path", model)
    return model


def train(tmp, cfg, model, variant="default"):
    """Short seeded training run for one of TRAIN_VARIANTS. Returns the folder it wrote to."""
    out = Path(tmp) / f"train_{variant}"
    (out / "stats").mkdir(parents=True)
    (out / "checkpoint").mkdir()
    run("tests/_cpu_main.py", "--config", cfg,
        "--musculoskeletal_model_path", model,
        "--initial_pose_path", GOLDEN / "initial_pose",
        "--total_episodes", TRAIN_EPISODES,
        "--save_iter", SAVE_ITER,
        "--statistics_folder", out / "stats",
        "--checkpoint_folder", out / "checkpoint",
        *TRAIN_VARIANTS[variant])
    return out


def run_pretrained(tmp, cfg, model):
    """Run the lab's shipped checkpoint in test mode. Returns the test data dict."""
    out = Path(tmp) / "test"
    out.mkdir()
    run("tests/_cpu_main.py", "--config", cfg, "--mode", "test",
        "--musculoskeletal_model_path", model,
        "--initial_pose_path", GOLDEN / "initial_pose",
        "--checkpoint_folder", REPO / "checkpoint",
        "--test_data_filename", out)
    with open(out / "test_data.pkl", "rb") as f:
        return pickle.load(f)


def sha256(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def load_stats(folder):
    return {k: np.load(Path(folder) / "stats" / f"stats_{k}.npy") for k in STAT_NAMES}


def checkpoint_summary(path):
    """Episode it was saved at, plus a hash of every network's weights."""
    ck = torch.load(path, map_location="cpu")
    summary = {"iteration": int(ck["iteration"])}
    for name in NET_NAMES:
        sd = ck[name]
        summary[name] = sha256(*[sd[k].cpu().numpy() for k in sorted(sd)])
    return summary


def summarize_test(data):
    """Hash of every saved test array, plus the readable mean hand-target
    distance (cm) per condition."""
    conds = sorted(data["emg"])
    summary = {"hashes": {k: {str(c): sha256(data[k][c]) for c in conds} for k in TEST_ARRAYS}}
    summary["mean_error_cm"] = {
        str(c): float(np.linalg.norm(data["kinematics_mbodies"][c] - data["kinematics_mtargets"][c],
                                     axis=-1).mean() * 100)
        for c in conds}
    return summary
