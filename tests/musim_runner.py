"""Helpers that run the real muSim scripts and summarise what they produce.

Used by the golden-output tests (test_golden.py), by make_golden.py (which
saves the expected results) and by scenarios.py.

Why the scripts run as separate processes: muSim reads its settings from the
command line as soon as its modules are imported, and it opens files with
paths relative to the repo root. So each run is started as a fresh Python
process from the repo root, with its own arguments. Everything a run writes
(models, statistics, checkpoints) goes into a temporary folder, so the repo's
own files are never touched.
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

# The muSim repo root (this file lives in <repo>/tests/)
REPO = Path(__file__).resolve().parents[1]
# Where the saved "expected" results live
GOLDEN = REPO / "tests" / "golden"

# Length of the short seeded training runs. 30 episodes is long enough to go
# through the shrinking target zone and save checkpoints, while the whole test
# suite still runs in about 2 minutes. A checkpoint is saved every 10 episodes.
TRAIN_EPISODES = 30
SAVE_ITER = 10

# The training setups that get golden output. Each is the lab's default config
# plus extra command-line settings. The extra setups switch on features the
# default leaves off, because a test can't notice a break in a feature that
# is switched off.
TRAIN_VARIANTS = {
    "default": [],
    "nusim": ["--zeta_nusim", "10000"],            # match recorded neurons (1e4, per the comment in sac.py)
    "senses": ["--joint_feedback", "True", "--muscle_forces", "True",
               "--visual_feedback", "True"],       # joint angles, muscle forces, body positions
    "velocity_delay": ["--visual_velocity", "hand", "target0",
                       "--sensory_delay_timepoints", "3"],  # hand/target velocity, feedback 3 steps late
}

# Per-episode training statistics that muSim saves (stats_<name>.npy)
STAT_NAMES = ["rewards", "steps", "policy_loss", "critic_loss"]
# The networks stored in each checkpoint file
NET_NAMES = ["agent_state_dict", "critic_state_dict", "critic_target_state_dict"]
# What muSim saves when testing a trained model (test_data.pkl):
# muscle commands, RNN activity, RNN inputs, and the body and target paths
TEST_ARRAYS = ["emg", "rnn_activity", "rnn_input", "rnn_input_fp",
               "kinematics_mbodies", "kinematics_mtargets"]


def run(script, *args):
    """Run one of the repo's scripts with the current Python, from the repo root.

    Returns what the script printed. If the script crashes, raises RuntimeError
    with the end of its error output, so the test failure shows the real cause.
    """
    cmd = [sys.executable, str(script), *map(str, args)]
    # PYTHONHASHSEED fixes Python's own hashing, as one more source of randomness removed
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "PYTHONHASHSEED": "0"})
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stderr[-3000:]}")
    return out.stdout


def write_config(tmp):
    """Copy the lab's default config into tmp, minus the `cuda = True` line.

    The tests run on CPU, and `--cuda` is an on-only switch, so once the config
    says `cuda = True` it can't be turned off from the command line (bug B8).
    Returns the path to the copy.
    """
    lines = (REPO / "configs" / "configs.txt").read_text().splitlines()
    cfg = Path(tmp) / "config.txt"
    cfg.write_text("\n".join(l for l in lines if not l.strip().startswith("cuda")) + "\n")
    return cfg


def build_model(tmp, cfg):
    """Copy the arm model into tmp and add the target bodies to it.

    This runs the lab's append_musculo_targets.py, which writes
    musculo_targets.xml next to the model. Returns the path to the copied
    musculoskeletal_model.xml.
    """
    model_dir = Path(tmp) / "model"
    model_dir.mkdir()
    src = REPO / "musculoskeletal_model"
    shutil.copy(src / "musculoskeletal_model.xml", model_dir)
    # The model refers to its mesh files by relative path, so link the folder in
    (model_dir / "Geometry").symlink_to(src / "Geometry")
    model = model_dir / "musculoskeletal_model.xml"
    run("append_musculo_targets.py", "--config", cfg, "--musculoskeletal_model_path", model)
    return model


def train(tmp, cfg, model, variant="default"):
    """Run a short seeded training for one of TRAIN_VARIANTS, on CPU.

    Uses the saved starting pose in golden/initial_pose, so the tests don't
    depend on the inverse-kinematics step (which needs a display).
    Returns the folder holding its stats/ and checkpoint/ output.
    """
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
    """Run the lab's shipped trained model (checkpoint/) in test mode, on CPU.

    Test mode runs every condition once with the trained network. Returns the
    test data muSim saves (muscle commands, RNN activity, paths, ...).
    """
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
    """A fingerprint (SHA-256 hash) of one or more numpy arrays.

    Changing even one number, the data type or the shape gives a completely
    different fingerprint. So "fingerprints match" means "every number matches",
    and we only need to store a short string instead of millions of numbers.
    """
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def load_stats(folder):
    """The per-episode training statistics a run saved, as {name: array}."""
    return {k: np.load(Path(folder) / "stats" / f"stats_{k}.npy") for k in STAT_NAMES}


def checkpoint_summary(path):
    """Summarise a checkpoint file: the episode it was saved at, plus a
    fingerprint of every network's weights (actor, critic, target critic)."""
    ck = torch.load(path, map_location="cpu")
    summary = {"iteration": int(ck["iteration"])}
    for name in NET_NAMES:
        sd = ck[name]
        # Sort the layer names so the fingerprint doesn't depend on their order
        summary[name] = sha256(*[sd[k].cpu().numpy() for k in sorted(sd)])
    return summary


def summarize_test(data):
    """Summarise test-mode output: a fingerprint of every saved array for each
    condition, plus the mean hand-target distance in cm per condition (a
    readable number, so a failure shows how much tracking changed)."""
    conds = sorted(data["emg"])
    summary = {"hashes": {k: {str(c): sha256(data[k][c]) for c in conds} for k in TEST_ARRAYS}}
    summary["mean_error_cm"] = {
        str(c): float(np.linalg.norm(data["kinematics_mbodies"][c] - data["kinematics_mtargets"][c],
                                     axis=-1).mean() * 100)
        for c in conds}
    return summary
