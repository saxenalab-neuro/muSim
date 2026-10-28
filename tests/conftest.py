"""Shared setup for the unit tests (test_env.py, test_actor.py, test_config_and_data.py).

pytest loads this file automatically before any test. It does three things:
  1. Imports the lab's modules the way main.py would (see below).
  2. Provides "fixtures": ready-made objects that tests ask for by naming
     them as arguments, e.g. `def test_x(env):` gets a fresh environment.
  3. Cleans up a log file MuJoCo may leave behind.

Why the import needs care: some muSim modules (kinematics_preprocessing_specs)
read the command line and load data files as soon as they are imported, using
paths relative to the repo root. So we move to the repo root, and point the
command line at the lab's default config only while those modules are imported.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

import musim_runner as mr

# Run from the repo root, because muSim opens files by relative path
os.chdir(mr.REPO)

# A copy of the lab's default config without `cuda = True` (tests run on CPU)
_tmp = Path(tempfile.mkdtemp(prefix="musim_unit_"))
CONFIG = mr.write_config(_tmp)

# Temporarily set the command line to "--config <default config>", import the
# modules that read it at import time, then restore the real command line
_argv = sys.argv
sys.argv = ["musim", "--config", str(CONFIG)]
import config  # noqa: E402
from SAC import kinematics_preprocessing_specs, reward_function_specs  # noqa: E402,F401
from SAC.RL_Framework_Mujoco import Muscle_Env  # noqa: E402
sys.argv = _argv


# MuJoCo writes warnings to this file in the repo root. Remember whether it was
# already there, so we only delete it if the tests created it.
_MUJOCO_LOG = mr.REPO / "MUJOCO_LOG.TXT"
_had_mujoco_log = _MUJOCO_LOG.exists()


def pytest_sessionfinish(session, exitstatus):
    """Runs once after all tests: remove the MuJoCo log file if we created it."""
    # The NaN scenarios in test_future.py make MuJoCo write warnings to this file
    if not _had_mujoco_log:
        _MUJOCO_LOG.unlink(missing_ok=True)


def parse_args(*extra):
    """The lab's default config, as main.py would parse it, plus any extra
    command-line settings (e.g. parse_args("--seed", 1))."""
    return config.config_parser().parse_args(["--config", str(CONFIG), *map(str, extra)])


@pytest.fixture(scope="session")
def model_xml(tmp_path_factory):
    """The arm model with targets (the saved golden copy), next to the mesh
    files it needs. Made once per test run."""
    d = tmp_path_factory.mktemp("model")
    shutil.copy(mr.GOLDEN / "musculo_targets.xml", d)
    # The model refers to its mesh files by relative path, so link the folder in
    (d / "Geometry").symlink_to(mr.REPO / "musculoskeletal_model" / "Geometry")
    return d / "musculo_targets.xml"


@pytest.fixture
def args():
    """The default settings, using the saved starting pose instead of running
    the inverse-kinematics step."""
    return parse_args("--initial_pose_path", mr.GOLDEN / "initial_pose")


@pytest.fixture
def env(model_xml, args):
    """A fresh environment for each test, built the way simulate.py builds it
    (the 1 is the frame skip simulate.py passes in)."""
    return Muscle_Env(str(model_xml), 1, args)
