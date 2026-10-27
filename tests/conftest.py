"""Shared fixtures for the in-process unit tests.

Some muSim modules read the command line and data files as soon as they are
imported (kinematics_preprocessing_specs), with paths relative to the repo
root. So tests run from the repo root, and the command line is pointed at the
lab's default config while those modules are imported.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

import musim_runner as mr

os.chdir(mr.REPO)

_tmp = Path(tempfile.mkdtemp(prefix="musim_unit_"))
CONFIG = mr.write_config(_tmp)

_argv = sys.argv
sys.argv = ["musim", "--config", str(CONFIG)]
import config  # noqa: E402
from SAC import kinematics_preprocessing_specs, reward_function_specs  # noqa: E402,F401
from SAC.RL_Framework_Mujoco import Muscle_Env  # noqa: E402
sys.argv = _argv


_MUJOCO_LOG = mr.REPO / "MUJOCO_LOG.TXT"
_had_mujoco_log = _MUJOCO_LOG.exists()


def pytest_sessionfinish(session, exitstatus):
    # The NaN scenarios in test_future.py make MuJoCo write warnings to this file
    if not _had_mujoco_log:
        _MUJOCO_LOG.unlink(missing_ok=True)


def parse_args(*extra):
    """The lab's default config, as main.py would parse it."""
    return config.config_parser().parse_args(["--config", str(CONFIG), *map(str, extra)])


@pytest.fixture(scope="session")
def model_xml(tmp_path_factory):
    """The golden arm model with targets, next to the mesh files it needs."""
    d = tmp_path_factory.mktemp("model")
    shutil.copy(mr.GOLDEN / "musculo_targets.xml", d)
    (d / "Geometry").symlink_to(mr.REPO / "musculoskeletal_model" / "Geometry")
    return d / "musculo_targets.xml"


@pytest.fixture
def args():
    return parse_args("--initial_pose_path", mr.GOLDEN / "initial_pose")


@pytest.fixture
def env(model_xml, args):
    """A fresh environment, built the way simulate.py builds it."""
    return Muscle_Env(str(model_xml), 1, args)
