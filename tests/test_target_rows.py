"""Tests for which data row moves the target ball in each direction.

Why this file exists:
    The kinematics data has 3 rows per target: row 0 = x, row 1 = y, row 2 = z.
    An unused direction is left blank (NaN). Muscle_Env.upd_theta() moves the
    ball by copying each row into the ball's x, y and z joints.

    y and z used to pick their row by counting how many directions before them
    were used. That only worked when x was used: with x blank, y read row 0
    (the blank x row) and z read row 1 (the y data). Now x, y and z always read
    rows 0, 1 and 2.

These tests call upd_theta() with a small stand-in for the environment, so no
MuJoCo model has to be built.

Run from the repository root:
    python -m pytest tests/test_target_rows.py
"""

import os
import sys
import types

import numpy as np
import pytest

# Make the repository root importable, so `import SAC` works.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SAC import kinematics_preprocessing_specs  # noqa: E402
from SAC.RL_Framework_Mujoco import Muscle_Env  # noqa: E402

# Easy-to-spot values for each row of the data, at every time point.
X, Y, Z = 0.11, 0.22, 0.33


def run_upd_theta(monkeypatch, rows, used):
    """Move the ball once using data `rows` ([x, y, z] values; None = blank).

    `used` says which directions the data uses, e.g. [0, 1, 1] for y and z.
    Returns the ball's joint positions as {"x": ..., "y": ..., "z": ...},
    for the joints that exist.
    """
    monkeypatch.setattr(kinematics_preprocessing_specs, "xyz_target", [used])

    timepoints = 10
    data = np.array([[np.full(timepoints, np.nan if v is None else v) for v in rows]])
    joints = {"box:x0": 0, "box:y0": 1, "box:z0": 2}   # joint name -> position slot
    result = {}

    env = types.SimpleNamespace(
        istep=0, n_fixedsteps=25, _max_episode_steps=25 + timepoints,
        kin_to_sim={0: data}, current_cond_to_sim=0,
        data=types.SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3)),
        model=types.SimpleNamespace(joint=lambda name: types.SimpleNamespace(qposadr=joints[name])),
        set_state=lambda qpos, qvel: result.update(qpos=qpos.copy()),
    )
    Muscle_Env.upd_theta(env)
    qpos = result["qpos"]
    return {axis: qpos[i] for i, axis in enumerate("xyz") if used[i]}


def test_x_unused_y_and_z_read_their_own_rows(monkeypatch):
    """Data with x blank: the ball's y gets row 1 and z gets row 2."""
    ball = run_upd_theta(monkeypatch, rows=[None, Y, Z], used=[0, 1, 1])
    assert ball == pytest.approx({"y": Y, "z": Z})


def test_all_three_directions(monkeypatch):
    """Data using x, y and z: each joint gets its own row."""
    ball = run_upd_theta(monkeypatch, rows=[X, Y, Z], used=[1, 1, 1])
    assert ball == pytest.approx({"x": X, "y": Y, "z": Z})


def test_lab_cycling_layout_is_unchanged(monkeypatch):
    """The monkey data uses x and y, with z blank: same rows as before the fix."""
    ball = run_upd_theta(monkeypatch, rows=[X, Y, None], used=[1, 1, 0])
    assert ball == pytest.approx({"x": X, "y": Y})
