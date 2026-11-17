"""Tests for the default values of optional settings in config.py.

Why this file exists:
    The visual settings `visual_feedback_bodies`, `visual_distance_bodies`
    and `visual_velocity` are optional. They used to default to None, but the
    environment (SAC/RL_Framework_Mujoco.py) counts their items with len(),
    so leaving one out of a config file crashed with
    "'NoneType' has no len()". They now default to an empty list, which means
    "no extra senses".

Run from the repository root:
    python -m pytest tests/test_config_defaults.py
"""

import os
import sys

# Make the repository root importable, so `import config` finds config.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import config  # noqa: E402  (import after the path is set up)

OPTIONAL_VISUAL_SETTINGS = [
    "visual_feedback_bodies",
    "visual_distance_bodies",
    "visual_velocity",
]


def test_optional_visual_settings_default_to_empty_list():
    """Left out entirely, each optional visual setting is an empty list.

    An empty list has a length of 0, so the environment's len() checks skip
    that sense instead of crashing.
    """
    args = config.config_parser().parse_args([])
    for name in OPTIONAL_VISUAL_SETTINGS:
        value = getattr(args, name)
        assert value == [], f"{name} should default to [], got {value!r}"
        assert len(value) == 0


def test_lab_config_still_sets_its_own_values():
    """The lab's own config file still overrides the new defaults.

    configs/configs.txt sets visual_distance_bodies to [[hand;target0]], so
    the change of default must not affect it.
    """
    lab_config = os.path.join(REPO_ROOT, "configs", "configs.txt")
    args = config.config_parser().parse_args(["--config", lab_config])
    assert args.visual_distance_bodies == [["hand", "target0"]]
