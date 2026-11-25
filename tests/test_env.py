"""Unit tests for the environment (SAC/RL_Framework_Mujoco.py and SAC/reward_function_specs.py).

They pin down, with exact numbers, how muSim behaves today:
  - what the network senses (its inputs)
  - the reward
  - when an episode ends (the shrinking zone around the target)
  - how the target follows the recorded hand path

These are the pieces items 1 and 2 of the refactor will move into the config.
After the move, these tests must still pass with the same expected values,
unless a change is meant to alter behaviour.

The `env` argument in many tests is a fresh environment from conftest.py.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from SAC import reward_function_specs

# The monkey arm model has 39 muscles, so the network sends 39 commands
N_MUSCLES = 39
# All muscle commands at 0: the arm is not actively driven
REST = np.zeros(N_MUSCLES)


def body_pos(env, name):
    """Current x, y, z position (metres) of a named body in the simulation."""
    return env.data.xpos[env.model.body(name).id].copy()


# --- What the network senses -------------------------------------------------

def test_observation_size(env):
    """The environment gives 81 values: 39 muscle lengths, 39 muscle velocities,
    and 3 for the hand-target gap. simulate.py then adds 3 per body in
    `visual_velocity` (none by default) and 1 condition cue, for 82 network inputs."""
    ob = env.reset(0)
    assert env.model.nu == N_MUSCLES
    assert len(ob) == 81 and env.observation_space.shape == (81,)
    # simulate.py adds visual velocities and the condition cue: 82 network inputs
    assert env.observation_space.shape[0] + len(env.sfs_visual_velocity) * 3 + 1 == 82


def test_observation_layout(env):
    """The order of the 81 values, and how the gap is squashed.

    [0:39] muscle lengths, [39:78] muscle velocities, [78:81] tanh(100 x gap),
    where gap = hand position - target position. The tanh keeps each value in
    -1..1, and the x100 makes it sensitive at the cm scale.
    """
    env.reset(0)
    # One step with a moderate command (0.3) so lengths and velocities are non-zero
    ob, *_ = env.step(np.full(N_MUSCLES, 0.3))
    ob = np.asarray(ob)
    np.testing.assert_array_equal(ob[:39], env.data.actuator_length)
    np.testing.assert_array_equal(ob[39:78], env.data.actuator_velocity)
    gap = body_pos(env, "hand") - body_pos(env, "target0")
    np.testing.assert_array_equal(ob[78:81], np.tanh(gap * 100))


# The 4 training conditions and their lengths T (number of data samples)
@pytest.mark.parametrize("cond,T", [(0, 702), (1, 803), (2, 932), (3, 1319)])
def test_condition_cue(env, cond, T):
    """The condition cue tells the network which condition it is in.

    It's (T - 600) / (1319 - 600). The 600 and 1319 are hardcoded and suit
    the cycling data only (H2; test_future.py shows other lengths give
    negative cues).
    """
    env.reset(cond)
    assert env.condition_scalar == (T - 600) / (1319 - 600)


# --- Reward --------------------------------------------------------------------

def fake_sim(hand, target):
    """A stand-in for the simulator with just a hand and a target at chosen
    positions. Lets us call the reward function with exact distances."""
    ids = {"hand": 0, "target0": 1}
    data = SimpleNamespace(xpos=np.array([hand, target], dtype=float))
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id=ids[name]))
    return data, model


def test_reward_max_on_target():
    """Hand exactly on the target: each of x, y, z scores 1/1000^0 = 1, so 3 in total."""
    data, model = fake_sim([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    # 0.008 m is the smallest zone (0.8 cm); the 0 is an unused placeholder for delayed senses
    assert reward_function_specs.reward_function(0, data, model, REST, 0.008) == 3.0


def test_reward_at_known_distance():
    """5 mm off in x: x scores 1/1000^0.005 (about 0.966), y and z score 1 each.

    So the reward drops only gently with distance. The pressure to be accurate
    comes from the shrinking zone that ends the episode, not from the reward.
    """
    data, model = fake_sim([0.105, 0.2, 0.3], [0.1, 0.2, 0.3])   # 5 mm off in x
    r = reward_function_specs.reward_function(0, data, model, REST, 0.008)
    assert r == pytest.approx(1000 ** -0.005 + 2)


def test_reward_penalty_outside_zone():
    """If the hand is outside the zone in any of x, y, z, the reward is a flat -5."""
    data, model = fake_sim([0.11, 0.2, 0.3], [0.1, 0.2, 0.3])    # 1 cm off, zone 0.8 cm
    assert reward_function_specs.reward_function(0, data, model, REST, 0.008) == -5


def test_step_reward_is_five_times_reward_function(env, monkeypatch):
    """env.step() multiplies the reward function's value by 5 (hardcoded in step()).

    We temporarily replace the reward function with one that always returns
    0.7, so the step reward must be 3.5.
    """
    monkeypatch.setattr(reward_function_specs, "reward_function", lambda *a: 0.7)
    env.reset(0)
    _, r, _, _ = env.step(REST)
    assert r == pytest.approx(3.5)


def test_first_step_near_max_reward(env):
    """The saved starting pose puts the hand on the target, so step 1 earns
    close to the maximum of 5 x 3 = 15, and the episode doesn't end."""
    # The starting pose puts the hand on the target, so step 1 earns almost 15
    env.reset(0)
    _, r, done, _ = env.step(REST)
    assert 14.5 < r <= 15 and not done


# --- When an episode ends ------------------------------------------------------

# Zone half-width (metres) at each step: 6.4 cm, then 3.2, 1.6 and 0.8 cm.
# The edges of each band are checked (e.g. 25 and 26) to pin the exact switch points.
@pytest.mark.parametrize("step,zone", [(1, 0.064), (25, 0.064), (26, 0.032), (99, 0.032),
                                       (100, 0.016), (149, 0.016), (150, 0.008), (5000, 0.008)])
def test_zone_schedule(env, step, zone):
    """The zone around the target shrinks as the episode goes on.

    Steps 1-25 keep the starting zone (6.4 cm), then it shrinks to 3.2 cm, at
    step 100 to 1.6 cm, and at step 150 to 0.8 cm for the rest of the episode.
    """
    env.reset(0)
    env.istep = step - 1      # jump ahead so the next step is `step`
    env.step(REST)
    assert env.threshold == zone


def test_leaving_zone_ends_episode_with_penalty(env):
    """Leaving the zone ends the episode, with reward 5 x -5 = -25."""
    env.reset(0)
    # Shrink the zone to almost nothing so the hand is outside it.
    # Safe to set by hand: step() doesn't change the zone during the first 25 steps.
    env.threshold = 1e-6          # not overwritten during the first 25 steps
    _, r, done, _ = env.step(REST)
    assert done and r == -25


def test_inside_zone_continues(env):
    """With the hand inside the zone (here 1 m wide), the episode continues."""
    env.reset(0)
    env.threshold = 1.0
    assert not env.is_done()


def test_time_limit_ends_episode(env):
    """At the time limit (timestep_limit in the config) the episode ends,
    even with the hand inside the zone."""
    env.reset(0)
    env.threshold = 1.0
    env.istep = env.timestep_limit
    assert env.is_done()


# --- Target follows the recorded path -----------------------------------------

def test_target_follows_kinematics(env):
    """The target moves along the recorded hand path, one data sample at a time.

    For the first 25 steps (n_fixedsteps) it holds at the first sample. After
    that, step s shows sample int((T - 1) / T x (s - 25)), so it moves through
    the whole recording.
    """
    cond = 1
    env.reset(cond)
    kin = env.kin_to_sim[cond]          # [marker, x/y/z, time] after scaling and centring
    T = kin.shape[-1]
    # Where the target's x and y joints are stored in the simulator state
    x0 = env.model.joint("box:x0").qposadr[0]
    y0 = env.model.joint("box:y0").qposadr[0]

    env.step(REST)                                   # held still for the first 25 steps
    assert env.tpoint_to_sim == 0
    assert env.data.qpos[x0] == kin[0, 0, 0] and env.data.qpos[y0] == kin[0, 1, 0]

    env.istep = 124
    env.step(REST)                                   # step 125: 100 steps into the path
    t = int((T - 1) / T * 100)
    assert env.tpoint_to_sim == t
    assert env.data.qpos[x0] == kin[0, 0, t] and env.data.qpos[y0] == kin[0, 1, t]


def test_target_axis_mapping(env):
    """Where the data puts the target in the simulator.

    The target starts at [0.1, 0.1, 0.85] m. Data x moves it along MuJoCo x,
    and data y moves it along MuJoCo z (up). Both are hardcoded (H7).
    """
    # Data x moves the target along MuJoCo x, data y along MuJoCo z (H7)
    env.reset(0)
    env.step(REST)
    kin = env.kin_to_sim[0]
    np.testing.assert_allclose(body_pos(env, "target0"),
                               [0.1 + kin[0, 0, 0], 0.1, 0.85 + kin[0, 1, 0]], atol=1e-12)
