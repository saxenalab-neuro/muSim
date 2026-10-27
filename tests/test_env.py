"""The environment: what the network senses, the reward, when an episode
ends, and how the target follows the recorded hand path."""
from types import SimpleNamespace

import numpy as np
import pytest

from SAC import reward_function_specs

N_MUSCLES = 39
REST = np.zeros(N_MUSCLES)


def body_pos(env, name):
    return env.data.xpos[env.model.body(name).id].copy()


# --- What the network senses -------------------------------------------------

def test_observation_size(env):
    ob = env.reset(0)
    assert env.model.nu == N_MUSCLES
    assert len(ob) == 81 and env.observation_space.shape == (81,)
    # simulate.py adds visual velocities and the condition cue: 82 network inputs
    assert env.observation_space.shape[0] + len(env.sfs_visual_velocity) * 3 + 1 == 82


def test_observation_layout(env):
    env.reset(0)
    ob, *_ = env.step(np.full(N_MUSCLES, 0.3))
    ob = np.asarray(ob)
    np.testing.assert_array_equal(ob[:39], env.data.actuator_length)
    np.testing.assert_array_equal(ob[39:78], env.data.actuator_velocity)
    gap = body_pos(env, "hand") - body_pos(env, "target0")
    np.testing.assert_array_equal(ob[78:81], np.tanh(gap * 100))


@pytest.mark.parametrize("cond,T", [(0, 702), (1, 803), (2, 932), (3, 1319)])
def test_condition_cue(env, cond, T):
    env.reset(cond)
    assert env.condition_scalar == (T - 600) / (1319 - 600)


# --- Reward --------------------------------------------------------------------

def fake_sim(hand, target):
    ids = {"hand": 0, "target0": 1}
    data = SimpleNamespace(xpos=np.array([hand, target], dtype=float))
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id=ids[name]))
    return data, model


def test_reward_max_on_target():
    data, model = fake_sim([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert reward_function_specs.reward_function(0, data, model, REST, 0.008) == 3.0


def test_reward_at_known_distance():
    data, model = fake_sim([0.105, 0.2, 0.3], [0.1, 0.2, 0.3])   # 5 mm off in x
    r = reward_function_specs.reward_function(0, data, model, REST, 0.008)
    assert r == pytest.approx(1000 ** -0.005 + 2)


def test_reward_penalty_outside_zone():
    data, model = fake_sim([0.11, 0.2, 0.3], [0.1, 0.2, 0.3])    # 1 cm off, zone 0.8 cm
    assert reward_function_specs.reward_function(0, data, model, REST, 0.008) == -5


def test_step_reward_is_five_times_reward_function(env, monkeypatch):
    monkeypatch.setattr(reward_function_specs, "reward_function", lambda *a: 0.7)
    env.reset(0)
    _, r, _, _ = env.step(REST)
    assert r == pytest.approx(3.5)


def test_first_step_near_max_reward(env):
    # The starting pose puts the hand on the target, so step 1 earns almost 15
    env.reset(0)
    _, r, done, _ = env.step(REST)
    assert 14.5 < r <= 15 and not done


# --- When an episode ends ------------------------------------------------------

@pytest.mark.parametrize("step,zone", [(1, 0.064), (25, 0.064), (26, 0.032), (99, 0.032),
                                       (100, 0.016), (149, 0.016), (150, 0.008), (5000, 0.008)])
def test_zone_schedule(env, step, zone):
    env.reset(0)
    env.istep = step - 1
    env.step(REST)
    assert env.threshold == zone


def test_leaving_zone_ends_episode_with_penalty(env):
    env.reset(0)
    env.threshold = 1e-6          # not overwritten during the first 25 steps
    _, r, done, _ = env.step(REST)
    assert done and r == -25


def test_inside_zone_continues(env):
    env.reset(0)
    env.threshold = 1.0
    assert not env.is_done()


def test_time_limit_ends_episode(env):
    env.reset(0)
    env.threshold = 1.0
    env.istep = env.timestep_limit
    assert env.is_done()


# --- Target follows the recorded path -----------------------------------------

def test_target_follows_kinematics(env):
    cond = 1
    env.reset(cond)
    kin = env.kin_to_sim[cond]
    T = kin.shape[-1]
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
    # Data x moves the target along MuJoCo x, data y along MuJoCo z (H7)
    env.reset(0)
    env.step(REST)
    kin = env.kin_to_sim[0]
    np.testing.assert_allclose(body_pos(env, "target0"),
                               [0.1 + kin[0, 0, 0], 0.1, 0.85 + kin[0, 1, 0]], atol=1e-12)
