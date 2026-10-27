# muSim tests

Run everything from the repo root (about 2 minutes on a laptop CPU):

```bash
python -m pytest
```

To run just the fast unit tests (under 1 s): `python -m pytest tests/test_env.py tests/test_actor.py tests/test_config_and_data.py`

## Unit tests

Small checks on the pieces the refactor will move. They run in-process and pin today's behaviour:

| File | Checks |
|---|---|
| `test_env.py` | What the network senses: 81 values from the environment (39 muscle lengths, 39 velocities, tanh(100 × hand–target gap)), plus the condition cue, for 82 inputs. The reward per coordinate, ×5, and the −25 penalty. The zone schedule (6.4 → 3.2 → 1.6 → 0.8 cm at steps 26, 100 and 150). When an episode ends. How the target follows the recorded path, including data y → MuJoCo z |
| `test_actor.py` | Muscle commands stay in [0, 1], the exact output mapping (0.5·tanh + 0.5), the model's control range, and that the shipped checkpoint fits an 82 → 256 → 39 network |
| `test_config_and_data.py` | Every value in `configs/configs.txt` as `main.py` parses it, constants that are still hardcoded today, data shapes (1 marker, z locked, 4 train and 2 test conditions, 49 neurons), and scaling and centring of the kinematics |

When a setting moves from the code into the config, its test should keep the same expected value.

## Future-feature tests (`test_future.py`)

What the refactor should make possible. Each one builds a small custom setup (dataset, model, config) in a temp folder with `scenarios.py`, and **fails on today's code**:

| Test | What it wants | Fails today because |
|---|---|---|
| `test_end_effector_with_any_name` | Train an arm whose tracking body isn't called `hand` | Episode-end check looks up `"hand"` by name (H1) |
| `test_every_marker_can_end_episode` | With 2 markers, either one leaving its zone ends the episode | Only marker 0 is checked; marker 1 gets the −25 penalty but the episode continues (H1) |
| `test_dataset_without_neural_data_keeps_kinematics` | A dataset with no neural recordings loads correctly | The kinematics are "normalised" as if they were neural data and become all NaN (B1) |
| `test_training_without_neural_data` | Training on it behaves normally | Every episode ends after 2 steps (B1) |
| `test_dataset_with_x_locked` | Track y and z with x locked | The target's y joint gets the NaN x row (B2) |
| `test_condition_cue_scales_to_dataset` | Condition cue stays in [0, 1] for other trial lengths | Uses the cycling data's 600 and 1319, so shorter trials give negative cues (H2) |
| `test_config_can_leave_out_optional_feedback` | A config can leave out `visual_velocity` / `visual_distance_bodies` | Their defaults are `None`, and the code calls `len()` on them (B6) |
| `test_model_file_with_any_name` | A model file not named `musculoskeletal_model.xml` | The path is built by cutting off that name's length (H6) |

They are marked `xfail(strict=True)`. A crash only counts as the expected failure if its error matches the known cause, so a broken test setup still shows up as a real failure. `test_scenario_builder_on_cycling_data` is an ordinary test that checks the builder works on the unchanged cycling setup.

**When a fix makes one pass**, pytest reports `XPASS(strict)` and the run fails. Remove that test's `xfail` marker so it becomes a normal test.

## Golden-output tests (`test_golden.py`)

"Nothing changed" checks for refactoring. Each one runs the real pipeline and compares it with saved output in `golden/`:

| Test | What it runs | Compared exactly |
|---|---|---|
| `test_append_targets_matches_golden` | `append_musculo_targets.py` | The generated `musculo_targets.xml` |
| `test_training_stats_match_golden` | 30 seeded training episodes (seed 123456) for each of 4 setups (below) | Reward, steps, policy loss and critic loss for every episode |
| `test_trained_weights_match_golden` | Same runs | Saved actor, critic and target-critic weights (checkpoint at episode 20 and best checkpoint) |
| `test_pretrained_*` | The shipped `checkpoint/` in test mode, all 6 conditions | Muscle commands, RNN activity and inputs, hand and target paths, and mean hand–target distance per condition |

The 4 training setups (`TRAIN_VARIANTS` in `musim_runner.py`) cover options the lab's default config leaves off:

| Setup | Switched on | Protects |
|---|---|---|
| `default` | Nothing extra (configs/configs.txt) | The standard uSim run |
| `nusim` | `zeta_nusim = 10000` (the weighting recommended in `SAC/sac.py`) | nuSim: matching RNN units to recorded neurons |
| `senses` | Joint angles and velocities, muscle forces, body positions | The extra sensory inputs |
| `velocity_delay` | Hand and target velocity, 3-step sensory delay | Visual velocity and delayed feedback |

Each extra setup catches breaks that the default run misses. That was checked by changing each feature by 0.1% (or the delay by one step): only that setup's test failed. The stimulus input isn't covered, because the repo has no stimulus data.

Everything runs on CPU in a temporary folder, so the repo's own model, checkpoint and statistics files are not touched. `golden/initial_pose/` is a fixed copy of the starting pose found by `find_init_pose.py`, so these tests don't depend on the IK step, which needs a display.

### When a test fails
- If the change was **not** meant to change behaviour, it's a regression. The message shows the first episode that differs.
- If it **was** meant to (for example, a bug fix), regenerate the golden output and say so in the commit: `python tests/make_golden.py`

### Limits
Exact comparison works because seeded CPU training is bit-for-bit repeatable on the same machine and library versions. The golden files were made on macOS (Apple silicon), Python 3.10, torch 2.14.1, mujoco 3.3.3 and numpy 1.26.4. Another machine, library version or a GPU may differ in the last digits. Those setups would need tolerances instead.
