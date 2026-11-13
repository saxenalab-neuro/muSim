# muSim tests

## What these tests are for

muSim is being changed so it works with other body models, end effectors and
datasets (not only the monkey cycling task). Changing code can break things
without anyone noticing, so the first tests here were written **before** any
change, to record exactly how muSim behaved. They now run after every change,
to show that the cycling results stay the same.

There are five groups:

1. **Golden-output tests** (`test_golden.py`): does muSim still give exactly the
   same results as before? They re-run real training and compare every number
   against saved results.
2. **Unit tests** (`test_env.py`, `test_actor.py`, `test_config_and_data.py`):
   do the individual pieces (inputs, reward, episode ending, target movement,
   network, settings) still work as they did?
3. **Fix tests** (`test_config_defaults.py`, `test_checkpoint_loading.py`,
   `test_model_paths.py`, `test_episode_end.py`, `test_target_rows.py`): one
   small file per bug fix, checking that fix directly.
4. **Generalisation tests** (`test_generalisation.py`): setups other than
   cycling that muSim now handles, run end to end. They used to fail.
5. **Expected-failure tests** (`test_future.py`): things muSim can't do yet.
   They fail today, on purpose, for a known reason.

Every test file starts with a comment explaining what it checks and why.

## Setting up (once per computer)

You need **Python 3.10** and about 1 GB of disk space. No GPU is needed; the
tests run on the CPU. From the repo root:

```bash
python3.10 -m venv .venv                          # a separate Python environment for muSim
source .venv/bin/activate                         # use it (on Windows: .venv\Scripts\activate)
python -m pip install -r tests/requirements.txt   # the exact package versions the tests expect
```

The lab's `requirements.yml` and `nuSim_env.yml` pin older versions (Python 3.7,
torch 1.13, or the old `mujoco-py`) that don't install on current machines, so
the tests use `tests/requirements.txt` instead.

## Running the tests

From the repo root, with the environment active:

```bash
python -m pytest
```

This takes about 3 minutes. A healthy result ends with:

```
76 passed, 3 xfailed
```

"xfailed" means "failed as expected": those are the 3 expected-failure tests
in group 5, and the run still counts as a pass.

To run only the fast unit tests (under 1 second):

```bash
python -m pytest tests/test_env.py tests/test_actor.py tests/test_config_and_data.py
```

To run only the fix tests (about 10 seconds):

```bash
python -m pytest tests/test_config_defaults.py tests/test_checkpoint_loading.py tests/test_model_paths.py tests/test_episode_end.py tests/test_target_rows.py
```

Add `-v` to see each test's name and result.

## Words used in the tests

| Word | Meaning |
|---|---|
| golden output | Results saved from the original code, used as the reference ("the golden copy") |
| seed | The starting number for the random generator. With the same seed (123456), CPU training gives exactly the same numbers every run, which is what makes golden tests possible |
| fingerprint | A short SHA-256 code computed from a large array (e.g. network weights). If any number in the array changes, the fingerprint changes, so comparing fingerprints is as strict as comparing every number |
| xfail / xfailed | A test marked "expected to fail". pytest reports it as xfailed and the run still passes |
| XPASS(strict) | An expected-failure test that started passing (something was fixed). The run fails on purpose, as a reminder to remove the test's `xfail` marker and move it to `test_generalisation.py` |
| condition | One recorded movement (one trial type) in the dataset: 4 for training, 2 for testing |
| zone | How far the hand may be from the target before the episode ends. It shrinks from 6.4 cm to 0.8 cm as the episode goes on |

## What the IDs mean (H1, B1, ...)

Test comments refer to hardcoded values (**H**, fixed to the monkey cycling
task) and bugs (**B**) found while reading the code. "Where" gives the place
in the original code (commit `961e1cf`), before any fix.

| ID | What it is | Where | Status |
|---|---|---|---|
| H1 | The episode-end check looked up bodies literally named `"hand"` and `"target0"`, so only that one pair could end an episode | `SAC/RL_Framework_Mujoco.py`, `is_done()`, lines 360–361 | **Fixed:** `is_done()` uses the body/target pairs from the data |
| H2 | The condition cue is `(T - 600) / (1319 - 600)`; 600 and 1319 are the cycling data's trial lengths | `SAC/RL_Framework_Mujoco.py`, line 253 | Open |
| H5 | The network's muscle commands are 0 to 1 regardless of the model's control range (0.002 to 1); MuJoCo clamps them | `SAC/sac.py` line 42 passes `action_space=None` to the actor (`SAC/model.py` lines 37–44) | Open (left for later) |
| H6 | The targets-model path was built by cutting `len("musculoskeletal_model.xml")` characters off the model path, so other file names broke | `simulate.py` lines 98–101, also `find_init_pose.py`, `find_init_pose_ik_cma.py`, `visualize_trajectories_ik.py`, `append_musculo_targets.py` line 48 | **Fixed:** `config.get_musculo_targets_path()` |
| H7 | The target's start position (0.1, 0.1, 0.85) and the mapping "data y → MuJoCo z (up)" are fixed | `append_musculo_targets.py` lines 30 and 37 | Open |
| B1 | With no neural data, the kinematics are used as a placeholder and then normalised **in place**, so the real hand path becomes all NaN and every episode ends after 2 steps, with no error | `SAC/RL_Framework_Mujoco.py`, lines 93–104 | Open |
| B2 | The data row for each target joint was chosen by counting tracked axes, which only worked when the locked axis was the last one (z) | `SAC/RL_Framework_Mujoco.py`, lines 585–591 | **Fixed:** x, y, z read rows 0, 1, 2 |
| B6 | `visual_velocity` and `visual_distance_bodies` defaulted to `None`, but the code calls `len()` on them, so leaving them out of a config crashed | `config.py` lines 243–253; `SAC/RL_Framework_Mujoco.py` line 384 | **Fixed:** they default to an empty list |
| B8 | The shipped checkpoint was saved on a GPU and `torch.load` had no `map_location`, so it couldn't be loaded on a CPU-only machine; `cuda = True` also crashed there | `simulate.py` lines 514–543; `SAC/sac.py` lines 29–32 | **Fixed:** `map_location`, and a CPU fallback when no GPU is found |

## Unit tests

Small checks on individual pieces. They run in-process and pin the cycling setup's behaviour:

| File | Checks |
|---|---|
| `test_env.py` | What the network senses: 81 values from the environment (39 muscle lengths, 39 velocities, tanh(100 × hand–target gap)), plus the condition cue, for 82 inputs. The reward per coordinate, ×5, and the −25 penalty. The zone schedule (6.4 → 3.2 → 1.6 → 0.8 cm at steps 26, 100 and 150). When an episode ends. How the target follows the recorded path, including data y → MuJoCo z |
| `test_actor.py` | Muscle commands stay in [0, 1], the exact output mapping (0.5·tanh + 0.5), the model's control range, and that the shipped checkpoint fits an 82 → 256 → 39 network |
| `test_config_and_data.py` | Every value in `configs/configs.txt` as `main.py` parses it, constants that are still hardcoded today, data shapes (1 marker, z locked, 4 train and 2 test conditions, 49 neurons), and scaling and centring of the kinematics |

When a setting moves from the code into the config, its test should keep the same expected value.

## Fix tests

One small file per bug fix. Each test fails on the code before that fix and passes after it:

| File | Fix | Checks |
|---|---|---|
| `test_config_defaults.py` | B6 | The optional visual settings default to an empty list, and the lab's config still sets its own values |
| `test_checkpoint_loading.py` | B8 | `cuda = True` with no GPU falls back to the CPU with a warning, and the lab's GPU-saved checkpoint loads with every weight matching |
| `test_model_paths.py` | H6 | The model-with-targets file goes next to the model (e.g. `mouse/arm.xml` → `mouse/musculo_targets.xml`), or where `musculo_targets_path` says |
| `test_episode_end.py` | H1 | A tracked body with any name works, and any tracked body leaving its zone ends the episode |
| `test_target_rows.py` | B2 | The target's x, y and z read data rows 0, 1 and 2, also when x is unused |

## Generalisation tests (`test_generalisation.py`)

Setups other than cycling that muSim now handles. Each one builds a small custom setup (dataset, model, config) in a temp folder with `scenarios.py` and runs the real muSim code on it. They were expected-failure tests until the fix named below:

| Test | What it checks | Used to fail because |
|---|---|---|
| `test_end_effector_with_any_name` | Train an arm whose tracking body isn't called `hand` | Episode-end check looked up `"hand"` by name (H1) |
| `test_every_marker_can_end_episode` | With 2 markers, either one leaving its zone ends the episode | Only marker 0 was checked; marker 1 got the −25 penalty but the episode continued (H1) |
| `test_dataset_with_x_locked` | Track y and z with x locked | The target's y joint got the NaN x row (B2) |
| `test_config_can_leave_out_optional_feedback` | A config can leave out `visual_velocity` / `visual_distance_bodies` | Their defaults were `None`, and the code calls `len()` on them (B6) |
| `test_model_file_with_any_name` | A model file not named `musculoskeletal_model.xml` | The path was built by cutting off that name's length (H6) |

## Expected-failure tests (`test_future.py`)

What muSim should handle next. Each one builds a small custom setup with `scenarios.py`, and **fails on today's code**:

| Test | What it wants | Fails today because |
|---|---|---|
| `test_dataset_without_neural_data_keeps_kinematics` | A dataset with no neural recordings loads correctly | The kinematics are "normalised" as if they were neural data and become all NaN (B1) |
| `test_training_without_neural_data` | Training on it behaves normally | Every episode ends after 2 steps (B1) |
| `test_condition_cue_scales_to_dataset` | Condition cue stays in [0, 1] for other trial lengths | Uses the cycling data's 600 and 1319, so shorter trials give negative cues (H2) |

They are marked `xfail(strict=True, raises=AssertionError)`: muSim has to run and give the wrong result for it to count as the expected failure, so a broken test setup still shows up as a real failure. `test_scenario_builder_on_cycling_data` is an ordinary test that checks the builder works on the unchanged cycling setup.

**When a fix makes one pass**, pytest reports `XPASS(strict)` and the run fails. Remove that test's `xfail` marker and move it to `test_generalisation.py`, so it becomes a normal test.

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
