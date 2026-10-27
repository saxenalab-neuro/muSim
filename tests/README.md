# muSim tests

Run everything from the repo root (about 35 s on a laptop CPU):

```bash
python -m pytest
```

To run just the fast unit tests (under 1 s): `python -m pytest --ignore tests/test_golden.py`

## Unit tests

Small checks on the pieces the refactor will move. They run in-process and pin today's behaviour:

| File | Checks |
|---|---|
| `test_env.py` | What the network senses: 81 values from the environment (39 muscle lengths, 39 velocities, tanh(100 × hand–target gap)), plus the condition cue, for 82 inputs. The reward per coordinate, ×5, and the −25 penalty. The zone schedule (6.4 → 3.2 → 1.6 → 0.8 cm at steps 26, 100 and 150). When an episode ends. How the target follows the recorded path, including data y → MuJoCo z |
| `test_actor.py` | Muscle commands stay in [0, 1], the exact output mapping (0.5·tanh + 0.5), the model's control range, and that the shipped checkpoint fits an 82 → 256 → 39 network |
| `test_config_and_data.py` | Every value in `configs/configs.txt` as `main.py` parses it, constants that are still hardcoded today, data shapes (1 marker, z locked, 4 train and 2 test conditions, 49 neurons), and scaling and centring of the kinematics |

When a setting moves from the code into the config, its test should keep the same expected value.

## Golden-output tests (`test_golden.py`)

"Nothing changed" checks for refactoring. Each one runs the real pipeline and compares it with saved output in `golden/`:

| Test | What it runs | Compared exactly |
|---|---|---|
| `test_append_targets_matches_golden` | `append_musculo_targets.py` | The generated `musculo_targets.xml` |
| `test_training_stats_match_golden` | 30 seeded training episodes (seed 123456) | Reward, steps, policy loss and critic loss for every episode |
| `test_trained_weights_match_golden` | Same run | Saved actor, critic and target-critic weights (checkpoint at episode 20 and best checkpoint) |
| `test_pretrained_*` | The shipped `checkpoint/` in test mode, all 6 conditions | Muscle commands, RNN activity and inputs, hand and target paths, and mean hand–target distance per condition |

Everything runs on CPU in a temporary folder, so the repo's own model, checkpoint and statistics files are not touched. `golden/initial_pose/` is a fixed copy of the starting pose found by `find_init_pose.py`, so these tests don't depend on the IK step, which needs a display.

### When a test fails
- If the change was **not** meant to change behaviour, it's a regression. The message shows the first episode that differs.
- If it **was** meant to (for example, a bug fix), regenerate the golden output and say so in the commit: `python tests/make_golden.py`

### Limits
Exact comparison works because seeded CPU training is bit-for-bit repeatable on the same machine and library versions. The golden files were made on macOS (Apple silicon), Python 3.10, torch 2.14.1, mujoco 3.3.3 and numpy 1.26.4. Another machine, library version or a GPU may differ in the last digits. Those setups would need tolerances instead.
