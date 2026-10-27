# muSim tests

## Golden-output tests (`test_golden.py`)

"Nothing changed" checks for refactoring. Each one runs the real pipeline and compares it with saved output in `golden/`:

| Test | What it runs | Compared exactly |
|---|---|---|
| `test_append_targets_matches_golden` | `append_musculo_targets.py` | The generated `musculo_targets.xml` |
| `test_training_stats_match_golden` | 30 seeded training episodes (seed 123456) | Reward, steps, policy loss and critic loss for every episode |
| `test_trained_weights_match_golden` | Same run | Saved actor, critic and target-critic weights (checkpoint at episode 20 and best checkpoint) |
| `test_pretrained_*` | The shipped `checkpoint/` in test mode, all 6 conditions | Muscle commands, RNN activity and inputs, hand and target paths, and mean hand–target distance per condition |

Run from the repo root (about 30 s on a laptop CPU):

```bash
python -m pytest
```

Everything runs on CPU in a temporary folder, so the repo's own model, checkpoint and statistics files are not touched. `golden/initial_pose/` is a fixed copy of the starting pose found by `find_init_pose.py`, so these tests don't depend on the IK step, which needs a display.

### When a test fails
- If the change was **not** meant to change behaviour, it's a regression. The message shows the first episode that differs.
- If it **was** meant to (for example, a bug fix), regenerate the golden output and say so in the commit: `python tests/make_golden.py`

### Limits
Exact comparison works because seeded CPU training is bit-for-bit repeatable on the same machine and library versions. The golden files were made on macOS (Apple silicon), Python 3.10, torch 2.14.1, mujoco 3.3.3 and numpy 1.26.4. Another machine, library version or a GPU may differ in the last digits. Those setups would need tolerances instead.
