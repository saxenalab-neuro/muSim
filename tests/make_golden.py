"""Regenerate the saved "expected" results in tests/golden/.

The golden-output tests (test_golden.py) compare every run against these
files. Only regenerate them when a change is *meant* to alter muSim's
behaviour (for example, a bug fix), and say so in the commit message, so
reviewers know the expected results changed on purpose.

Usage, from the repo root:  python tests/make_golden.py   (about 2 minutes)
"""
import json
import shutil
import tempfile

import numpy as np

import musim_runner as mr


def main():
    # Work in a temporary folder that is deleted afterwards
    with tempfile.TemporaryDirectory() as tmp:
        # 1. The arm model with target bodies added
        cfg = mr.write_config(tmp)
        model = mr.build_model(tmp, cfg)
        shutil.copy(model.parent / "musculo_targets.xml", mr.GOLDEN / "musculo_targets.xml")

        # 2. One short training run per setup: save the per-episode statistics,
        #    and fingerprints of the networks in both checkpoint files
        for variant in mr.TRAIN_VARIANTS:
            run = mr.train(tmp, cfg, model, variant)
            np.savez(mr.GOLDEN / f"train_{variant}_stats.npz", **mr.load_stats(run))
            nets = {f: mr.checkpoint_summary(run / "checkpoint" / f"{f}.pth")
                    for f in ["agent_networks", "agent_networks_best"]}
            (mr.GOLDEN / f"train_{variant}_checkpoints.json").write_text(json.dumps(nets, indent=2) + "\n")

        # 3. The lab's trained model in test mode: fingerprints and tracking error
        summary = mr.summarize_test(mr.run_pretrained(tmp, cfg, model))
        (mr.GOLDEN / "pretrained_test.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote golden output to {mr.GOLDEN}")


if __name__ == "__main__":
    main()
