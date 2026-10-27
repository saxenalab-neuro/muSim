"""Regenerate the golden output in tests/golden/.

Only run this when a change is *meant* to alter muSim's behaviour, and say so
in the commit. Usage, from the repo root:  python tests/make_golden.py
"""
import json
import shutil
import tempfile

import numpy as np

import musim_runner as mr


def main():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = mr.write_config(tmp)
        model = mr.build_model(tmp, cfg)
        shutil.copy(model.parent / "musculo_targets.xml", mr.GOLDEN / "musculo_targets.xml")

        for variant in mr.TRAIN_VARIANTS:
            run = mr.train(tmp, cfg, model, variant)
            np.savez(mr.GOLDEN / f"train_{variant}_stats.npz", **mr.load_stats(run))
            nets = {f: mr.checkpoint_summary(run / "checkpoint" / f"{f}.pth")
                    for f in ["agent_networks", "agent_networks_best"]}
            (mr.GOLDEN / f"train_{variant}_checkpoints.json").write_text(json.dumps(nets, indent=2) + "\n")

        summary = mr.summarize_test(mr.run_pretrained(tmp, cfg, model))
        (mr.GOLDEN / "pretrained_test.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote golden output to {mr.GOLDEN}")


if __name__ == "__main__":
    main()
