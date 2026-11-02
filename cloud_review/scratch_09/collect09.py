"""Package completed Colab evidence. All new per-wafer data stay in the private bundle."""
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def bundle(out, members):
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for src, name in members:
            archive.write(src, name)
    return {"file": out.name, "sha256": digest(out), "bytes": out.stat().st_size}


def gallery(train, path):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    spots = sorted((train / "fits/synthetic_s42").glob("spot*.npz"))
    assert len(spots) == 4
    fig, axes = plt.subplots(4, 3, figsize=(7, 8), constrained_layout=True)
    cmap = ListedColormap(["#f1f5f9", "#7cb3bc", "#b82544"])
    for i, src in enumerate(spots):
        with np.load(src, allow_pickle=False) as data:
            for ax, key in zip(axes[i], ("donor", "background", "synthetic")):
                ax.imshow(data[key], vmin=0, vmax=2, cmap=cmap, interpolation="nearest")
                ax.set_axis_off()
    for ax, title in zip(axes[0], ("Real Scratch donor", "Real normal background", "Synthetic Scratch")):
        ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def collect(root):
    train, gate, ev, qa = (root/n for n in ("training", "rf_gate", "evaluation", "audit09"))
    src = Path("/content")
    receipt = load(ev/"run_receipt.json")
    assert receipt["complete"] and receipt["official_test_used"] is False
    for name, h in receipt["files"].items():
        assert digest(ev/name) == h
    fits = sorted((train/"fits").glob("*/run.json"))
    assert len(fits) == (12 if load(gate/"select_receipt.json")["pseudo_train"] else 9)
    assert qa.is_dir() and any(qa.glob("*.csv"))
    pub = root/"public_results"
    for name, files in (
        ("evaluation", ev.glob("*")), ("audit09", qa.glob("*")),
        ("rf_gate", [gate/n for n in ("qualification.json", "confirmation.csv", "design_grid.csv",
                                      "design_freeze.json", "descriptive_by_size.csv", "select_receipt.json")]),
        ("training", [train/n for n in ("protocol.json", "runtime.json", "template_qa.json")]),
    ):
        dest = pub/name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            assert f.is_file() and f.suffix in {".md", ".csv", ".json"}
            shutil.copy2(f, dest/f.name)
    with (pub/"training/completion.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "epochs", "examples_per_epoch", "steps_per_epoch", "extra_slots_per_epoch", "checkpoint_sha256"])
        for path in fits:
            d = load(path)
            counts = {tuple(e[k] for k in ("examples", "steps", "substitutions")) for e in d["epochs"]}
            assert len(d["epochs"]) == 7 and len(counts) == 1
            assert digest(path.parent/"checkpoint.pt") == d["checkpoint_sha256"]
            w.writerow([d["model"], d["seed"], 7, *next(iter(counts)), d["checkpoint_sha256"]])
    gallery(train, root/"synthetic_examples.png")
    private = [(p, str(p.relative_to(root))) for d in (train, gate, ev, qa) for p in sorted(d.rglob("*")) if p.is_file()]
    private += [(p, p.name) for p in (root/"synthetic_examples.png", root/"run.log")]
    private += [(p, "scripts/"+p.name) for p in (src/n for n in ("train09.py", "pseudo09.py", "select_pseudo09.py",
                "evaluate09.py", "audit09.py", "collect09.py")) if p.exists()]
    public = [(p, str(p.relative_to(pub))) for p in sorted(pub.rglob("*")) if p.is_file()]
    output = [bundle(src/"WM811K_09_public_results.zip", public), bundle(src/"WM811K_09_private_evidence.zip", private)]
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    collect(Path(sys.argv[1]))
