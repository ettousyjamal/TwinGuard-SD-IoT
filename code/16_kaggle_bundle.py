"""Build the input bundle of the GPU training notebook (notebooks/TwinGuard_revision_training_kaggle.ipynb).

The bundle holds the model code, the selected hyperparameters, the graph windows of the public dataset, and the
simulator training data of the three topologies with their line-graph adjacency. Upload the zip to Kaggle as a
dataset, run the notebook on a GPU, and unpack its twinguard_outputs.zip into results/ (checkpoints of
results/sim*/models and results/matched_capacity/models).
Run from the repository root after the simulator data exist (code/06_sim_data.py, also with TG_TOPOLOGY=nsfnet and
TG_TOPOLOGY=two_tier):  python code/16_kaggle_bundle.py
Output: kaggle/twinguard_kaggle_inputs.zip
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "twinguard_kaggle_inputs"


def main():
    shutil.rmtree(OUT, ignore_errors=True); (OUT / "real").mkdir(parents=True)
    shutil.copy(ROOT / "twinguard" / "model.py", OUT / "twinguard_model.py")
    shutil.copy(ROOT / "results" / "best_hyperparameters.json", OUT)
    shutil.copy(ROOT / "data" / "processed" / "graph_pipeline.npz", OUT / "real")
    for t in ("dataset", "nsfnet", "two_tier"):
        src = ROOT / "data" / ("sim" if t == "dataset" else f"sim_{t}"); dst = OUT / f"sim_{t}"; dst.mkdir()
        for f in ("train.npz", "val.npz"):
            shutil.copy(src / f, dst)
        adj = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'code'); import numpy as np; "
                              "from code_sim_common import adjacency; np.save(sys.argv[1], adjacency())",
                              str(dst / "adjacency.npy")], cwd=ROOT, env={**os.environ, "TG_TOPOLOGY": t}, check=True)
    zip_path = shutil.make_archive(str(OUT), "zip", OUT)
    print("written", zip_path, f"{Path(zip_path).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
